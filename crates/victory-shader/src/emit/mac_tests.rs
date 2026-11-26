//! The Mac line's settings (`lower::mac`, `Options::early_depth`). These
//! tests look at the translation only and run on any host.

use super::container::SigElement;
use super::lower::{self, Mac};
use super::{Options, container, emit_with};
use crate::dxbc::{Container, FourCC};
use crate::native::{ComputeIo, HwStage, NativeProgram, StageIo};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
/// `DB_SHADER_CONTROL` of the title's alpha pre-pass: early then late Z
/// order (bit 4), kill (6), run without colour targets (10), depth before
/// the shader (12).
const PREPASS: u32 = 0x1450;
/// `SPI_PS_INPUT_ADDR`: perspective centre and the position's four floats,
/// as the pre-pass has them; and with the coverage mask.
const POSITION: u32 = 0xF02;
const COVERAGE: u32 = 1 << 14;

/// `v_mov_b32 v<dst>, <src>` (128 + k: the constant k; 256 + n: VGPR n).
const fn v_mov(dst: u8, src: u32) -> u32 {
    0x7E00_0200 | (dst as u32) << 17 | src
}

fn element(name: &str, kind: u8, comp: u8, row: Option<u32>, cols: u8) -> SigElement {
    SigElement {
        name: name.into(),
        index: 0,
        kind,
        comp,
        interp: 0,
        row,
        col: 0,
        cols,
        usage: 0,
        half: false,
    }
}

fn target() -> SigElement {
    element("SV_Target", 16, 9, Some(0), 4)
}

/// A pixel program: four constants to render target 0.
fn colour() -> Vec<u32> {
    let mut w: Vec<u32> = (0..4).map(|k| v_mov(k, 128 + u32::from(k))).collect();
    w.extend([0xF800_180F, 0x0302_0100]); // exp mrt0, v0, v1, v2, v3 done vm
    w.push(S_ENDPGM);
    w
}

/// A pixel program that also exports depth (`exp mrtz v0`), not last.
fn colour_and_depth() -> Vec<u32> {
    let mut w: Vec<u32> = (0..4).map(|k| v_mov(k, 128 + u32::from(k))).collect();
    w.extend([0xF800_0081, 0x0000_0000]); // exp mrtz v0, off, off, off
    w.extend([0xF800_180F, 0x0302_0100]);
    w.push(S_ENDPGM);
    w
}

fn pixel(words: &[u32], db_shader_control: u32, input_addr: u32) -> NativeProgram {
    NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 16,
        layout: vec![],
        used: vec![],
        io: StageIo::Ps {
            inputs: vec![],
            input_addr,
            db_shader_control,
            patches: vec![],
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn lowered(native: &NativeProgram, outputs: Vec<SigElement>, mac: Mac) -> lower::Lowered {
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, native).unwrap();
    lower::lower(&lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs,
        prims: vec![],
        mesh_stride: 0,
        atomic_locks: false,
        bias_as_gradients: false,
        host_wave32: false,
        narrow_waves: false,
        halves_local: Default::default(),
        mac,
        view_marks: false,
    })
    .unwrap()
}

const EARLY: Mac = Mac { early_depth: true };

/// The rule by itself: bit 12, and nothing that D3D's flag would change.
#[test]
fn the_early_depth_rule_is_bit_12_without_depth_stencil_or_coverage() {
    let rule = lower::early_depth_rule;
    let colours = [target()];
    // The title's four values with bit 12 (`notes/hair.md`).
    for control in [PREPASS, 0x1050, 0x1010, 0x1410] {
        assert!(rule(control, POSITION, &colours), "{control:#x}");
        assert!(
            rule(control, POSITION, &[]),
            "{control:#x}, no colour target"
        );
    }
    // Without bit 12 the title leaves the order to the hardware: no flag.
    for control in [0, 0x10, 0x50, 0x70, 0x200, 0x240, 0x10240] {
        assert!(!rule(control, POSITION, &colours), "{control:#x}");
    }
    // Depth, stencil and coverage exports the register enables.
    for bit in [0, 1, 2, 8] {
        assert!(!rule(PREPASS | 1 << bit, POSITION, &colours), "bit {bit}");
    }
    // The same as outputs of the translated program.
    for (name, kind) in [
        ("SV_Coverage", 14),
        ("SV_Depth", 17),
        ("SV_DepthLessEqual", 18),
        ("SV_DepthGreaterEqual", 19),
        ("SV_StencilRef", 20),
    ] {
        let outputs = [target(), element(name, kind, 9, None, 1)];
        assert!(!rule(PREPASS, POSITION, &outputs), "{name}");
    }
    // A program that reads the coverage mask.
    assert!(!rule(PREPASS, POSITION | COVERAGE, &colours));
}

/// The flag is set where the setting is on and the rule holds, and the
/// translation is byte for byte the same wherever either is not so.
#[test]
fn the_early_depth_flag_follows_the_setting_and_the_rule() {
    let off = Mac::default();
    let prepass = pixel(&colour(), PREPASS, POSITION);
    let with = lowered(&prepass, vec![target()], EARLY);
    let without = lowered(&prepass, vec![target()], off);
    assert!(with.early_depth);
    assert!(!without.early_depth);
    assert_ne!(with.dxil, without.dxil);
    // The setting off is the translation of a program without the bit.
    let plain = pixel(&colour(), 0x50, POSITION);
    assert_eq!(lowered(&plain, vec![target()], off).dxil, without.dxil);

    // The rule does not hold: no flag, the same bytes.
    let depth = [target(), element("SV_Depth", 17, 9, None, 1)];
    let cases: [(&str, NativeProgram, Vec<SigElement>); 4] = [
        ("no bit 12", plain, vec![target()]),
        (
            "a depth export",
            pixel(&colour_and_depth(), PREPASS | 1, POSITION),
            depth.to_vec(),
        ),
        (
            "a depth output the register does not name",
            pixel(&colour_and_depth(), PREPASS, POSITION),
            depth.to_vec(),
        ),
        (
            "a coverage input",
            pixel(&colour(), PREPASS, POSITION | COVERAGE),
            vec![target()],
        ),
    ];
    for (label, native, outputs) in cases {
        let with = lowered(&native, outputs.clone(), EARLY);
        assert!(!with.early_depth, "{label}");
        assert_eq!(with.dxil, lowered(&native, outputs, off).dxil, "{label}");
    }
}

/// Only pixel programs: a compute program is the same with the setting.
#[test]
fn early_depth_leaves_other_stages_alone() {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [1, 1, 1],
        }),
        code: [v_mov(0, 128), S_ENDPGM]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect(),
    };
    let with = lowered(&native, vec![], EARLY);
    assert!(!with.early_depth);
    assert_eq!(with.dxil, lowered(&native, vec![], Mac::default()).dxil);
}

/// `Options::early_depth` reaches the lowering through `emit_with`.
#[test]
fn the_early_depth_option_reaches_the_translation() {
    let native = pixel(&colour(), PREPASS, POSITION);
    let osg1 = container::signature(&[target()], false);
    let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
    let c = Container::parse(&bytes).unwrap();
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let emit = |options| emit_with(&c, &native, &p, &st, &io, &res, options).unwrap();
    let off = emit(Options::default());
    let on = emit(Options {
        early_depth: true,
        ..Default::default()
    });
    assert_ne!(on.dxil, off.dxil);
    // The flag has no `SFI0` feature bit: the two differ in the module only.
    let part = |dxil: &[u8], cc| {
        Container::parse(dxil)
            .unwrap()
            .part(cc)
            .unwrap()
            .data
            .to_vec()
    };
    for cc in [FourCC::SFI0, FourCC::OSG1, FourCC::PSV0] {
        assert_eq!(part(&on.dxil, cc), part(&off.dxil, cc));
    }
    // dxil.dll, where there is one, checks the flag against the module.
    #[cfg(windows)]
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        v.validate(&mut on.dxil.clone()).unwrap();
    }
}
