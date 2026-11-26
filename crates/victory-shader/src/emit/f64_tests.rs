//! `v_cvt_f32_f64` on any host. Its GPU test is in `valu_tests`, which needs
//! a D3D12 device (Windows). Here every operand is a constant of the
//! program, so the translation's integer steps are over constants alone:
//! the test works them out on the CPU ([`Module::constant_call_args`], which
//! also refuses a shift LLVM leaves undefined) and compares the result with
//! the host CPU's own conversion.
//!
//! [`Module::constant_call_args`]: super::bitcode::Module

use super::lower;
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, range, resource};

const S_ENDPGM: u32 = 0xBF81_0000;
/// Operand codes: a literal dword follows; VGPR n; the constant 0.
const LITERAL: u32 = 255;
const V1: u32 = 256 + 1;
/// `v_cvt_f32_f64` as VOP1, and as VOP3 (0x180 more).
const CVT: u32 = 0x00f;

/// `s_mov_b32 s<dst>, <src>`.
const fn s_mov(dst: u32, src: u32) -> u32 {
    0xBE80_0300 | dst << 16 | src
}

/// VOP1 `op v<dst>, <src>`.
const fn vop1(op: u32, dst: u32, src: u32) -> u32 {
    0x7E00_0000 | dst << 17 | op << 9 | src
}

/// VOP3 `op v<dst>, <src>` of a VOP1 operation, with `abs` and `neg` of the
/// operand.
const fn vop3(op: u32, dst: u32, src: u32, abs: bool, neg: bool) -> [u32; 2] {
    [
        0x35 << 26 | (0x180 + op) << 16 | (abs as u32) << 8 | dst,
        src | (neg as u32) << 29,
    ]
}

/// One conversion: the words that leave its result in v0.
#[derive(Clone, Copy)]
enum Form {
    /// Of v[1:2], set from literals.
    Vector(u64),
    /// Of s[2:3], set from literals.
    Scalar(u64),
    /// Of v[1:2] with the operand's absolute value taken, negated, or both.
    Modified(u64, bool, bool),
    /// Of an operand code by itself: an inline constant, or a literal
    /// (which is the high dword).
    Operand(u32, Option<u32>),
}

fn words(form: Form) -> Vec<u32> {
    let pair = |x: u64| [x as u32, (x >> 32) as u32];
    match form {
        Form::Vector(x) | Form::Modified(x, ..) => {
            let [lo, hi] = pair(x);
            let mut w = vec![vop1(1, 1, LITERAL), lo, vop1(1, 2, LITERAL), hi];
            match form {
                Form::Modified(_, abs, neg) => w.extend(vop3(CVT, 0, V1, abs, neg)),
                _ => w.push(vop1(CVT, 0, V1)),
            }
            w
        }
        Form::Scalar(x) => {
            let [lo, hi] = pair(x);
            vec![
                s_mov(2, LITERAL),
                lo,
                s_mov(3, LITERAL),
                hi,
                vop1(CVT, 0, 2),
            ]
        }
        Form::Operand(code, literal) => {
            let mut w = vec![vop1(CVT, 0, code)];
            w.extend(literal);
            w
        }
    }
}

/// What the translation stores for each of `forms`, in order: a one-thread
/// compute program that runs each and stores v0 to its own dword of the raw
/// unordered view of the table at user data 4. `None` where the stored
/// value is not made from constants alone.
fn converted(forms: &[Form]) -> Vec<Option<u32>> {
    assert!(forms.len() <= 1024);
    let mut code = vec![
        s_mov(5, 128), // s[4:5]: the table's address
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
    ];
    for (i, &form) in forms.iter().enumerate() {
        code.extend(words(form));
        // buffer_store_dword v0, off, s[8:11], 0 offset:4 * i
        code.extend([0xE070_0000 | 4 * i as u32, 0x8002_0000]);
    }
    code.push(S_ENDPGM);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 4,
            slot: 1,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [1, 1, 1],
        }),
        code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let (p, st) = range::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = st.unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
        atomic_locks: false,
        bias_as_gradients: false,
        host_wave32: false,
        narrow_waves: false,
        halves_local: Default::default(),
        mac: Default::default(),
        view_marks: false,
    })
    .unwrap();
    assert_eq!(lowered.stored_constants.len(), forms.len());
    lowered.stored_constants
}

/// The 32-bit float nearest to `x` (ties to even), as bits: the CPU's own
/// conversion, and for a NaN its sign and the top of its payload, quiet.
fn nearest(x: u64) -> u32 {
    let f = f64::from_bits(x);
    if f.is_nan() {
        let sign = ((x >> 63) as u32) << 31;
        return sign | 0x7FC0_0000 | (x >> 29) as u32 & 0x007F_FFFF;
    }
    (f as f32).to_bits()
}

fn check(values: &[u64], form: impl Fn(u64) -> Form, want: impl Fn(u64) -> u32) {
    for chunk in values.chunks(256) {
        let forms: Vec<Form> = chunk.iter().map(|&x| form(x)).collect();
        for (&x, got) in chunk.iter().zip(converted(&forms)) {
            let want = want(x);
            assert_eq!(
                got,
                Some(want),
                "v_cvt_f32_f64({x:#018x} = {:e}): want {want:#010x}",
                f64::from_bits(x),
            );
        }
    }
}

/// The values the fork's GPU test has, and the ends of every range the
/// lowering tells apart.
fn edges() -> Vec<u64> {
    let mut v: Vec<u64> = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        3.5,
        -123_456.75,
        f64::from(f32::MAX),
        f64::from(f32::MAX) * 1.000_001,
        -f64::from(f32::MAX) * 1.000_001,
        1e300,
        f64::from(f32::MIN_POSITIVE),
        f64::from(f32::MIN_POSITIVE) * 0.5,
        f64::from(f32::MIN_POSITIVE) * 0.75,
        f64::from(f32::from_bits(0x007F_FFFF)),
        f64::from(f32::from_bits(1)),
        1e-200,
        -1e-200,
        f64::MIN_POSITIVE,
        f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ]
    .iter()
    .map(|x: &f64| x.to_bits())
    .collect();
    v.extend([
        0x3FF0_0000_1000_0000, // 1 + half a step: to even, 1
        0x3FF0_0000_3000_0000, // 1 + a step and a half: to even, 1 + 2 steps
        0x3FF0_0000_1000_0001, // just over half a step: up
        0x3FF0_0000_0FFF_FFFF, // just under: down
        0x47EF_FFFF_F000_0000, // the largest float + half a step: infinity
        0x47EF_FFFF_EFFF_FFFF, // just under: the largest float
        0x380F_FFFF_F000_0000, // the largest subnormal + half a step: the least normal
        0x3690_0000_0000_0000, // half the least subnormal: to even, 0
        0x3690_0000_0000_0001, // just over: the least subnormal
        0xB690_0000_0000_0001,
        0x368F_FFFF_FFFF_FFFF, // just under: 0
        0x36A8_0000_0000_0000, // a subnormal and a half: to even, 2
        0x0000_0000_0000_0001, // the least 64-bit subnormal
        0x000F_FFFF_FFFF_FFFF,
        0x7FF8_0000_0000_0000, // NaNs: quiet, signalling, with a sign, payloads
        0x7FF0_0000_0000_0001,
        0xFFF0_0000_2000_0000,
        0xFFF4_5678_9ABC_DEF0,
        0x7FFF_FFFF_FFFF_FFFF,
    ]);
    v
}

/// Reproducible values: `n` from each of the lowering's exponent ranges,
/// with fractions that are random, ties, and next to ties.
fn spread(n: usize) -> Vec<u64> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    // Exponents: all, and around 32-bit subnormals, 1, the largest float,
    // 64-bit subnormals, infinity.
    let ranges = [
        (0u64, 2047u64),
        (870, 900),
        (1020, 1026),
        (1146, 1153),
        (0, 2),
        (2045, 2047),
    ];
    let mut out = Vec::new();
    for (low, high) in ranges {
        for i in 0..n {
            let r = next();
            let exponent = low + (r >> 52) % (high - low + 1);
            let mut fraction = next() & 0x000F_FFFF_FFFF_FFFF;
            match i % 4 {
                // A tie of the normal range, and its neighbours.
                1 => fraction = fraction & !0x1FFF_FFFF | 0x1000_0000,
                2 => fraction = (fraction & !0x1FFF_FFFF | 0x1000_0000) + (r & 2) - 1,
                // Few bits set: ties of the subnormal range.
                3 => fraction &= next() & next(),
                _ => {}
            }
            out.push(r & 1 << 63 | exponent << 52 | fraction);
        }
    }
    out
}

#[test]
fn v_cvt_f32_f64_gives_the_nearest_float_in_integer_steps() {
    let edges = edges();
    check(&edges, Form::Vector, nearest);
    check(&edges, Form::Scalar, nearest);
    check(&spread(2048), Form::Vector, nearest);
}

#[test]
fn v_cvt_f32_f64_takes_the_absolute_value_then_the_negative() {
    let mut values = edges();
    values.extend(spread(64));
    const SIGN: u64 = 1 << 63;
    check(
        &values,
        |x| Form::Modified(x, true, false),
        |x| nearest(x & !SIGN),
    );
    check(
        &values,
        |x| Form::Modified(x, false, true),
        |x| nearest(x ^ SIGN),
    );
    check(
        &values,
        |x| Form::Modified(x, true, true),
        |x| nearest(x | SIGN),
    );
    check(&values, |x| Form::Modified(x, false, false), nearest);
}

#[test]
fn v_cvt_f32_f64_reads_inline_constants_and_a_literal_as_the_high_dword() {
    let inline = [
        (128, 0.0f32),
        (240, 0.5),
        (241, -0.5),
        (242, 1.0),
        (243, -1.0),
        (244, 2.0),
        (245, -2.0),
        (246, 4.0),
        (247, -4.0),
        (248, (1.0 / std::f64::consts::TAU) as f32),
    ];
    let mut forms: Vec<Form> = inline
        .iter()
        .map(|&(c, _)| Form::Operand(c, None))
        .collect();
    let mut want: Vec<u32> = inline.iter().map(|&(_, f)| f.to_bits()).collect();
    for x in [100.0f64, -0.375, 65_536.0] {
        assert_eq!(x.to_bits() as u32, 0);
        forms.push(Form::Operand(LITERAL, Some((x.to_bits() >> 32) as u32)));
        want.push((x as f32).to_bits());
    }
    let got = converted(&forms);
    for (i, (got, want)) in got.into_iter().zip(want).enumerate() {
        assert_eq!(got, Some(want), "operand {i}");
    }
}
