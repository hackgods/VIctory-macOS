//! A descriptor's dwords read as data (`lower::mac`,
//! `Mac::descriptor_as_data`, `Options::descriptor_as_data`): the console
//! compiler reads a buffer's size out of its V# instead of asking for it,
//! and the host has that dword in the view table it fills for the others;
//! a load through a root descriptor's address reads the resource's own
//! memory.
//!
//! Hand-built GFX10 programs: a table load of a V# whose dword 2 (or 3) a
//! thread copies out, and a scalar load through a root descriptor. Without
//! the setting each is refused; with it the dwords the source answers are
//! the translation's, and dword 3, which no source answers, still stops the
//! program.
//!
//! From ScarlettMac 4921b7e6 (its `emit/descriptor_as_data_tests.rs`), as
//! it is.

use super::lower::{self, Mac};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const S_WAITCNT_LGKM: u32 = 0xBF8C_C07F; // s_waitcnt lgkmcnt(0)
const NULL_SGPR: u32 = 125; // s0 (SOFFSET's null: the load's soffset)

/// `s_mov_b32 s<sdst>, s<src0>` (128 + k: the constant k).
const fn s_mov(sdst: u32, src0: u32) -> u32 {
    0xBE80_0000 | sdst << 16 | 0x03 << 8 | src0
}

/// `s_load_dwordx4 s[sdst:+4], s[sbase:+2], offset`.
const fn s_load_x4(sdst: u32, sbase: u32, offset: u32) -> [u32; 2] {
    [
        0xF400_0000 | 2 << 18 | sdst << 6 | (sbase >> 1),
        NULL_SGPR << 25 | offset,
    ]
}

/// `s_load_dwordx2 s[sdst:+2], s[sbase:+2], offset`.
const fn s_load_x2(sdst: u32, sbase: u32, offset: u32) -> [u32; 2] {
    [
        0xF400_0000 | 1 << 18 | sdst << 6 | (sbase >> 1),
        NULL_SGPR << 25 | offset,
    ]
}

/// `v_mov_b32 v<dst>, s<src>` (128 + k: the constant k).
const fn v_mov(dst: u32, src: u32) -> u32 {
    0x7E00_0200 | dst << 17 | src
}

/// `buffer_store_dword v<vdata>, v16, s[sbase:+4], 0 offen offset:<offset>`.
const fn store(vdata: u32, sbase: u32, offset: u32) -> [u32; 2] {
    [0xE070_1000 | offset, 0x8000_0010 | (sbase >> 2) << 16 | vdata << 8]
}

/// The table in `s0`: the V# at dword 0, the raw buffer the threads store
/// to at dword 8. A wave32 program of 32 threads that copies the V#'s dword
/// `dword` (0-3) to its own four bytes.
fn size_out_of_the_descriptor(dword: u32) -> Vec<u32> {
    let mut w = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
    ];
    w.extend(s_load_x4(8, 4, 0)); // s_load_dwordx4 s[8:11], s[4:5], 0
    w.extend(s_load_x4(40, 4, 0x20)); // s_load_dwordx4 s[40:43], s[4:5], 0x20
    w.extend([
        S_WAITCNT_LGKM,
        0x3420_0085,            // v_lshlrev_b32 v16, 5, v0
        v_mov(1, 8 + dword),    // v_mov_b32 v1, s<8 + dword>
    ]);
    w.extend(store(1, 40, 0));
    w.push(S_ENDPGM);
    w
}

/// A root descriptor's dwords at a title address, its two halves in
/// `s[0:1]`, and the table in `s[2:3]` with a raw UAV at dword 0: the
/// wave32 program of 32 threads reads eight bytes of the resource at +4 and
/// stores what it read to four bytes of its own.
fn root_descriptor_dwords() -> Vec<u32> {
    let mut w = vec![s_mov(3, 0x80)]; // s_mov_b32 s3, 0
    w.extend(s_load_x2(8, 0, 4)); // s_load_dwordx2 s[8:9], s[0:1], 4
    w.extend(s_load_x4(12, 2, 0)); // s_load_dwordx4 s[12:15], s[2:3], 0
    w.extend([
        S_WAITCNT_LGKM,
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        v_mov(1, 8), // v_mov_b32 v1, s8
    ]);
    w.extend(store(1, 12, 0));
    w.push(S_ENDPGM);
    w
}

/// The table in `s0`: the V# at dword 0, the raw buffer beside it at
/// dword 8.
fn table_layout() -> Vec<RootParam> {
    vec![RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data: 0,
        slot: 0,
    }]
}

fn table_used() -> Vec<UsedResource> {
    let range = |kind, table_offset_dwords, dwords| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data: 0,
        slot: 0,
    };
    vec![range(ResourceType::Srv, 0, 4), range(ResourceType::Uav, 8, 4)]
}

/// A compute program of 32 threads.
fn native(code: Vec<u32>, layout: Vec<RootParam>, used: Vec<UsedResource>) -> NativeProgram {
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [32, 1, 1],
        }),
        code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// The translation's lowering, or the reason it was stopped.
fn lowered(native: &NativeProgram, descriptor_as_data: bool) -> Result<lower::Lowered, String> {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, native).expect("io::raise");
    lower::lower(&lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        view_marks: false,
        atomic_locks: false,
        atomic_lock_rounds: false,
        atomic_umax: false,
        bias_as_gradients: false,
        host_wave32: false,
        narrow_waves: false,
        halves_local: Default::default(),
        mac: Mac {
            descriptor_as_data,
            ..Mac::default()
        },
    })
    .map_err(|e| e.to_string())
}

/// A V#'s dword 2, its record count, is what the program reads: refused by
/// default, from the host's view table with the setting (which the program
/// then has bound).
#[test]
fn a_buffers_record_count_read_as_data_needs_the_setting() {
    let program = native(
        size_out_of_the_descriptor(2),
        table_layout(),
        table_used(),
    );
    let refused = match lowered(&program, false) {
        Ok(_) => panic!("translated without the setting"),
        Err(why) => why,
    };
    assert!(
        refused.contains("a descriptor read only as data"),
        "{refused}"
    );
    let made = lowered(&program, true).expect("translated with the setting");
    assert!(made.view_addresses, "the view addresses are read");
    // Dword 1's stride and swizzle bits read 0 in the table: the shortcut
    // of `Lower::view_addresses`.
    assert!(!made.dxil.is_empty());
}

/// A V#'s dword 3 has no host value (its console format words are not the
/// D3D12 view's): the program is stopped with or without the setting,
/// rather than given a made-up dword.
#[test]
fn a_buffers_format_word_read_as_data_is_still_refused() {
    let program = native(
        size_out_of_the_descriptor(3),
        table_layout(),
        table_used(),
    );
    for descriptor_as_data in [false, true] {
        let refused = match lowered(&program, descriptor_as_data) {
            Ok(_) => panic!("translated with descriptor_as_data = {descriptor_as_data}"),
            Err(why) => why,
        };
        assert!(
            refused.contains("a descriptor read only as data"),
            "{refused}"
        );
    }
}

/// A load through a root descriptor's address reads the resource itself:
/// refused by default, a title memory read with the setting (the page
/// table, not the view addresses).
#[test]
fn a_root_descriptors_dwords_are_its_memory() {
    let range = |table_offset_dwords| UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 2,
        slot: 1,
    };
    let layout = vec![
        RootParam {
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            sgprs: 2,
            base_register: 0,
            user_data: 0,
            slot: 0,
        },
        RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 2,
            slot: 1,
        },
    ];
    let program = native(root_descriptor_dwords(), layout, vec![range(0)]);
    let refused = match lowered(&program, false) {
        Ok(_) => panic!("translated without the setting"),
        Err(why) => why,
    };
    assert!(
        refused.contains("a descriptor read only as data"),
        "{refused}"
    );
    let made = lowered(&program, true).expect("translated with the setting");
    assert!(made.memory, "the resource's memory is read");
    assert!(!made.view_addresses, "no view address is read");
}
