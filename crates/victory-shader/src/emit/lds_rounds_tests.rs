//! LDS that a group of one console wave shares round after round of a
//! loop, with lanes that leave on the way (`lower::lds_rounds`,
//! `Options::lds_rounds`; `notes/replay-determinism.md`, "HairOnly, 27
//! November"). The translation tests and the model run on any host. The
//! GPU test (Windows) runs the programs in halves on every hardware adapter
//! and compares with the console's values.
//!
//! The programs have the shape of the title's hair shadow-mask program
//! (`0a5ecfee6b79e13b`): lanes leave by an early-out, the others go round a
//! loop whose count every lane reads from a value the whole group holds
//! alike, and in each round thread 0 alone writes an LDS dword that every
//! lane reads right after.

use super::halves_local_tests::{PROLOGUE, S_ENDPGM, V16_IS_32_TIMES_V0, native, sop1, store};
use super::lower::{self, HalvesLocal, Mac};
use crate::{gfx10, io, ir, resource, structure};

const S_WAITCNT_LGKM: u32 = 0xBF8C_C07F;
const EXEC: u8 = 126;
const VCC: u8 = 106;
/// The race's own refusal, and the LDS order's.
const REFUSED: &str = "LDS one console wave64 shares without barriers, on a host of 32-lane waves";

/// `v_mov_b32 v<dst>, <k>`.
const fn v_mov(dst: u8, k: u32) -> u32 {
    0x7E00_0200 | (dst as u32) << 17 | (128 + k)
}
/// `v_cmpx_gt_u32 exec, <k>, v0`: the lanes below `k` stay.
const fn below(k: u32) -> u32 {
    0x7DA8_0000 | (128 + k)
}
const fn execz(n: u32) -> u32 {
    0xBF88_0000 | n
}
const fn scc1(n: u32) -> u32 {
    0xBF85_0000 | n
}
const fn sop2(op: u32, dst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000 | op << 23 | (dst as u32) << 16 | (src1 as u32) << 8 | src0 as u32
}
/// A DS instruction: `op` on the dword at `v<addr>`, data `v<data>`,
/// result `v<dst>`.
const fn ds(op: u32, addr: u8, data: u8, dst: u8) -> [u32; 2] {
    [
        0xD800_0000 | op << 18,
        (dst as u32) << 24 | (data as u32) << 8 | addr as u32,
    ]
}
const DS_WRITE: u32 = 13;
const DS_READ: u32 = 54;

/// The start: each lane stores 0 (a lane that leaves keeps it) and EXEC on
/// entry is kept in `s[88:89]`, as the title's program keeps it.
fn start() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([V16_IS_32_TIMES_V0, v_mov(1, 0)]);
    w.extend(store(1, 0));
    w.push(sop1(88, 0x04, EXEC)); // s_mov_b64 s[88:89], exec
    w
}

/// Three rounds (`s20` from 0 while below `s21`): thread 0 alone writes
/// `s20 + 60` to LDS dword 0, every lane adds what it reads to `v1`; then
/// every lane stores `v1`. 183 in every lane that goes round on the
/// console.
fn rounds() -> Vec<u32> {
    let mut w = looped();
    w.extend(store(1, 0));
    w
}

/// The loop of [`rounds`] alone.
fn looped() -> Vec<u32> {
    let mut w = vec![sop1(20, 0x03, 128)]; // s_mov_b32 s20, 0
    let top = w.len();
    w.extend([
        0x7D84_0080,        // v_cmp_eq_u32 vcc, 0, v0
        sop1(2, 0x24, VCC), // s_and_saveexec_b64 s[2:3], vcc
        execz(5),
        sop2(0, 22, 20, 128 + 60), // s_add_u32 s22, s20, 60
        0x7E0C_0216,               // v_mov_b32 v6, s22
        v_mov(7, 0),
    ]);
    w.extend(ds(DS_WRITE, 7, 6, 0)); // ds_write_b32 v7, v6
    w.extend([sop1(EXEC, 0x04, 2), v_mov(7, 0)]); // s_mov_b64 exec, s[2:3]
    w.extend(ds(DS_READ, 7, 0, 8)); // ds_read_b32 v8, v7
    w.extend([
        S_WAITCNT_LGKM,
        0x4A02_0308, // v_add_nc_u32 v1, v8, v1
        0x8014_8114, // s_add_u32 s20, s20, 1
        0xBF0A_1514, // s_cmp_lt_u32 s20, s21
    ]);
    w.push(scc1(0x1_0000 - (w.len() - top + 1) as u32));
    w
}

/// The lanes below `stay` go round, the count 3 read with
/// `v_readfirstlane` from a VGPR every lane holds alike.
fn broadcast(stay: u32) -> Vec<u32> {
    let body = [
        v_mov(5, 3),
        0x7E2A_0505, // v_readfirstlane_b32 s21, v5
    ];
    behind_an_early_out(stay, &body)
}

/// `start`, the early-out to the end, `count` (which leaves the loop's
/// count in `s21`), the rounds.
fn behind_an_early_out(stay: u32, count: &[u32]) -> Vec<u32> {
    program(stay, count, false)
}

/// [`behind_an_early_out`], with `outer` the rounds inside a loop of two
/// rounds the group goes round together (`s24`), so that leaving them does
/// not end the program: then the rounds' own count must be the group's.
fn program(stay: u32, count: &[u32], outer: bool) -> Vec<u32> {
    let mut w = start();
    let mut rest: Vec<u32> = count.to_vec();
    if outer {
        rest.push(sop1(24, 0x03, 128)); // s_mov_b32 s24, 0
        let top = rest.len();
        rest.extend(looped());
        rest.extend([0x8018_8118, 0xBF0A_8218]); // s_add_u32 s24, s24, 1; s_cmp_lt_u32 s24, 2
        rest.push(scc1(0x1_0000 - (rest.len() - top + 1) as u32));
        rest.extend(store(1, 0));
    } else {
        rest.extend(rounds());
    }
    w.extend([below(stay), execz(rest.len() as u32)]);
    w.extend(rest);
    w.push(S_ENDPGM);
    w
}

/// The count from the tile's coordinate, as the title computes it: `(s0 <<
/// k) + tid >> k`, read at the first lane of EXEC (`s_ff1` of `s[88:89] &
/// exec`), then made 3 from it (`s - s + 3`), for rounds inside a loop the
/// group goes round together. Every lane of a group of 64 holds it alike
/// with `k` = 6; with 5 the thread id's bit 5 stays in.
fn coordinate(k: u32) -> Vec<u32> {
    let count = [
        0xD746_0003, // v_lshl_add_u32 v3, s0, k, v0
        0x0401_0000 | (128 + k) << 9,
        0x2C08_0600 | (128 + k), // v_lshrrev_b32 v4, k, v3
        sop2(15, 24, 88, EXEC),  // s_and_b64 s[24:25], s[88:89], exec
        sop1(26, 20, 24),        // s_ff1_i32_b64 s26, s[24:25]
        0xD760_0015,             // v_readlane_b32 s21, v4, s26
        0x0000_3504,
        sop2(1, 23, 21, 21),      // s_sub_u32 s23, s21, s21
        sop2(0, 21, 23, 128 + 3), // s_add_u32 s21, s23, 3
    ];
    program(40, &count, true)
}

/// The loop counted to each host wave's first lane's thread id (0 or 32,
/// at least once), so the host waves go round it apart, inside a loop of
/// two rounds the group goes round together (so that leaving it does not
/// end the program).
fn each_wave_counts() -> Vec<u32> {
    program(64, &[0x7E2A_0500], true) // v_readfirstlane_b32 s21, v0
}

/// Thread 0 reads the dword again in its own branch, right after its write:
/// a read in code only some lanes run of what such code wrote since the
/// last barrier, which no barrier can order (refused as
/// [`lower::lds_order`]'s barriers refuse it).
fn read_where_only_thread_0_runs() -> Vec<u32> {
    let mut w = broadcast(64);
    // In thread 0's branch: `ds_read_b32 v9, v7` after its write.
    let write = w
        .windows(2)
        .position(|p| p == ds(DS_WRITE, 7, 6, 0))
        .expect("the write");
    w.splice(write + 2..write + 2, ds(DS_READ, 7, 0, 9));
    // Its skip grows by the read's two words.
    let skip = w[..write]
        .iter()
        .rposition(|&x| x == execz(5))
        .expect("skip");
    w[skip] = execz(7);
    // The loop's branch back, and the early-out's, by two more words.
    let back = w.iter().rposition(|&x| x >> 16 == 0xBF85).expect("back");
    w[back] -= 2;
    let out = w.iter().position(|&x| x >> 16 == 0xBF88).expect("out");
    w[out] += 2;
    w
}

fn text(words: &[u32]) -> String {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    (gfx10::decode(&code).expect("decodes").iter())
        .map(|i| i.text(true))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn the_test_programs_are_what_their_comments_say() {
    let all = text(&broadcast(40));
    for want in [
        "s_mov_b64 s[88:89], exec\nv_cmpx_gt_u32 exec, 40, v0\ns_cbranch_execz",
        "v_mov_b32 v5, 3\nv_readfirstlane_b32 s21, v5\ns_mov_b32 s20, 0\n\
         v_cmp_eq_u32 vcc, 0, v0\ns_and_saveexec_b64 s[2:3], vcc\ns_cbranch_execz",
        "s_add_u32 s22, s20, 60\nv_mov_b32 v6, s22\nv_mov_b32 v7, 0\nds_write_b32 v7, v6\n\
         s_mov_b64 exec, s[2:3]\nv_mov_b32 v7, 0\nds_read_b32 v8, v7\n",
        "v_add_nc_u32 v1, v8, v1\ns_add_u32 s20, s20, 1\ns_cmp_lt_u32 s20, s21\ns_cbranch_scc1",
    ] {
        assert!(all.contains(want), "{want:?} not in:\n{all}");
    }
    let all = text(&coordinate(6));
    for want in ["v_lshl_add_u32 v3, s0, 6, v0\nv_lshrrev_b32 v4, 6, v3\n\
         s_and_b64 s[24:25], s[88:89], exec\ns_ff1_i32_b64 s26, s[24:25]\n\
         v_readlane_b32 s21, v4, s26\ns_sub_u32 s23, s21, s21\ns_add_u32 s21, s23, 3"]
    {
        assert!(all.contains(want), "{want:?} not in:\n{all}");
    }
    assert!(text(&coordinate(5)).contains("v_lshrrev_b32 v4, 5, v3"));
    assert!(
        text(&each_wave_counts())
            .contains("v_readfirstlane_b32 s21, v0\ns_mov_b32 s24, 0\ns_mov_b32 s20, 0")
    );
    assert!(
        text(&each_wave_counts())
            .contains("s_add_u32 s24, s24, 1\ns_cmp_lt_u32 s24, 2\ns_cbranch_scc1")
    );
    let all = text(&read_where_only_thread_0_runs());
    assert!(
        all.contains("ds_write_b32 v7, v6\nds_read_b32 v9, v7\ns_mov_b64 exec, s[2:3]"),
        "{all}"
    );
    // Every program ends where its branches say: the skips land on the
    // restore and on the end.
    for words in [
        broadcast(40),
        coordinate(6),
        each_wave_counts(),
        read_where_only_thread_0_runs(),
    ] {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = ir::build(&code, true, 32).expect("ir::build");
        assert!(structure::structure(&p).is_ok());
        let last = p.blocks.last().expect("blocks");
        assert_eq!(last.insts.last().map(|i| i.name()), Some("s_endpgm"));
        assert!(
            p.blocks
                .iter()
                .all(|b| b.exit.successors().iter().all(|&s| s < p.blocks.len()))
        );
    }
}

const RACE: HalvesLocal = HalvesLocal {
    reads: false,
    lds: true,
};

fn with_rounds(mac: Mac) -> Mac {
    Mac {
        lds_rounds: true,
        ..mac
    }
}

/// `words` as a wave64 compute program of one row of 64 threads with 512
/// bytes of LDS, lowered for a host of 32-lane waves.
fn lowered(
    words: &[u32],
    mac: Mac,
    local: HalvesLocal,
) -> Result<lower::Lowered, crate::ShaderError> {
    let native = native(words, 64, 512);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    lower::lower(&parts(&native, &p, &st, &io, &res, mac, local))
}

fn parts<'a>(
    native: &'a crate::native::NativeProgram,
    p: &'a ir::Program,
    st: &'a structure::Structured,
    io: &'a io::Io,
    res: &'a resource::Resources,
    mac: Mac,
    halves_local: HalvesLocal,
) -> lower::Parts<'a> {
    lower::Parts {
        native,
        p,
        st,
        io,
        res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        view_marks: false,
        atomic_locks: false,
        atomic_lock_rounds: false,
        atomic_umax: false,
        bias_as_gradients: false,
        narrow_waves: false,
        halves_local,
        mac,
        host_wave32: true,
    }
}

/// The barriers the form places, as the instruction's name or "end" for a
/// block's end, in program order; `None` where it places none.
fn placed(words: &[u32]) -> Option<Vec<String>> {
    let native = native(words, 64, 512);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let x = parts(
        &native,
        &p,
        &st,
        &io,
        &res,
        with_rounds(Mac::default()),
        HalvesLocal::default(),
    );
    let mut at: Vec<_> = lower::lds_rounds::wanted(&x, 64, 128)?
        .into_iter()
        .collect();
    at.sort_unstable_by_key(|&(b, i)| (p.blocks[b].offset, i));
    Some(
        at.into_iter()
            .map(|(b, i)| match p.blocks[b].insts.get(i) {
                Some(inst) => inst.name().to_owned(),
                None => "end".to_owned(),
            })
            .collect(),
    )
}

fn outcome(words: &[u32], mac: Mac, local: HalvesLocal) -> String {
    match lowered(words, mac, local) {
        Ok(l) => format!("translated, {} bytes", l.dxil.len()),
        Err(e) => e.to_string(),
    }
}

/// The broadcast in a loop behind an early-out: refused before (the race's
/// own form translates it unsynced), and with the setting translated with
/// nothing left to a half: a barrier before the read every lane makes, and
/// one at the end of its block (the next round's write is in thread 0's
/// branch). The same with no lane leaving, some lanes of the upper half,
/// or all of them.
#[test]
fn the_broadcast_in_a_loop_gets_its_barriers() {
    for stay in [64, 40, 32] {
        let words = broadcast(stay);
        assert!(outcome(&words, Mac::default(), HalvesLocal::default()).contains(REFUSED));
        let order = Mac {
            lds_order: true,
            ..Mac::default()
        };
        assert!(outcome(&words, order, HalvesLocal::default()).contains(REFUSED));
        assert!(outcome(&words, Mac::default(), RACE).starts_with("translated"));
        let ours = outcome(&words, with_rounds(Mac::default()), HalvesLocal::default());
        assert!(ours.starts_with("translated"), "{stay}: {ours}");
        // The race's switch does not matter once the LDS is ordered.
        assert_eq!(ours, outcome(&words, with_rounds(Mac::default()), RACE));
        assert_eq!(
            placed(&words).as_deref(),
            Some(&["ds_read_b32".to_owned(), "end".to_owned()][..])
        );
    }
}

/// The count from the tile's coordinate read at EXEC's first lane is the
/// group's: met, ordered (with half-local reads, which that read needs in
/// halves). With a shift that leaves a bit of the thread id in, it is not.
#[test]
fn a_coordinate_with_the_thread_id_shifted_out_is_the_groups() {
    let local = HalvesLocal {
        reads: true,
        lds: false,
    };
    let good = coordinate(6);
    assert!(placed(&good).is_some());
    assert!(outcome(&good, with_rounds(Mac::default()), local).starts_with("translated"));
    assert!(outcome(&good, Mac::default(), local).contains(REFUSED));
    let bad = coordinate(5);
    assert_eq!(placed(&bad), None);
    assert!(outcome(&bad, with_rounds(Mac::default()), local).contains(REFUSED));
}

/// What has no such order stays as it was: a loop each host wave counts
/// for itself (inside one the group goes round together), and a read in
/// thread 0's branch of what was written there.
#[test]
fn what_has_no_order_across_rounds_stays_as_it_was() {
    for words in [each_wave_counts(), read_where_only_thread_0_runs()] {
        assert_eq!(placed(&words), None);
        for local in [HalvesLocal::default(), RACE] {
            assert_eq!(
                outcome(&words, with_rounds(Mac::default()), local),
                outcome(&words, Mac::default(), local)
            );
        }
    }
}

/// The setting changes nothing where the barriers of every lane already
/// order the LDS, or the LDS order does.
#[test]
fn the_setting_leaves_ordered_programs_as_they_were() {
    // No loop, no early-out: thread 0 writes, every lane reads, in blocks
    // every lane runs once but the write's.
    let mut once = start();
    once.extend([
        0x7D84_0080,
        sop1(2, 0x24, VCC),
        execz(4),
        v_mov(6, 61),
        v_mov(7, 0),
    ]);
    once.extend(ds(DS_WRITE, 7, 6, 0));
    once.extend([sop1(EXEC, 0x04, 2), v_mov(7, 0)]);
    once.extend(ds(DS_READ, 7, 0, 8));
    once.extend([S_WAITCNT_LGKM, 0x4A02_0308]);
    once.extend(store(1, 0));
    once.push(S_ENDPGM);
    let mac = Mac {
        lds_order: true,
        ..Mac::default()
    };
    for local in [HalvesLocal::default(), RACE] {
        let before = lowered(&once, mac, local).map(|l| l.dxil).expect("ordered");
        let after = lowered(&once, with_rounds(mac), local)
            .map(|l| l.dxil)
            .expect("ordered");
        assert!(before == after);
    }
}

/// A host wave of the model: where it is in its rounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pc {
    /// About to write (thread 0's wave) or to go to the first barrier.
    Write(usize),
    /// At the barrier before the read, at the read, at the barrier after it.
    Before(usize),
    Read(usize),
    After(usize),
    Done,
}

impl Pc {
    fn waiting(self) -> bool {
        matches!(self, Pc::Before(_) | Pc::After(_))
    }
}

/// Every set of values the reading host wave can end with, and whether some
/// order of the steps leaves a wave waiting for ever: wave 0 holds thread
/// 0 and writes `round + 60` each round, wave 1 reads (`reader` false: its
/// lanes all left before the loop, so it has ended). `after`: the barrier
/// at the end of the read's block.
fn model(rounds: usize, reader: bool, after: bool) -> (Vec<Vec<u32>>, bool) {
    #[derive(Clone)]
    struct S {
        pc: [Pc; 2],
        /// Barriers each wave has passed.
        passed: [usize; 2],
        lds: u32,
        got: Vec<u32>,
    }
    let mut out = Vec::new();
    let mut stuck = false;
    let start = S {
        pc: [Pc::Write(0), if reader { Pc::Write(0) } else { Pc::Done }],
        passed: [0; 2],
        lds: 0,
        got: Vec::new(),
    };
    let mut stack = vec![start];
    while let Some(s) = stack.pop() {
        if s.pc == [Pc::Done; 2] {
            if !out.contains(&s.got) {
                out.push(s.got);
            }
            continue;
        }
        let mut moved = false;
        for w in 0..2 {
            // A wave passes its barrier when every other wave that has not
            // ended has come to the same one (or has passed it already).
            let o = 1 - w;
            let open = s.pc[o] == Pc::Done
                || s.passed[o] > s.passed[w]
                || s.passed[o] == s.passed[w] && s.pc[o].waiting();
            let mut n = s.clone();
            if s.pc[w].waiting() && open {
                n.passed[w] += 1;
            }
            n.pc[w] = match s.pc[w] {
                Pc::Write(r) if r == rounds => Pc::Done,
                Pc::Write(r) => {
                    if w == 0 {
                        n.lds = r as u32 + 60;
                    }
                    Pc::Before(r)
                }
                Pc::Before(r) if open => Pc::Read(r),
                Pc::Read(r) => {
                    if w == 1 {
                        n.got.push(s.lds);
                    }
                    if after {
                        Pc::After(r)
                    } else {
                        Pc::Write(r + 1)
                    }
                }
                Pc::After(r) if open => Pc::Write(r + 1),
                _ => continue,
            };
            moved = true;
            stack.push(n);
        }
        stuck |= !moved;
    }
    (out, stuck)
}

/// With both barriers the reading wave gets each round's value in every
/// order, and a wave that has ended holds up nobody; with the first alone,
/// some order gives it the next round's value.
#[test]
fn the_reading_half_gets_each_rounds_value_only_with_both_barriers() {
    let want: Vec<u32> = (0..3).map(|r| r + 60).collect();
    assert_eq!(model(3, true, true), (vec![want.clone()], false));
    let (one, stuck) = model(3, true, false);
    assert!(!stuck && one.contains(&want) && one.len() > 1, "{one:?}");
    assert_eq!(model(3, false, true), (vec![Vec::new()], false));
}

#[cfg(windows)]
mod gpu {
    use super::super::host::{Dispatch, Host, Root, Stride, View};
    use super::*;
    use crate::native::HwStage;

    /// Runs `words` (one group of 64, each lane storing a dword at
    /// `l * 32`) in halves with the setting on every hardware adapter, 16
    /// times. A line per lane whose dword is not `want`'s.
    fn differences(what: &str, words: &[u32], want: &[u32]) -> Vec<String> {
        let l =
            lowered(words, with_rounds(Mac::default()), HalvesLocal::default()).expect("ordered");
        let shader = super::super::HostShader {
            stage: HwStage::Cs,
            dxil: l.dxil,
            user_data: l.user_data,
            static_samplers: Vec::new(),
            samplers: l.samplers,
            root_cbvs: l.root_cbvs,
            root_srvs: l.root_srvs,
            root_uavs: l.root_uavs,
            root_srv_bytes: l.root_srv_bytes,
            memory: l.memory,
            memory_writes: l.memory_writes,
            view_addresses: l.view_addresses,
            heap: l.heap,
            heap_writes: l.heap_writes,
            view_marks: l.view_marks,
        };
        let views = [
            View::Texture(1, 1, &[[0.0; 4]]),
            View::Buffer {
                data: &[0; 4],
                stride: Stride::Structured(16),
            },
            View::RawUav(64 * 8),
        ];
        let _one = super::super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let hosts = Host::each_hardware();
        if hosts.is_empty() {
            eprintln!("no SM 6.6 hardware device; skipped");
        }
        let mut failed = Vec::new();
        for (adapter, host) in &hosts {
            let label = format!("{adapter} ({what})");
            let Ok((root, pso)) = host.compute_pipeline(&shader) else {
                panic!("{label}: no pipeline");
            };
            let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for round in 0..16 {
                let out = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![(0, Root::Constants(ud.clone()))],
                            views: &views,
                            samplers: &[],
                            groups: [1, 1, 1],
                        },
                    )
                    .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
                let wrong: Vec<String> = (out[0].chunks(8).zip(want).enumerate())
                    .filter(|(_, (got, want))| got[0] != **want)
                    .map(|(l, (got, want))| {
                        format!(
                            "{label}, round {round}: lane {l}: {}, the console {want}",
                            got[0]
                        )
                    })
                    .collect();
                if !wrong.is_empty() {
                    eprintln!("{label}: {} lanes differ in round {round}", wrong.len());
                    failed.extend(wrong);
                    break;
                }
            }
        }
        failed
    }

    /// Every lane that goes round reads thread 0's value of each round
    /// (183 in all), in both halves, and no barrier waits for lanes that
    /// left: none of the upper half, some of it, all of it.
    #[test]
    fn the_broadcast_in_a_loop_is_the_consoles_past_lanes_that_left_on_every_gpu() {
        let mut failed = Vec::new();
        for stay in [64, 40, 32] {
            let want: Vec<u32> = (0..64).map(|l| if l < stay { 183 } else { 0 }).collect();
            failed.extend(differences(
                &format!("{stay} stay"),
                &broadcast(stay),
                &want,
            ));
        }
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }
}
