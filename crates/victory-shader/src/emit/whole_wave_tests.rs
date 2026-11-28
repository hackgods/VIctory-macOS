//! Whole-wave code and first-lane reads under a narrowed EXEC in programs
//! run as two 32-lane halves (`Options::halves_whole_wave`, `walk`,
//! "Whole-wave code"; `lower::exchange`; `notes/far-ray.md`). The
//! translation tests run on any host; the GPU tests (Windows) run the
//! programs on every hardware adapter and compare with the console's values.
//!
//! The programs have the shape of the title's ray bin offsets
//! (`sr_tracing:CS_GenerateRayBinOffsets`): inside an `if` that only some
//! lanes take, the console's compiler sets EXEC to every lane for a wave's
//! sum (`s_or_saveexec_b64 s, -1`), so lanes outside the `if` run that code
//! too; then, under the `if`'s EXEC again, one lane's value is read with
//! `v_readfirstlane_b32` (the base an atomic returned).

use super::halves_local_tests::{PROLOGUE, S_ENDPGM, V16_IS_32_TIMES_V0, native, sop1, store};
use super::lower::{self, HalvesLocal, Mac};
use super::walk;
use super::walk_tests::translated;
use crate::{gfx10, io, ir, resource, structure};

const EXEC: u8 = 126;
/// `v_mov_b32 v1, 0`.
const V1_IS_0: u32 = 0x7E02_0280;
/// `s_mov_b64 s[2:3], exec` and back.
const SAVE: u32 = sop1(2, 0x04, EXEC);
const RESTORE: u32 = sop1(EXEC, 0x04, 2);
/// `v_cmpx_lt_u32 exec, 29, v0`: the lanes inside are those from 30 on.
const INSIDE: u32 = 0x7DA2_009D;
/// `s_mov_b32 s20, 0`; and the loop's end: `s_add_u32 s20, s20, 1`,
/// `s_cmp_lt_u32 s20, 2`, `s_cbranch_scc1` back over `body` words.
const S20_IS_0: u32 = sop1(20, 0x03, 128);
fn twice(body: u32) -> [u32; 3] {
    [0x8014_8114, 0xBF0A_8214, 0xBF85_0000 | (0x1_0000 - (body + 3))]
}

/// `s_cbranch_execz` over `n` words.
const fn execz(n: u32) -> u32 {
    0xBF88_0000 | n
}

/// The `if`'s body: whole-wave code puts each lane's thread id, or 0 for
/// a lane outside the `if`, in v2; under the `if`'s EXEC again the lanes
/// count the `if`'s lanes and read the first one's v2, and v1 takes
/// `first * 256 + count`. v2 is not read after the `if`.
const BODY: [u32; 10] = [
    sop1(18, 0x25, 193), // s_or_saveexec_b64 s[18:19], -1
    0xD501_0002,
    0x004A_0080,          // v_cndmask_b32 v2, 0, v0, s[18:19]
    sop1(EXEC, 0x04, 18), // s_mov_b64 exec, s[18:19]
    sop1(12, 0x10, 18),   // s_bcnt1_i32_b64 s12, s[18:19]
    0x7E1A_0502,          // v_readfirstlane_b32 s13, v2
    0x8F0D_880D,          // s_lshl_b32 s13, s13, 8
    0x800D_0C0D,          // s_add_u32 s13, s13, s12
    0x7E02_020D,          // v_mov_b32 v1, s13
    RESTORE,
];

/// Each lane stores 0 first; then the `if` over the lanes from 30 on.
fn start() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([V16_IS_32_TIMES_V0, V1_IS_0]);
    w.extend(store(1, 0));
    w
}

/// The `if` once. On the console lanes 0 to 29 store 0 and lanes 30 to 63
/// store 30 * 256 + 34 = 7,714; a second console wave (lanes 64 to 127, all
/// inside) stores 64 * 256 + 64 = 16,448. Each half on its own would store
/// 7,682 (lanes 30, 31) and 8,224.
fn in_an_if() -> Vec<u32> {
    let mut w = start();
    w.extend([SAVE, INSIDE, execz(BODY.len() as u32 - 1)]);
    w.extend(BODY);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// The same `if` twice round a loop the group goes round together, as the
/// title's program goes round its bins: the same values.
fn in_a_loop() -> Vec<u32> {
    let mut w = start();
    w.push(S20_IS_0);
    let body = 3 + BODY.len() as u32;
    w.extend([SAVE, INSIDE, execz(BODY.len() as u32 - 1)]);
    w.extend(BODY);
    w.extend(twice(body));
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// [`in_an_if`] storing v2 as well after the `if` (at `+4`): what the
/// whole-wave code left in a lane outside the `if` is read, so the walk
/// must not open it.
fn whole_wave_value_read_after() -> Vec<u32> {
    let mut w = start();
    w.extend([SAVE, INSIDE, execz(BODY.len() as u32 - 1)]);
    w.extend(BODY);
    w.extend(store(1, 0));
    w.extend(store(2, 4));
    w.push(S_ENDPGM);
    w
}

/// `v_cmp_gt_u32 vcc, 38, v0`: threads 30 to 37 (of those inside).
const BELOW_38: u32 = 0x7D88_00A6;
/// `s_and_saveexec_b64 s[4:5], vcc` and back.
const ONLY_BELOW_38: u32 = sop1(4, 0x24, 106);
const BACK: u32 = sop1(EXEC, 0x04, 4);

/// A DS instruction on the dword at `v<addr>`, data `v<data>`, result
/// `v<dst>`.
const fn ds(op: u32, addr: u8, data: u8, dst: u8) -> [u32; 2] {
    [0xD800_0000 | op << 18, (dst as u32) << 24 | (data as u32) << 8 | addr as u32]
}

/// The shape of the title's ray budgeting with ray bins
/// (`sr_tracing:CS_RayBudgeting_Cull_RayBinning`): behind an early-out (the
/// lanes from 30 on stay), threads 30 to 37 clear an LDS counter, every lane
/// left adds 1, and threads 30 to 37 read it back. One console wave runs
/// it in lockstep, without a barrier: they store 34, every other lane 0.
fn lds_behind_an_early_out() -> Vec<u32> {
    let mut w = start();
    w.extend([INSIDE, execz(15)]);
    w.extend([0x7E06_0280, 0x7E08_0281, BELOW_38, ONLY_BELOW_38]); // v3 = 0, v4 = 1
    w.extend(ds(13, 3, 1, 0)); // ds_write_b32 v3, v1
    w.push(BACK);
    w.extend(ds(0, 3, 4, 0)); // ds_add_u32 v3, v4
    w.push(ONLY_BELOW_38);
    w.extend(ds(54, 3, 0, 1)); // ds_read_b32 v1, v3
    w.push(BACK);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// `words` as [`translated`] has it, with 4 bytes of LDS, on a host of
/// narrow waves (as under Wine, where barriers can order a split wave's
/// LDS), LDS raced between the halves where `race` asks
/// (`Options::halves_shared_lds`), and as `emit::walk::exact` does with the
/// switch: raced only in the lowering that finds the places to open.
fn with_lds(words: &[u32], reads: bool, race: bool, mac: Mac) -> Result<lower::Lowered, crate::ShaderError> {
    let native = native(words, 64, 4);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lower = |(p, st, io, res): walk::Stages<'_>, reads: bool, race: bool| {
        lower::lower(&lower::Parts {
            native: &native,
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
            narrow_waves: true,
            halves_local: HalvesLocal { reads, lds: race },
            mac,
            host_wave32: true,
        })
    };
    let start = (&p, &st, &io, &res);
    let walked = |stages: walk::Stages<'_>, local: bool| {
        let lds_part = mac.whole_wave && !mac.whole_wave_no_lds;
        lower(stages, local, if lds_part { local } else { race })
    };
    if mac.walk
        && let Some(exact) = walk::open(&native, start, mac.whole_wave, walked)
    {
        return Ok(exact);
    }
    lower(start, reads, race)
}

fn walking(whole_wave: bool) -> Mac {
    Mac {
        exchange: true,
        walk: true,
        whole_wave,
        ..Mac::default()
    }
}

/// The hand-encoded words are the instructions their comments name.
#[test]
fn the_test_programs_are_what_their_comments_say() {
    // (Decoded with an end after them, as a program has.)
    let text = |words: &[u32]| {
        let code: Vec<u8> = words
            .iter()
            .chain(&[S_ENDPGM])
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let insts = gfx10::decode(&code).expect("decodes");
        let n = insts.len() - 1;
        insts[..n].iter().map(|i| i.text(true)).collect::<Vec<_>>().join("\n")
    };
    let body = text(&BODY);
    let want = [
        "s_or_saveexec_b64 s[18:19], -1",
        "v_cndmask_b32 v2, 0, v0, s[18:19]",
        "s_mov_b64 exec, s[18:19]",
        "s_bcnt1_i32_b64 s12, s[18:19]",
        "v_readfirstlane_b32 s13, v2",
        "s_lshl_b32 s13, s13, 8",
        "s_add_u32 s13, s13, s12",
        "v_mov_b32 v1, s13",
        "s_mov_b64 exec, s[2:3]",
    ]
    .join("\n");
    assert_eq!(body, want);
    let head = "s_mov_b64 s[2:3], exec\nv_cmpx_lt_u32 exec, 29, v0\ns_cbranch_execz";
    for words in [in_an_if(), in_a_loop(), whole_wave_value_read_after()] {
        let all = text(&words);
        assert!(all.contains(head) && all.contains(&want), "{all}");
    }
    let lds = text(&lds_behind_an_early_out());
    let want_lds = "v_mov_b32 v3, 0\nv_mov_b32 v4, 1\nv_cmp_gt_u32 vcc, 38, v0\n\
                    s_and_saveexec_b64 s[4:5], vcc\nds_write_b32 v3, v1\ns_mov_b64 exec, s[4:5]\n\
                    ds_add_u32 v3, v4\ns_and_saveexec_b64 s[4:5], vcc\nds_read_b32 v1, v3\n\
                    s_mov_b64 exec, s[4:5]";
    assert!(lds.contains(want_lds) && lds.contains("v_cmpx_lt_u32 exec, 29, v0"), "{lds}");
    let looped = text(&in_a_loop());
    let tail = "s_add_u32 s20, s20, 1\ns_cmp_lt_u32 s20, 2\ns_cbranch_scc1";
    assert!(looped.contains("s_mov_b32 s20, 0") && looped.contains(tail), "{looped}");
}

/// Without the switch the walk does not open the `if` (lanes outside it run
/// its whole-wave code), so the count and the first lane are refused, or
/// with half-local reads each half's own. With it the program translates
/// with every read across the halves handed over: the same translation with
/// half-local reads or without, and other than the half-local one.
#[test]
fn whole_wave_code_in_an_if_is_walked_and_its_reads_handed_over() {
    for (what, words) in [("if", in_an_if()), ("loop", in_a_loop())] {
        let halves = |reads, mac| translated(&words, 64, true, reads, mac);
        assert!(halves(false, walking(false)).is_err(), "{what}: refused without the switch");
        let local = halves(true, walking(false)).expect("half-local");
        let on = halves(false, walking(true)).unwrap_or_else(|e| panic!("{what}: {e}"));
        assert_eq!(on.wave_size, Some(32), "{what}");
        // The EXEC count, the first lane's ballot and its value.
        assert!(on.exchanges >= 3, "{what}: {} exchanges", on.exchanges);
        let with_local = halves(true, walking(true)).expect("handed over");
        assert_eq!(with_local.dxil, on.dxil, "{what}");
        assert_ne!(local.dxil, on.dxil, "{what}");
        // Two console waves in a group.
        translated(&words, 128, true, false, walking(true))
            .unwrap_or_else(|e| panic!("{what}, two waves: {e}"));
        // A host of 64-lane waves answers by itself, switch or not.
        let whole = |mac| translated(&words, 64, false, false, mac).expect("one wave64").dxil;
        assert_eq!(whole(walking(true)), whole(Mac::default()), "{what}");
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut on.dxil.clone())
                .unwrap_or_else(|e| panic!("{what}: DXIL validation failed: {e}"));
        }
    }
}

/// A VGPR the whole-wave code wrote is read after the `if`: a lane outside
/// it would see what that code left, where on the console (whose wave went
/// round the `if` when no lane was inside) it sees the old value. The walk
/// leaves such an `if` alone, and the translation is what it was without
/// the switch.
#[test]
fn whole_wave_code_whose_vgpr_is_read_after_stays_as_it_was() {
    let words = whole_wave_value_read_after();
    let halves = |reads, mac| translated(&words, 64, true, reads, mac);
    assert!(halves(false, walking(true)).is_err());
    let on = halves(true, walking(true)).expect("half-local").dxil;
    let off = halves(true, walking(false)).expect("half-local").dxil;
    assert_eq!(on, off);
}

/// Without the switch the LDS behind the early-out is refused (no block
/// every lane runs holds a place for a barrier), or with
/// `VICTORY_HALVES_SHARED_LDS` raced. With it the walk opens the early-out
/// and the program translates with group barriers, raced or not asked:
/// the same bytes, and other than the raced ones.
#[test]
fn lds_behind_an_early_out_is_walked_and_ordered() {
    let words = lds_behind_an_early_out();
    assert!(with_lds(&words, false, false, walking(false)).is_err());
    let raced = with_lds(&words, true, true, walking(false)).expect("raced");
    // (Ordered by barriers, it no longer needs the console's lanes.)
    let on = with_lds(&words, false, false, walking(true)).expect("ordered");
    let on_raced = with_lds(&words, true, true, walking(true)).expect("ordered");
    assert_eq!(on.dxil, on_raced.dxil);
    assert_ne!(on.dxil, raced.dxil);
}

/// With the LDS part left out (`VICTORY_HALVES_WHOLE_WAVE_LDS` not naming
/// the program) the LDS behind the early-out is as without the switch:
/// refused, or raced with the same bytes.
#[test]
fn without_the_lds_part_the_lds_stays_as_it_was() {
    let words = lds_behind_an_early_out();
    let no_lds = Mac {
        whole_wave_no_lds: true,
        ..walking(true)
    };
    assert!(with_lds(&words, false, false, no_lds).is_err());
    let raced = with_lds(&words, true, true, walking(false)).expect("raced").dxil;
    assert_eq!(with_lds(&words, true, true, no_lds).expect("raced").dxil, raced);
}

#[cfg(windows)]
mod gpu {
    use super::super::host::{Dispatch, Host, Root, Stride, View};
    use super::*;
    use crate::native::HwStage;

    /// Runs `words` (one group of `threads`, each lane storing a dword at
    /// `l * 32`) with the switch on every hardware adapter, 16 times. A
    /// line per lane whose dword is not `want`'s.
    fn differences(what: &str, words: &[u32], threads: u32, want: &[u32]) -> Vec<String> {
        let l = translated(words, threads, true, false, walking(true)).expect("handed over");
        run(what, l, threads, want)
    }

    /// Runs the translation `l` (one group of `threads`) on every hardware
    /// adapter, 16 times; as [`differences`].
    fn run(what: &str, l: lower::Lowered, threads: u32, want: &[u32]) -> Vec<String> {
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
            View::RawUav(threads * 8),
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
            // Many times: a hand-over that is only usually right shows.
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
                let wrong: Vec<String> = out[0]
                    .chunks(8)
                    .zip(want)
                    .enumerate()
                    .filter(|(_, (got, want))| got[0] != **want)
                    .map(|(l, (got, want))| {
                        format!("{label}, round {round}: lane {l}: {}, the console {want}", got[0])
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

    /// The count and the first lane are the console's in both halves (34
    /// lanes from 30 on, the first 30: 7,714), lanes outside the `if` keep
    /// their 0, and a second console wave in the group has its own (64
    /// lanes from 64: 16,448); in an `if` and twice round a loop.
    #[test]
    fn the_count_and_the_first_lane_are_the_consoles_in_both_halves_on_every_gpu() {
        let one: Vec<u32> = (0..64).map(|l| if l < 30 { 0 } else { 7714 }).collect();
        let two: Vec<u32> = one.iter().copied().chain([16448; 64]).collect();
        let mut failed = Vec::new();
        for (what, words) in [("if", in_an_if()), ("loop", in_a_loop())] {
            failed.extend(differences(what, &words, 64, &one));
            let what = format!("{what}, two waves");
            failed.extend(differences(&what, &words, 128, &two));
        }
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }

    /// The LDS counter behind the early-out is the console's 34 in threads
    /// 30 to 37 every time (the halves' adds all after the clear and
    /// before the read), and every other lane keeps 0.
    #[test]
    fn lds_behind_an_early_out_is_the_consoles_on_every_gpu() {
        let l = with_lds(&lds_behind_an_early_out(), false, false, walking(true)).expect("ordered");
        let want: Vec<u32> = (0..64).map(|l| if (30..38).contains(&l) { 34 } else { 0 }).collect();
        let failed = run("LDS behind an early-out", l, 64, &want);
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }
}
