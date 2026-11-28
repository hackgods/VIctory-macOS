//! Dispatches counted in threads (`Options::thread_counts`,
//! `lower::thread_counts`): EXEC on entry is the lanes the dispatch
//! launches. The translation tests run on any host; the GPU test (Windows)
//! runs the programs on every hardware adapter, in halves with the
//! hand-over where the console wave is 64 lanes, and compares with what the
//! console's lanes do.

use super::halves_local_tests::{PROLOGUE, S_ENDPGM, native, sop1, vop1};
use super::lower::{self, HalvesLocal, Mac};
use crate::native::{ComputeIo, NativeProgram, RootType, StageIo};
use crate::structure::{Atom, Bool};
use crate::{gfx10, io, ir, resource, structure};

const EXEC: u8 = 126;

fn exchange() -> Mac {
    Mac {
        exchange: true,
        uav_level0: false,
        ..Mac::default()
    }
}

/// Each lane that runs adds 1 at dword `l` and the count of EXEC (the
/// lanes on in its console wave) at dword `64 + l` of the raw UAV, `l` its
/// thread id: the box unpackers' `s_bcnt1_i32_b64 exec` at entry. `wave64`:
/// a wave64 program, else wave32 (`s_bcnt1_i32_b32` of EXEC's low dword).
fn counted(wave64: bool) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        // s_bcnt1_i32_b64 s12, exec (or _b32 of exec_lo)
        sop1(12, if wave64 { 0x10 } else { 0x0F }, EXEC),
        vop1(3, 1, 12), // v_mov_b32 v3, s12
        0x3402_0082,    // v_lshlrev_b32 v1, 2, v0
        0x7E04_0281,    // v_mov_b32 v2, 1
        0xE0C8_1000,
        0x800A_0201, // buffer_atomic_add v2, v1, s[40:43], 0 offen
        0xE0C8_1100,
        0x800A_0301, // buffer_atomic_add v3, v1, s[40:43], 0 offen offset:256
        S_ENDPGM,
    ]);
    w
}

/// `counted` as a program of `threads` (one row, or `[x, y]`), wave64 or
/// not.
fn program(wave64: bool, threads: [u32; 2]) -> NativeProgram {
    let mut n = native(&counted(wave64), threads[0], 0);
    n.wave64 = wave64;
    n.io = StageIo::Cs(ComputeIo {
        rsrc2: 1 << 1,
        threads: [threads[0], threads[1], 1],
    });
    n
}

/// `native` lowered for a host of 32-lane waves, as a program for
/// dispatches counted in threads or not.
fn lowered(native: &NativeProgram, thread_counts: bool, mac: Mac) -> Result<lower::Lowered, crate::ShaderError> {
    let mut p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    p.thread_counts = thread_counts;
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
        narrow_waves: false,
        halves_local: HalvesLocal { reads: false, lds: false },
        mac,
        host_wave32: true,
    })
}

/// The words a translation reads for the thread counts.
fn counts(l: &lower::Lowered) -> Vec<super::UserData> {
    let code = RootType::Other(super::USER_DISPATCH_THREADS);
    l.user_data.iter().copied().filter(|u| u.root == code).collect()
}

#[test]
fn the_test_program_is_what_its_comments_say() {
    let code: Vec<u8> = counted(true).iter().flat_map(|w| w.to_le_bytes()).collect();
    let text: Vec<String> = gfx10::decode(&code).unwrap().iter().map(|i| i.text(true)).collect();
    for want in [
        "s_bcnt1_i32_b64 s12, exec",
        "v_mov_b32 v3, s12",
        "v_lshlrev_b32 v1, 2, v0",
        "v_mov_b32 v2, 1",
        "buffer_atomic_add v2, v1, s[40:43], 0 offen",
        "buffer_atomic_add v3, v1, s[40:43], 0 offen offset:256",
    ] {
        assert!(text.iter().any(|t| t.contains(want)), "{want:?} not in {text:?}");
    }
    let code: Vec<u8> = counted(false).iter().flat_map(|w| w.to_le_bytes()).collect();
    let text = gfx10::decode(&code).unwrap()[4].text(true);
    assert!(text.contains("s_bcnt1_i32_b32 s12, exec_lo"), "{text}");
}

/// Without the setting EXEC on entry is every lane: the first instruction
/// has no guard. With it, EXEC on entry is a lane mask of its own, and
/// every instruction that reads EXEC is guarded by that mask's bit.
#[test]
fn exec_on_entry_is_the_lanes_the_dispatch_launches() {
    let n = program(true, [64, 1]);
    let mut p = ir::build(&n.code, true, n.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let first = |st: &structure::Structured| {
        let b = &p.blocks[0];
        let i = b.insts.iter().position(|i| i.name().starts_with("v_")).unwrap();
        st.guards[0][i]
    };
    assert_eq!(first(&st), None, "every lane is on");
    p.thread_counts = true;
    let st = structure::structure(&p).unwrap();
    let exec = p.inputs.iter().find(|i| i.0 == crate::ir::Reg::S(crate::ir::EXEC)).unwrap().1;
    let g = first(&st).expect("guarded");
    assert_eq!(st.bools.node(g), Some((Atom::Lane(exec), Bool::FALSE, Bool::TRUE)));
}

/// One count a dimension wider than one thread, past the console's user
/// data (one SGPR here), and none without the setting or for a group of one
/// thread.
#[test]
fn a_program_for_thread_counts_reads_one_count_a_wide_dimension() {
    let word = |index, dword| super::UserData {
        index,
        slot: 0,
        root: RootType::Other(super::USER_DISPATCH_THREADS),
        dword,
    };
    let row = program(true, [64, 1]);
    assert_eq!(counts(&lowered(&row, true, exchange()).unwrap()), [word(1, 0)]);
    assert_eq!(counts(&lowered(&row, false, exchange()).unwrap()), []);
    let square = program(false, [8, 8]);
    assert_eq!(counts(&lowered(&square, true, Mac::default()).unwrap()), [word(1, 0), word(2, 1)]);
    let one = program(false, [1, 1]);
    assert_eq!(counts(&lowered(&one, true, Mac::default()).unwrap()), []);
}

/// In a wave64 program run as two 32-lane halves, the count of EXEC on
/// entry is a lane mask read as data under thread counts (no longer a
/// constant 64). It is handed over: by this line's exchange where asked
/// for, else by the Mac port's own (`Lower::across_first`); the two
/// translations differ.
#[test]
fn the_count_of_exec_in_halves_is_handed_over() {
    let n = program(true, [64, 1]);
    let all = lowered(&n, false, exchange()).expect("every lane on: a constant");
    let handed = lowered(&n, true, exchange()).expect("handed over");
    let across = lowered(&n, true, Mac::default()).expect("the port's hand-over");
    assert_ne!(all.dxil, handed.dxil);
    assert_ne!(handed.dxil, across.dxil);
}

#[cfg(windows)]
mod gpu {
    use super::super::host::{Dispatch, Host, Root, Stride, View};
    use super::*;
    use crate::native::HwStage;

    /// What `native` (a group of 64 threads) writes in `groups` groups
    /// with thread count `count` in x, on `host`: the per-thread dword
    /// sums [`counted`] makes.
    fn run(host: &Host, label: &str, native: &NativeProgram, mac: Mac, groups: u32, count: u32) -> Vec<u32> {
        let l = lowered(native, true, mac).expect("translates");
        let [limit] = counts(&l)[..] else {
            panic!("{label}: one count");
        };
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
            View::RawUav(128),
        ];
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            panic!("{label}: no pipeline");
        };
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        ud[limit.index as usize] = count;
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [groups, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out[0].clone()
    }

    /// The console's sums: thread `l` of group `g` runs if `64 g + l` is
    /// below the count (0: every thread), and counts the lanes on in its
    /// console wave of `wave` lanes.
    fn console(groups: u32, count: u32, wave: u32) -> Vec<u32> {
        let all = groups * 64;
        let n = if count == 0 { all } else { count.min(all) };
        let runs = |t: u32| t < n;
        let on_in_wave = |t: u32| {
            let first = t / wave * wave;
            (first..first + wave).filter(|&u| runs(u)).count() as u32
        };
        let mut want = vec![0; 128];
        for g in 0..groups {
            for l in 0..64 {
                let t = 64 * g + l;
                if runs(t) {
                    want[l as usize] += 1;
                    want[64 + l as usize] += on_in_wave(t);
                }
            }
        }
        want
    }

    /// Only the threads below the count run, in one group or two, in a
    /// wave64 program (two 32-lane halves, the count of EXEC handed over)
    /// and a wave32 one; a count of 0 runs them all. Each count 4 times.
    #[test]
    fn only_the_threads_below_the_count_run_on_every_gpu() {
        let _one = super::super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let hosts = Host::each_hardware();
        if hosts.is_empty() {
            eprintln!("no SM 6.6 hardware device; skipped");
            return;
        }
        let mut failed = Vec::new();
        for (adapter, host) in &hosts {
            for (wave64, mac) in [(true, exchange()), (false, Mac::default())] {
                let n = program(wave64, [64, 1]);
                let wave = if wave64 { 64 } else { 32 };
                for groups in [1, 2] {
                    for count in [0u32, 1, 7, 31, 32, 33, 45, 63, 64, 65, 70, 100, 127, 128] {
                        if groups == 1 && count > 64 {
                            continue;
                        }
                        let label = format!("{adapter} (wave64 {wave64}, {groups} groups, count {count})");
                        let want = console(groups, count, wave);
                        for round in 0..4 {
                            let got = run(host, &label, &n, mac, groups, count);
                            if got != want {
                                failed.push(format!("{label}, round {round}: {got:?}, the console {want:?}"));
                                break;
                            }
                        }
                    }
                }
            }
            eprintln!("{adapter}: thread counts checked");
        }
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }
}
