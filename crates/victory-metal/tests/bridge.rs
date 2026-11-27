//! The bridge under Wine: Metal through the unixlib, on memory the Windows
//! side allocated, with no copy. tools/macos/test-wine.sh builds the bridge
//! and sets it up; elsewhere (Windows, or Wine without the bridge) these
//! tests say so and pass.
//!
//! `bridge_against_d3d12` (ignored) compares the bridge's costs with the
//! same work through D3D12, which is D3DMetal under Wine:
//!
//! `tools/macos/test-wine.sh -p victory-metal --release --test bridge -- --ignored --nocapture`
#![cfg(windows)]

use std::time::{Duration, Instant};

use victory_metal::{Bridge, OpenError};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
};

/// The bridge, or `None` (with the reason printed) where it cannot be.
fn bridge() -> Option<Bridge> {
    match Bridge::open() {
        Ok(b) => Some(b),
        Err(e @ (OpenError::NotWine | OpenError::NotAsked)) => {
            eprintln!("no Metal bridge here ({e}): skipped");
            None
        }
        Err(e) => panic!("the Metal bridge was asked for but did not open: {e}"),
    }
}

/// Committed read-write memory from `VirtualAlloc`, as the title's own
/// allocations are made, freed on drop.
struct Region {
    base: *mut u32,
    bytes: usize,
}

impl Region {
    fn new(bytes: usize) -> Self {
        // SAFETY: a fresh reservation; nothing else knows its address.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                bytes,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        assert!(!base.is_null(), "VirtualAlloc of {bytes} bytes failed");
        Self {
            base: base.cast(),
            bytes,
        }
    }

    fn words(&mut self) -> &mut [u32] {
        // SAFETY: committed, read-write, ours, and borrowed through `self`.
        unsafe { std::slice::from_raw_parts_mut(self.base, self.bytes / 4) }
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        // SAFETY: the reservation made in `new`; no borrow outlives `self`.
        unsafe { VirtualFree(self.base.cast(), 0, MEM_RELEASE) };
    }
}

#[test]
fn device_is_apple_silicon_with_unified_memory() {
    let Some(bridge) = bridge() else { return };
    let device = bridge.device().expect("the unixlib made no Metal device");
    assert!(device.unified_memory, "{device:?}");
    assert!(device.registry_id != 0, "{device:?}");
    assert!(device.max_buffer_length >= 1 << 30, "{device:?}");
    // Rosetta's pages, which the no-copy buffers below start on.
    assert_eq!(device.page_size, 4096, "{device:?}");
}

#[test]
fn gpu_writes_virtualalloc_memory_in_place() {
    let Some(bridge) = bridge() else { return };
    // 1 MiB for the GPU, then 64 KiB it must not touch.
    const GPU: usize = 1 << 20;
    let mut region = Region::new(GPU + (64 << 10));
    let words = region.words();
    words.fill(0xdead_beef);
    let (gpu, rest) = words.split_at_mut(GPU / 4);
    let times = bridge.fill(gpu, 64, 0x1000).expect("fill");
    for (k, &w) in gpu.iter().enumerate() {
        assert_eq!(
            w,
            0x1000 + k as u32,
            "word {k} after the GPU wrote the memory in place"
        );
    }
    assert!(
        rest.iter().all(|&w| w == 0xdead_beef),
        "the GPU wrote past its range"
    );
    assert!(times.gpu > Duration::ZERO, "{times:?}");

    // The CPU's own stores in between are what the GPU sees next: no copy
    // in either direction, and nothing stale.
    gpu.fill(7);
    bridge
        .fill(&mut gpu[..1024], 1, 0x50_0000)
        .expect("second fill");
    assert!(
        gpu[..1024]
            .iter()
            .enumerate()
            .all(|(k, &w)| w == 0x50_0000 + k as u32)
    );
    assert!(gpu[1024..].iter().all(|&w| w == 7));
}

#[test]
fn fill_refuses_memory_off_page_boundaries() {
    let Some(bridge) = bridge() else { return };
    let mut region = Region::new(64 << 10);
    let words = region.words();
    let e = bridge
        .fill(&mut words[1..1025], 1, 0)
        .expect_err("an unaligned start");
    assert!(e.message.contains("whole pages"), "{e}");
    let e = bridge
        .fill(&mut words[..1024], 3, 0)
        .expect_err("an uneven split");
    assert!(e.message.contains("split evenly"), "{e}");
}

#[test]
fn empty_submissions_complete() {
    let Some(bridge) = bridge() else { return };
    let total = bridge.submit_empty(8).expect("empty submissions");
    assert!(total > Duration::ZERO);
}

/// Indirect draws read the arguments dispatches wrote just before them, in
/// encoders of their own or batched (the unixlib checks the records and the
/// target, and fails the call otherwise).
#[test]
fn draws_read_the_arguments_dispatches_wrote() {
    let Some(bridge) = bridge() else { return };
    for batched in [false, true] {
        let times = bridge.turns(16, batched).expect("turns");
        assert!(times.gpu > Duration::ZERO, "{times:?}");
    }
    assert!(bridge.turns(0, false).is_err(), "no pairs");
}

/// A frame's shape on our M5 Pro: about 4,750 draws (here dispatches) and
/// 23 submissions, each waited for.
const DISPATCHES: u32 = 5000;
const SUBMISSIONS: u32 = 200;
const NOOPS: u32 = 1_000_000;

#[test]
#[ignore = "a benchmark: run it alone, with --nocapture"]
fn bridge_against_d3d12() {
    let Some(bridge) = bridge() else { return };
    let device = bridge.device().unwrap();
    println!("Metal device through the bridge: {device:?}");

    // The way in and back.
    for _ in 0..1000 {
        bridge.noop();
    }
    let t = Instant::now();
    for _ in 0..NOOPS {
        bridge.noop();
    }
    let noop = t.elapsed() / NOOPS;
    println!(
        "unix call (no-op):             {:>9.1} ns",
        noop.as_nanos() as f64
    );

    // Submissions that do nothing, each waited for.
    bridge.submit_empty(20).unwrap();
    let metal_empty = bridge.submit_empty(SUBMISSIONS).unwrap() / SUBMISSIONS;
    let d3d12 = d3d12::Bench::new();
    d3d12.submit_empty(20);
    let d3d12_empty = d3d12.submit_empty(SUBMISSIONS) / SUBMISSIONS;
    println!(
        "empty submission + wait:       Metal {:>7.1} us   D3D12/D3DMetal {:>7.1} us",
        us(metal_empty),
        us(d3d12_empty)
    );

    // DISPATCHES dispatches of 1,024 words each in one submission.
    let bytes = DISPATCHES as usize * 4096;
    let mut region = Region::new(bytes);
    let mut metal = Vec::new();
    for run in 0..6 {
        let t = Instant::now();
        let times = bridge.fill(region.words(), DISPATCHES, run).unwrap();
        let wall = t.elapsed();
        let words = region.words();
        assert!(words.iter().enumerate().all(|(k, &w)| w == run + k as u32));
        metal.push((wall, times));
    }
    let mut d3d = Vec::new();
    for run in 0..6 {
        d3d.push((
            d3d12.fill(bytes, DISPATCHES, run, false),
            d3d12.fill(bytes, DISPATCHES, run, true),
        ));
    }
    println!("{DISPATCHES} dispatches in one submission, run by run (the first compiles):");
    for (run, ((wall, m), (root, table))) in metal.iter().zip(&d3d).enumerate() {
        println!(
            "  run {run}: Metal wall {:>6.2} ms (wrap {:>5.3}, encode {:>5.2}, wait {:>5.2}, GPU {:>5.2})",
            ms(*wall),
            ms(m.wrap),
            ms(m.encode),
            ms(m.wait),
            ms(m.gpu),
        );
        for (how, d) in [("root UAV", root), ("view table", table)] {
            println!(
                "         D3D12/D3DMetal, {how:<10} wall {:>6.2} ms (record {:>5.2}, submit to fence {:>6.2})",
                ms(d.record + d.execute),
                ms(d.record),
                ms(d.execute),
            );
        }
    }
}

/// How a D3D12/D3DMetal submission's time grows with what its bound view
/// heap names: a title's frame binds one heap of millions of slots over
/// some 20,000 resources. A Metal host keeps its resources in a residency
/// set and its views in a buffer, so none of that is per submission.
#[test]
#[ignore = "a benchmark: run it alone, with --nocapture"]
fn d3d12_submission_against_heap_size() {
    if Bridge::open().is_err() {
        eprintln!("not under Wine with the bridge: skipped");
        return;
    }
    let d3d12 = d3d12::Bench::new();
    println!(
        "one submission of N dispatches through one view table, with the heap naming R buffers of 64 KiB:"
    );
    for (resources, slots) in [
        (0, 1),
        (1000, 1001),
        (10_000, 10_001),
        (20_000, 20_001),
        (0, 1_000_000),
    ] {
        let (one, many) = d3d12.heap_submission(resources, slots);
        println!(
            "  R {resources:>6}, slots {slots:>8}: N=1 {:>7.3} ms   N=5000 {:>7.3} ms (median submit to fence)",
            ms(one),
            ms(many)
        );
    }
}

fn us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

mod d3d12;
