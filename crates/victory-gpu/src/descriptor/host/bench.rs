//! Heap-sync benchmark: a synthetic heap of 2,048 slots synced in pages of
//! 128, an eighth of its slots changed between sweeps (a tester's pattern on
//! #409), by one thread alone and with another taking the slot lock all the
//! while, as title threads binding views do. Run alone, in release, under
//! Wine:
//! `tools/macos/test-wine.sh -p victory-gpu --release --lib -- --ignored --nocapture heap_sync_bench`

use super::*;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

const SLOTS: u32 = 2048;
const PAGE: u32 = 128;
const SWEEPS: usize = 2000;

/// The median time of a sweep over every page, in microseconds, the slot
/// lock taken meanwhile by another thread if `contended`.
fn sweep_us(heap: &HostHeap, contended: bool) -> f64 {
    let null = std::ptr::null();
    // Two views of no memory (made null, as cheaply as a view is made).
    let mut views = [[0u8; 32]; 2];
    crate::descriptor::create_cbv(null, &[0, 0, 256, 0], views[0].as_mut_ptr() as u64);
    crate::descriptor::create_cbv(null, &[0, 0, 512, 0], views[1].as_mut_ptr() as u64);
    let mut memory = vec![0u8; SLOTS as usize * 32];
    for slot in memory.as_chunks_mut::<32>().0 {
        *slot = views[0];
    }
    let start = memory.as_ptr() as u64;
    let sweep = |memory: &[u8]| {
        for first in (0..SLOTS).step_by(PAGE as usize) {
            let bytes = &memory[first as usize * 32..(first + PAGE) as usize * 32];
            heap.sync_slots_bytes(
                first,
                first + PAGE,
                start,
                32,
                start + u64::from(first) * 32,
                bytes,
            );
        }
    };
    sweep(&memory);
    let stop = AtomicBool::new(false);
    std::thread::scope(|s| {
        if contended {
            s.spawn(|| {
                while !stop.load(Ordering::Relaxed) {
                    let made = heap.made.lock().unwrap();
                    std::hint::black_box(made[0]);
                }
            });
        }
        let mut times = Vec::with_capacity(SWEEPS);
        for pass in 0..SWEEPS + 100 {
            // An eighth of the slots get the other view.
            for (i, slot) in memory.as_chunks_mut::<32>().0.iter_mut().enumerate() {
                if i % 8 == pass % 8 {
                    *slot = views[usize::from(*slot == views[0])];
                }
            }
            let begun = Instant::now();
            sweep(&memory);
            if pass >= 100 {
                times.push(begun.elapsed().as_nanos());
            }
        }
        stop.store(true, Ordering::Relaxed);
        // Every slot made from the record it holds last.
        let made = heap.made.lock().unwrap();
        for (i, slot) in memory.as_chunks::<32>().0.iter().enumerate() {
            let word = |k: usize| u32::from_le_bytes(slot[4 * k..4 * k + 4].try_into().unwrap());
            assert_eq!(
                made[i],
                u64::from(word(2)) << 32 | u64::from(word(1)),
                "slot {i}"
            );
        }
        drop(made);
        times.sort_unstable();
        times[times.len() / 2] as f64 / 1e3
    })
}

#[test]
#[ignore = "a benchmark: run alone, in release"]
fn heap_sync_bench() {
    if host().is_none() {
        return;
    }
    let heap = HostHeap::new(false, SLOTS).expect("a host heap");
    let alone = sweep_us(&heap, false);
    let contended = sweep_us(&heap, true);
    println!("heap sync sweep: {alone:.1} us alone, {contended:.1} us with a contending reader");
}
