//! Throughput of the copies a submission makes of title memory into host
//! copies, piece by piece: the bytes alone (memcpy, the kernel read
//! `copy_mapped` uses, one thread and four), and the refresh as the batch
//! runs it (`Shadow::update`: the changed pages taken, guarded again and
//! copied), over pages the title wrote. Run this test alone, in release,
//! with `--ignored --nocapture --test-threads=1` and
//! `VICTORY_FAULT_TRACKING=1`.

use super::*;

const MIB: usize = 1 << 20;
/// Bytes copied per case: as much as the world's busiest copies a frame.
const BYTES: usize = 64 * MIB;
/// Copies the refresh case splits them over (the batch runs one a worker).
const COPIES: usize = 4;

/// GB/s of `bytes` in `us` microseconds.
fn rate(bytes: usize, us: u128) -> f64 {
    bytes as f64 / us.max(1) as f64 / 1e3
}

/// The median of 7 timings of `f` after 2 warm-ups, microseconds.
fn median(mut f: impl FnMut()) -> u128 {
    let mut times = Vec::new();
    for round in 0..9 {
        let started = std::time::Instant::now();
        f();
        if round >= 2 {
            times.push(started.elapsed().as_micros());
        }
    }
    times.sort_unstable();
    times[times.len() / 2]
}

#[test]
#[ignore = "times copies of title memory into host copies"]
fn copies_of_title_memory_into_host_copies() {
    let Some(host) = host() else {
        return; // no host GPU here
    };
    let src = crate::watch::tests::pool(BYTES / PAGE);
    // SAFETY: our pool's bytes.
    unsafe { std::ptr::write_bytes(src as *mut u8, 0x5a, BYTES) };
    let (_buffer, dst) = host.shared_buffer(BYTES as u64).expect("a host buffer");
    let dst = dst as usize;
    let mut heap = vec![0u8; BYTES];
    let pages = BYTES / PAGE;
    let piece = |copy: &dyn Fn(usize, usize, usize), to: usize, from: usize, n: usize| {
        for k in 0..n {
            copy(to + k * PAGE, from + k * PAGE, PAGE);
        }
    };
    let memcpy = |to: usize, from: usize, n: usize| {
        // SAFETY: inside the pool and the destination.
        unsafe { std::ptr::copy_nonoverlapping(from as *const u8, to as *mut u8, n) }
    };
    let kernel = |to: usize, from: usize, n: usize| {
        crate::host::copy_mapped(to as *mut u8, from as *const u8, n, false)
    };
    let heap_at = heap.as_mut_ptr() as usize;
    let cases: [(&str, u128); 6] = [
        (
            "memcpy, 64 KiB pieces, into the host copy",
            median(|| piece(&memcpy, dst, src, pages)),
        ),
        (
            "memcpy, 64 KiB pieces, into our heap",
            median(|| piece(&memcpy, heap_at, src, pages)),
        ),
        (
            "kernel read, 64 KiB pieces, into the host copy",
            median(|| piece(&kernel, dst, src, pages)),
        ),
        (
            "kernel read, one call, into the host copy",
            median(|| kernel(dst, src, BYTES)),
        ),
        (
            "kernel read, 64 KiB pieces, four threads",
            median(|| {
                crate::texture::in_parallel(4, |t| {
                    let at = t * BYTES / 4;
                    piece(&kernel, dst + at, src + at, pages / 4);
                })
            }),
        ),
        (
            "memcpy, 64 KiB pieces, four threads",
            median(|| {
                crate::texture::in_parallel(4, |t| {
                    let at = t * BYTES / 4;
                    piece(&memcpy, dst + at, src + at, pages / 4);
                })
            }),
        ),
    ];
    for (name, us) in cases {
        eprintln!("{name}: {us} us, {:.1} GB/s", rate(BYTES, us));
    }
    drop(heap);
    crate::watch::tests::free(src);

    // The refresh as the batch runs it: four copies of 16 MiB over a pool,
    // every page written by the title (a fault each, not timed), then all
    // four brought up to date at once on the workers.
    let len = BYTES / COPIES;
    let p = crate::watch::tests::pool(BYTES / PAGE);
    let copies: Vec<Arc<Shadow>> = (0..COPIES)
        .map(|k| {
            let at = p + k * len;
            register(at, at as u64, len as u64);
            find(at as u64, len as u64).expect("a host copy")
        })
        .collect();
    for (k, c) in copies.iter().enumerate() {
        c.update(p + k * len, len);
    }
    let (mut faults, mut updates, mut written) = (Vec::new(), Vec::new(), 0);
    for round in 0..9u8 {
        let started = std::time::Instant::now();
        crate::texture::in_parallel(COPIES, |k| {
            for page in 0..len / PAGE {
                // SAFETY: a byte of our pool.
                unsafe { ((p + k * len + page * PAGE) as *mut u8).write_volatile(round) };
            }
        });
        let fault_us = started.elapsed().as_micros();
        let copied = std::sync::atomic::AtomicUsize::new(0);
        let started = std::time::Instant::now();
        crate::texture::in_parallel(COPIES, |k| {
            let n = copies[k].update(p + k * len, len);
            copied.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
        });
        if round >= 2 {
            updates.push(started.elapsed().as_micros());
            faults.push(fault_us);
        }
        written = copied.into_inner();
    }
    faults.sort_unstable();
    updates.sort_unstable();
    let (fault_us, update_us) = (faults[faults.len() / 2], updates[updates.len() / 2]);
    eprintln!(
        "refresh of {COPIES} copies on the workers, every page written: {update_us} us for {written} bytes, \
         {:.1} GB/s (the title's {} writes before, a fault each: {fault_us} us)",
        rate(written, update_us),
        BYTES / PAGE
    );
    for (k, _) in copies.iter().enumerate() {
        unregister(p + k * len, (p + k * len) as u64);
    }
    drop(copies);
    crate::watch::tests::free(p);
}
