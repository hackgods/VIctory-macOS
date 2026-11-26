//! mac: tests of a host heap whose slots have locks of their own
//! (`VICTORY_HEAP_LOCKS=1`, `crate::descriptor::slots`), on the host GPU.

use std::sync::atomic::{AtomicBool, Ordering};

use super::*;

/// What a slot shows of itself: whether it is made and holds a host
/// resource, the record `mirror` finds without the lock, its view-stride
/// entry, and the range the census has for it.
pub(super) fn shown(heap: &HostHeap, index: u32) -> (bool, bool, u64, u32, Option<crate::bindless::Range>) {
    let stride = heap.strides.as_ref().map_or(0, |(_, data)| {
        // SAFETY: a dword of the mapped table, one per slot.
        unsafe { (*data as *const u32).add(index as usize).read_volatile() }
    });
    let ready = heap.ready[index as usize].load(Ordering::Acquire);
    (heap.made(index), heap.kept(index), ready, stride, heap.viewed.range_at(index))
}

/// A title heap's records for tests: `slots` of 32 bytes, zeroed, and
/// records of distinct shader views of a buffer to copy into them as the
/// title does (as bytes).
pub(super) struct Title {
    pub(super) resource: crate::object::This,
    records: Box<[[u32; 8]]>,
    pub(super) views: Vec<[u32; 8]>,
}

// SAFETY: the resource is a title object, used from any thread as the
// title's are; the records are written and read as the title's heap is.
unsafe impl Sync for Title {}

impl Title {
    pub(super) fn new(slots: u32, views: u32) -> Title {
        const BYTES: u32 = 4096;
        let resource = crate::resource::test_buffer(u64::from(BYTES));
        let mut scratch = [0u32; 8];
        let views = (0..views)
            .map(|v| {
                // R8_UINT (format, buffer dimension, mapping, padding, first
                // element in two words, count, stride, flags, padding): 16
                // elements from another place each.
                let first = v * 8 % (BYTES - 16);
                let desc = [62, 1, 0x1688, 0, first, 0, 16, 0, 0, 0];
                crate::descriptor::create_srv(std::ptr::null(), resource, &desc, scratch.as_mut_ptr() as u64);
                scratch
            })
            .collect();
        Title { resource, records: vec![[0u32; 8]; slots as usize].into(), views }
    }

    pub(super) fn start(&self) -> u64 {
        self.records.as_ptr() as u64
    }

    pub(super) fn at(&self, index: u32) -> u64 {
        self.start() + 32 * u64::from(index)
    }

    /// Writes `record` (none: zeros) into slot `index`, its first 16 bytes
    /// (the tag, the view and its generation) in one store, as the title's
    /// 32-byte stores land them.
    pub(super) fn write(&self, index: u32, record: Option<&[u32; 8]>) {
        use std::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_storeu_si128};
        let zeros = [0u32; 8];
        let from = record.unwrap_or(&zeros).as_ptr().cast::<__m128i>();
        let to = self.at(index) as *mut __m128i;
        // SAFETY: 32 bytes of a record of ours and of a slot of `records`;
        // other threads read the slot as they read the title's heap.
        unsafe {
            _mm_storeu_si128(to.add(1), _mm_loadu_si128(from.add(1)));
            _mm_storeu_si128(to, _mm_loadu_si128(from));
        }
    }
}

impl Drop for Title {
    fn drop(&mut self) {
        crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(self.resource, 0x10)(self.resource);
    }
}

/// A small generator of a thread's own.
pub(super) fn random(seed: u64) -> impl FnMut() -> u64 {
    let mut x = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(seed + 1);
    move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    }
}

/// Several threads at one heap whose slots have locks of their own: title
/// threads rewriting records and making them by call, a second pass like a
/// heap sync's, a draw's reads, and the looks that take a slot's lock and
/// the kept pieces. After a last sync with the title at rest every slot
/// shows what a heap with the one lock, made by one thread from the same
/// records, shows; and the host is content.
#[test]
fn slots_made_by_several_threads_show_what_one_thread_makes() {
    let Some(host) = host() else {
        return; // no host GPU here
    };
    const SLOTS: u32 = 4096;
    // As the split form runs: a held lock is tried again before a sleep.
    crate::descriptor::slots::spin_in_tests();
    let title = Title::new(SLOTS, 96);
    let (Some(split), Some(one)) = (HostHeap::new_as(false, SLOTS, true), HostHeap::new_as(false, SLOTS, false)) else {
        return;
    };
    assert!(split.made.split() && split.kept.split() && !one.made.split() && !one.kept.split());
    let stop = AtomicBool::new(false);
    std::thread::scope(|scope| {
        // Title threads: a record (now and then none) into a slot, and
        // every other time the call that makes it.
        let titles: Vec<_> = (0..2u64)
            .map(|t| {
                let (title, split) = (&title, &split);
                scope.spawn(move || {
                    let mut next = random(t);
                    for round in 0..30_000u32 {
                        let index = (next() % u64::from(SLOTS)) as u32;
                        let view = (round % 13 != 0).then(|| &title.views[(next() % title.views.len() as u64) as usize]);
                        title.write(index, view);
                        if round % 2 == 0 {
                            split.mirror(index, title.at(index), false);
                        }
                    }
                })
            })
            .collect();
        // Heap syncs, of a 64 KiB page of records at a time, two at once
        // (in the game one at a time: the submission lock).
        for t in 2..4u64 {
            let (title, split, stop) = (&title, &split, &stop);
            scope.spawn(move || {
                let mut next = random(t);
                while !stop.load(Ordering::Relaxed) {
                    let first = (next() % u64::from(SLOTS / 2048)) as u32 * 2048;
                    split.sync_slots(first, first + 2048, title.start(), 32);
                }
            });
        }
        // A draw's reads, and the looks at single slots.
        {
            let (title, split, stop) = (&title, &split, &stop);
            scope.spawn(move || {
                let mut next = random(4);
                while !stop.load(Ordering::Relaxed) {
                    let index = (next() % u64::from(SLOTS)) as u32;
                    let _ = split.mirror(index, title.at(index), true);
                    split.stamp_read(index);
                    let _ = split.unmade_placed(index);
                    let _ = split.textures_at(&[index, index + 1]);
                    // No texture is held here: nothing to give up, every
                    // piece walked.
                    split.evict(&std::collections::HashSet::from([1usize]));
                }
            });
        }
        // The others go on until the title threads are done.
        for t in titles {
            t.join().expect("a title thread");
        }
        stop.store(true, Ordering::Relaxed);
    });
    // The title at rest: a last sync of each heap.
    for heap in [&split, &one] {
        heap.sync_slots(0, 2048, title.start(), 32);
        heap.sync_slots(2048, SLOTS, title.start(), 32);
    }
    let mut made = 0;
    for i in 0..SLOTS {
        assert_eq!(shown(&split, i), shown(&one, i), "slot {i}");
        made += u32::from(shown(&split, i).0);
    }
    assert!(made > SLOTS / 2, "most slots hold a view: {made}");
    // SAFETY: a plain device query.
    assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    assert_eq!(host.log_messages(), None);
    drop((split, one));
}

/// A pass made on several threads (`VICTORY_HEAP_SYNC_THREADS`,
/// `HostHeap::sync_wide`) leaves every slot as the pass alone leaves it,
/// pass after pass as the title rewrites its records; and the host is
/// content. Says how long the passes took each way (`--nocapture`).
#[test]
fn a_pass_made_on_several_threads_shows_what_the_pass_alone_makes() {
    let Some(host) = host() else {
        return; // no host GPU here
    };
    const SLOTS: u32 = 8192;
    // As the split form runs: a held lock is tried again before a sleep.
    crate::descriptor::slots::spin_in_tests();
    let title = Title::new(SLOTS, 160);
    let pools: Vec<crate::descriptor::pool::Pool> = [2, 3, 4].map(crate::descriptor::pool::Pool::new).into();
    let wide: Vec<HostHeap> = pools.iter().filter_map(|_| HostHeap::new_as(false, SLOTS, true)).collect();
    let (Some(alone), Some(one)) = (HostHeap::new_as(false, SLOTS, true), HostHeap::new_as(false, SLOTS, false)) else {
        return;
    };
    if wide.len() != pools.len() {
        return;
    }
    let all: Vec<u32> = (0..SLOTS).collect();
    let at = |i: u32| title.at(i);
    let mut next = random(9);
    // Milliseconds: the pass alone on a split heap, on a one-lock heap,
    // and on each pool.
    let mut ms = vec![0f64; 2 + pools.len()];
    let mut timed = |k: usize, f: &mut dyn FnMut()| {
        let began = std::time::Instant::now();
        f();
        ms[k] += began.elapsed().as_secs_f64() * 1e3;
    };
    const PASSES: u32 = 6;
    for pass in 0..PASSES {
        // Every slot gets another record than it has (now and then none).
        for i in 0..SLOTS {
            let view = (pass + i) as usize * 7 + (next() % 3) as usize;
            title.write(i, (next() % 23 != 0).then(|| &title.views[view % title.views.len()]));
        }
        timed(0, &mut || alone.sync_slots(0, SLOTS, title.start(), 32));
        timed(1, &mut || one.sync_slots(0, SLOTS, title.start(), 32));
        for (k, (heap, pool)) in wide.iter().zip(&pools).enumerate() {
            timed(2 + k, &mut || heap.sync_wide(pool, &all, &at));
        }
        for i in 0..SLOTS {
            let expected = shown(&one, i);
            assert_eq!(shown(&alone, i), expected, "slot {i}, pass {pass}, split locks");
            for (heap, pool) in wide.iter().zip(&pools) {
                assert_eq!(shown(heap, i), expected, "slot {i}, pass {pass}, {} threads", pool.threads());
            }
        }
    }
    let slots = f64::from(PASSES * SLOTS);
    eprintln!(
        "heap sync of {SLOTS} changed slots, {PASSES} passes, microseconds a slot: alone with split locks {:.2}, \
         alone with the one lock {:.2}, on 2, 3 and 4 threads {:.2?}",
        ms[0] * 1e3 / slots,
        ms[1] * 1e3 / slots,
        ms[2..].iter().map(|m| m * 1e3 / slots).collect::<Vec<_>>(),
    );
    // SAFETY: a plain device query.
    assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    assert_eq!(host.log_messages(), None);
    drop((wide, alone, one));
}
