//! The submitting thread's wait for the host GPU, and what it does meanwhile
//! with `VICTORY_ASYNC_SUBMIT=1` (off by default): stage 1 of overlapping
//! CPU and GPU work.
//!
//! Every submission is still waited for before [`Host::execute`] returns:
//! copy-back, the `after` work, bottom-of-pipe writes and the title's fence
//! signals still come only once the GPU has run it, in the same order. What
//! changes is the wait. Instead of sleeping until the fence, the thread
//! makes the descriptors the title copied into its shader-visible heaps as
//! bytes since the submission's own sync ([`crate::descriptor::sync_heaps`]),
//! which the next submission would otherwise make before it could start:
//! 89-130 ms of the 204-264 ms a world frame spends preparing submissions,
//! against 226-236 ms waiting for the GPU (diagnostics runs on an M5 Pro).
//!
//! That is the work title threads do already when they write a view by call
//! while the GPU runs (`crate::descriptor` `mirror`): a slot a running
//! submission reads keeps its view unless the title rewrote its record,
//! which on the console races the GPU too, and a resource a remade slot
//! stops viewing is retired until the submission has run
//! ([`crate::host::retire`]). What the GPU writes into heap memory goes back
//! after the wait through [`crate::watch::writing`], which marks the heap's
//! pages, so the next submission's own sync still sees it. A view heap made
//! and filled meanwhile ([`Host::quiet`], #31) first waits for the
//! submission to end.

use std::cell::Cell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

use windows::Win32::System::Threading::WaitForSingleObject;

use super::Host;

/// How long the wait sleeps before it looks for heap pages written again;
/// the fence's event ends it sooner.
const POLL_MS: u32 = 2;

/// How long the wait works before it leaves the rest to the plain wait
/// ([`Host::wait_for`]), which reports a hang.
const WORK_FOR: Duration = Duration::from_millis(250);

/// Whether waits do heap syncs (`VICTORY_ASYNC_SUBMIT=1`).
pub(crate) fn enabled() -> bool {
    #[cfg(test)]
    if let Some(on) = FORCED.with(Cell::get) {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = switch(std::env::var("VICTORY_ASYNC_SUBMIT").ok().as_deref());
        if on {
            tracing::info!(
                "VICTORY_ASYNC_SUBMIT=1: heap syncs run while the host GPU runs each submission"
            );
        }
        on
    })
}

/// The switch's value: on for `1` only.
fn switch(value: Option<&str>) -> bool {
    value == Some("1")
}

thread_local! {
    /// The host fence value this thread waits for while it works in the
    /// wait ([`Host::wait_overlapping`]); 0 for none.
    static IN_FLIGHT: Cell<u64> = const { Cell::new(0) };
}

/// The submission this thread works beside, if it does.
pub(crate) fn in_flight() -> Option<u64> {
    Some(IN_FLIGHT.with(Cell::get)).filter(|&v| v != 0)
}

/// Since the last frame line: waits; microseconds the thread slept in them;
/// microseconds of work done while the GPU ran; microseconds of work in
/// rounds the GPU finished during; rounds.
static WAITS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// What one wait did.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Waited {
    /// Time asleep, waiting for the GPU.
    pub(crate) blocked: Duration,
    /// Heap syncs done while the GPU still ran, and their time.
    pub(crate) rounds_running: u32,
    pub(crate) working: Duration,
    /// Time in a last sync the GPU finished during.
    pub(crate) after: Duration,
}

impl Waited {
    fn count(&self) {
        let us = |d: Duration| d.as_micros() as u64;
        WAITS[0].fetch_add(1, Relaxed);
        WAITS[1].fetch_add(us(self.blocked), Relaxed);
        WAITS[2].fetch_add(us(self.working), Relaxed);
        WAITS[3].fetch_add(us(self.after), Relaxed);
        WAITS[4].fetch_add(
            u64::from(self.rounds_running) + u64::from(!self.after.is_zero()),
            Relaxed,
        );
    }
}

impl Host {
    /// Waits until the host fence reaches `value`, a submission's: asleep,
    /// or doing heap syncs meanwhile (`overlapped`).
    pub(super) fn wait_gpu(&self, value: u64, overlapped: bool) {
        #[cfg(test)]
        if let Some(work) = DURING.with(Cell::take) {
            IN_FLIGHT.with(|f| f.set(if overlapped { value } else { 0 }));
            work();
        }
        let waited = if overlapped {
            self.wait_overlapping(value)
        } else {
            let started = Instant::now();
            self.wait_for(value);
            Waited {
                blocked: started.elapsed(),
                ..Waited::default()
            }
        };
        waited.count();
        #[cfg(test)]
        LAST.with(|l| l.set(Some(waited)));
    }

    /// The wait with heap syncs while the GPU runs: one whenever pages were
    /// written, at most every [`POLL_MS`]; the fence is checked after each
    /// sync and each wake-up (its event alone may never come, #50).
    fn wait_overlapping(&self, value: u64) -> Waited {
        let started = Instant::now();
        let mut waited = Waited::default();
        IN_FLIGHT.with(|f| f.set(value));
        // SAFETY: our fence.
        let done = || unsafe { self.fence.GetCompletedValue() } >= value;
        while !done() && started.elapsed() < WORK_FOR {
            if crate::descriptor::heaps_written() {
                let at = Instant::now();
                crate::descriptor::sync_heaps();
                let took = at.elapsed();
                if done() {
                    waited.after = took;
                    break;
                }
                waited.working += took;
                waited.rounds_running += 1;
            }
            // SAFETY: our event. A wake-up by an event an earlier
            // submission left set only means the fence is checked again.
            unsafe { WaitForSingleObject(self.event, POLL_MS) };
        }
        IN_FLIGHT.with(|f| f.set(0));
        self.finish(value);
        waited.blocked = started
            .elapsed()
            .saturating_sub(waited.working + waited.after);
        waited
    }
}

/// Logs the GPU waits since the last line, per frame over `frames`.
pub(super) fn log(frames: f64) {
    let [waits, blocked, working, after, rounds] = WAITS.each_ref().map(|c| c.swap(0, Relaxed));
    if waits == 0 || frames <= 0.0 {
        return;
    }
    let ms = |us: u64| us as f64 / 1e3 / frames;
    tracing::info!(
        "GPU waits per frame: {:.1} submissions; the submitting thread slept {:.1} ms waiting for the GPU; \
         work done meanwhile: heap syncs {:.1} ms while the GPU ran, {:.1} ms more in syncs it finished during, \
         {:.1} syncs (VICTORY_ASYNC_SUBMIT={})",
        waits as f64 / frames,
        ms(blocked),
        ms(working),
        ms(after),
        rounds as f64 / frames,
        u8::from(enabled()),
    );
}

#[cfg(test)]
thread_local! {
    /// Overrides the switch for this thread's submissions.
    static FORCED: Cell<Option<bool>> = const { Cell::new(None) };
    /// Run once in this thread's next wait, once its submission is on the
    /// GPU, before anything is waited for.
    static DURING: Cell<Option<Box<dyn FnOnce()>>> = const { Cell::new(None) };
    /// What this thread's last wait did.
    static LAST: Cell<Option<Waited>> = const { Cell::new(None) };
}

/// Sets the switch for this thread's submissions (`None`: the variable's).
#[cfg(test)]
pub(crate) fn force(on: Option<bool>) {
    FORCED.with(|f| f.set(on));
}

/// Runs `work` in this thread's next wait, with its submission on the GPU.
#[cfg(test)]
pub(crate) fn during(work: impl FnOnce() + 'static) {
    DURING.with(|d| d.set(Some(Box::new(work))));
}

/// What this thread's last wait did.
#[cfg(test)]
pub(crate) fn last() -> Option<Waited> {
    LAST.with(Cell::take)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::backend::HostGpu;
    use crate::host::marks::Spin;
    use crate::object::{S_OK, This, slot};
    use std::rc::Rc;

    /// A draw the submitting thread waits 12 ms or more for, and its loop
    /// count (under D3DMetal macOS ends one that runs about 50 ms).
    pub(crate) fn long_draw(host: &Host) -> (Spin, u32) {
        let spin = Spin::new(host, 256, 256);
        let mut n = 256u32;
        force(Some(false));
        for tag in 1.. {
            let (_, ran) = spin.draw(host, n, tag);
            assert_eq!(ran, 256 * 256, "a calibration draw runs whole");
            let gpu = last().expect("the wait ran").blocked;
            if gpu >= Duration::from_millis(12) || n >= 1 << 28 {
                break;
            }
            n *= if gpu < Duration::from_millis(3) { 4 } else { 2 };
        }
        force(None);
        (spin, n)
    }

    #[test]
    fn only_1_turns_the_switch_on() {
        assert!(switch(Some("1")));
        for off in [None, Some("0"), Some(""), Some("true"), Some("2")] {
            assert!(!switch(off), "{off:?}");
        }
    }

    /// A shader-visible heap of `count` slots (`kind` 0 views, 32 bytes a
    /// slot, or 1 samplers, 16) placed in a tracked pool, as Grand Theft
    /// Auto VI places its heaps in its own memory: (heap, pool).
    fn placed_heap(kind: u32, count: u32) -> (This, usize) {
        #[repr(C)]
        struct Placed {
            desc: [u32; 4],
            address: u64,
            size: u64,
        }
        let size = u64::from(count) * [32, 16][kind as usize];
        let memory = crate::watch::tests::pool((size as usize).div_ceil(crate::watch::PAGE));
        let iid = crate::object::guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        let placed = Placed {
            desc: [kind, count, 1, 1],
            address: memory as u64,
            size,
        };
        let mut h = std::ptr::null();
        let desc = std::ptr::from_ref(&placed).cast();
        assert_eq!(
            crate::descriptor::create_heap(std::ptr::null(), desc, &iid, &mut h),
            S_OK
        );
        (h, memory)
    }

    /// A title thread copies a sampler record into a heap slot as bytes
    /// while a submission runs on the GPU. With the switch the wait makes
    /// the slot while the GPU runs, so it is made before the submission
    /// returns, and the running draw is untouched: every pixel ran.
    #[test]
    fn a_record_copied_while_a_submission_runs_is_made_in_its_wait() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let (h, memory) = placed_heap(1, 64);
        if let Some(slots) = crate::descriptor::heap(h).host() {
            let (spin, n) = long_draw(host);
            let mut elsewhere = Box::new([0u64; 2]);
            let at = elsewhere.as_mut_ptr() as u64;
            crate::descriptor::create_sampler(
                std::ptr::null(),
                &[21, 3, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0],
                at,
            );
            let slot = memory as u64 + 5 * 16;
            let list = host.new_list().unwrap();
            spin.record(&list, n, 0x5eed);
            list.close();
            force(Some(true));
            let running = Rc::new(Cell::new(false));
            let seen = running.clone();
            during(move || {
                // SAFETY: our fence.
                seen.set(
                    unsafe { crate::host::host().unwrap().fence.GetCompletedValue() }
                        < in_flight().unwrap(),
                );
                // SAFETY: 16 bytes of each, ours; the heap's page is guarded
                // (our handler opens it).
                unsafe { std::ptr::copy_nonoverlapping(at as *const u8, slot as *mut u8, 16) };
            });
            assert!(!slots.made(5));
            host.execute(&[&list]);
            force(None);
            let waited = last().expect("the wait ran");
            assert!(running.get(), "the record was copied while the draw ran");
            assert!(
                waited.rounds_running >= 1,
                "a sync ran while the GPU did: {waited:?}"
            );
            assert!(
                slots.made(5),
                "made in the wait, before the submission returned"
            );
            assert_eq!(spin.ran(0x5eed), 256 * 256, "the running draw ran whole");
            drop(elsewhere);
        }
        slot::<extern "system" fn(This) -> u32>(h, 0x10)(h);
        crate::watch::tests::free(memory);
    }

    /// What a submission's shaders wrote reaches title memory only once the
    /// GPU has run it, then its `after` work, then its bottom-of-pipe
    /// writes, and all of it before the submission returns, with the
    /// switch as without.
    #[test]
    fn results_land_after_the_gpu_and_before_the_submission_returns() {
        use crate::watch::PAGE;
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let (spin, n) = long_draw(host);
        for on in [true, false] {
            let pool = crate::watch::tests::pool(1);
            // SAFETY: our pool, not guarded yet.
            unsafe { ((pool + 16) as *mut u8).write_volatile(0x11) };
            crate::shadow::register(pool, pool as u64, PAGE as u64);
            let copy = crate::shadow::find(pool as u64, PAGE as u64).unwrap();
            let word = Box::new(std::sync::atomic::AtomicU64::new(0));
            let word_at = std::ptr::from_ref(&*word) as usize;
            // SAFETY: a byte of our pool.
            let title = move || unsafe { ((pool + 16) as *const u8).read_volatile() };
            // SAFETY: our word, alive while the test runs.
            let value = move || unsafe { (word_at as *const u64).read_volatile() };
            let list = host.new_list().unwrap();
            let tag = 0x600d + u32::from(on);
            spin.record(&list, n, tag);
            let address = (pool + 16) as u64;
            copy.refresh(&list, address, 16);
            copy.gpu_writes(&list, address, 16, false);
            // The shader's write into the copy.
            let writer = copy.clone();
            list.defer_host(move || writer.write_host(address, &[0x22]));
            let order = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let noted = order.clone();
            list.defer_after(move || noted.lock().unwrap().push((title(), value())));
            list.defer_write(word_at as *mut u8, crate::queue::Written::Value(7), 8);
            list.close();
            force(Some(on));
            let during_run = Rc::new(Cell::new(None));
            let seen = during_run.clone();
            during(move || seen.set(Some((title(), value()))));
            host.execute(&[&list]);
            force(None);
            assert_eq!(
                during_run.get(),
                Some((0x11, 0)),
                "nothing lands while the GPU runs (switch {on})"
            );
            assert_eq!(
                *order.lock().unwrap(),
                [(0x22, 0)],
                "copy-back, then after, then the write (switch {on})"
            );
            assert_eq!(
                (title(), value()),
                (0x22, 7),
                "all landed on return (switch {on})"
            );
            assert_eq!(spin.ran(tag), 256 * 256);
            crate::shadow::unregister(pool, pool as u64);
            drop(copy);
            crate::watch::tests::free(pool);
        }
    }

    /// A view heap made and filled while the wait works beside a running
    /// submission (`Host::quiet`, #31: lists submitted during a fill lost
    /// their views) waits for that submission to end first. Wine only, where
    /// fills hold submissions off.
    #[test]
    fn a_heap_fill_in_the_wait_waits_for_the_running_submission() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        if !crate::under_wine() {
            return;
        }
        let (spin, n) = long_draw(host);
        let list = host.new_list().unwrap();
        spin.record(&list, n, 0xf111);
        list.close();
        force(Some(true));
        let ended = Rc::new(Cell::new(None));
        let seen = ended.clone();
        during(move || {
            let host = crate::host::host().unwrap();
            let value = in_flight().expect("in the wait");
            // SAFETY: our fence.
            let before = unsafe { host.fence.GetCompletedValue() } >= value;
            let quiet = host.quiet();
            // SAFETY: as above.
            let after = unsafe { host.fence.GetCompletedValue() } >= value;
            seen.set(Some((before, quiet.is_none(), after)));
        });
        host.execute(&[&list]);
        force(None);
        assert_eq!(
            ended.get(),
            Some((false, true, true)),
            "running, then ended by the time the fill may start"
        );
        assert_eq!(spin.ran(0xf111), 256 * 256);
    }

    /// Not a check: what the wait hides. A title thread copies view records
    /// into the slots of a 16,384-slot view heap while a submission runs a
    /// draw of 12 ms or more, the switch on and off by turns; an empty
    /// submission after each makes what is left, as the next one would.
    /// Prints, per round: the time of both submissions, the wait, and the
    /// heap syncs done in it while the GPU ran and after it had finished.
    /// `tools/macos/test-wine.sh --release -p victory-gpu --lib
    /// heap_syncs_beside_the_gpu -- --ignored --nocapture`
    #[test]
    #[ignore = "a benchmark: run alone, in release"]
    fn heap_syncs_beside_the_gpu() {
        const SLOTS: usize = 16_384;
        const ROUNDS: u32 = 60;
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let (h, memory) = placed_heap(0, SLOTS as u32);
        if crate::descriptor::heap(h).host().is_some() {
            let (spin, n) = long_draw(host);
            // Two records (constant buffer views of no memory) in the last
            // two slots, copied into the others by turns, so that each of
            // them changes at every round.
            let records = [SLOTS - 2, SLOTS - 1].map(|i| memory + 32 * i);
            for (k, &at) in records.iter().enumerate() {
                crate::descriptor::create_cbv(std::ptr::null(), &[0, 0, 256 << k, 0], at as u64);
            }
            // (time, wait, syncs while the GPU ran, syncs it finished during), by switch.
            let mut sums = [[Duration::ZERO; 4]; 2];
            // The first rounds make every slot for the first time.
            for round in 0..ROUNDS + 4 {
                let on = round % 2 == 1;
                let list = host.new_list().unwrap();
                spin.record(&list, n, 0xbe00 + round);
                list.close();
                force(Some(on));
                let record = records[round as usize % 2];
                during(move || {
                    for i in 0..SLOTS - 2 {
                        // SAFETY: 32 bytes of our heap's record and slot.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                record as *const u8,
                                (memory + 32 * i) as *mut u8,
                                32,
                            )
                        };
                    }
                });
                let started = Instant::now();
                host.execute(&[&list]);
                let w = last().expect("the wait ran");
                force(Some(false));
                let flush = host.new_list().unwrap();
                flush.close();
                host.execute(&[&flush]);
                force(None);
                if round >= 4 {
                    let s = &mut sums[usize::from(on)];
                    for (sum, d) in s.iter_mut().zip([
                        started.elapsed(),
                        w.blocked + w.working + w.after,
                        w.working,
                        w.after,
                    ]) {
                        *sum += d;
                    }
                }
            }
            for (on, s) in sums.iter().enumerate() {
                let [time, wait, running, after] =
                    s.map(|d| d.as_secs_f64() * 2e3 / f64::from(ROUNDS));
                println!(
                    "switch {on}: {time:.2} ms a round, {wait:.2} ms of it in the wait; heap syncs in the wait {running:.2} ms while the GPU ran, {after:.2} ms after it had finished"
                );
            }
        }
        slot::<extern "system" fn(This) -> u32>(h, 0x10)(h);
        crate::watch::tests::free(memory);
    }
}
