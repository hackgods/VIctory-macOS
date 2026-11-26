//! Title submissions the host GPU is not waited for
//! (`VICTORY_ASYNC_SUBMIT=1`; off by default, and then nothing here runs).
//!
//! The idea and its settle points are Belobug's (fork commits 2bfe95a,
//! a049b48, eebcf07, 9b89ec2, bdecda3, cf57ca3, a59f3e5), written again for
//! this line's submissions.
//!
//! [`Host::run`] prepares a submission of the title's, gives it to the host
//! GPU and returns ([`Host::leave`]), so the title's thread goes on while
//! the GPU runs it. What the submission still owes (the check that it ran,
//! what its shaders wrote copied back into title memory, its lists' work
//! for after, their bottom-of-pipe writes, the retired resources, its lists
//! given back) is done by [`Host::finish`], the same code as for a
//! submission that was waited for, when something waits for it:
//!
//! - the next submission, the title's or ours (the present's copy, texture
//!   uploads, captures), before anything of it is prepared
//!   ([`Host::submissions`]): one submission at a time from preparation to
//!   copy-back, as before, and ours always wait, as they read or free what
//!   they made as soon as they return;
//! - a queue's Signal, before the title's fence moves (`crate::queue`),
//!   if the submission is that queue's ([`settle_signal`]);
//! - a queue's head whose wait does not hold, if the submission owes a
//!   bottom-of-pipe write where the wait reads (`crate::hold::World::owed`,
//!   [`settle_waits`]); with `VICTORY_ASYNC_SETTLE_ALL=1` every Signal and
//!   every such head, as in the fork;
//! - a present, before the picture is looked at (`crate::queue`);
//! - a queue's page mapping (`CopyPageMappingsBatchX`), which the console
//!   does after the queue's work before it: what that work owes title
//!   memory goes where the pages still are;
//! - [`Host::quiet`], which waits for the GPU alone ([`Host::stand`]): the
//!   rest is left to the next of the above, as its caller holds locks;
//! - failing all of those, a thread of ours ([`idle`]) once the GPU has
//!   run it: the title may read memory for a bottom-of-pipe write (or for
//!   what its shaders wrote) without calling us, as it may on the console,
//!   where the GPU writes it with nobody waiting.
//!
//! At most one submission is left so, and the submission lock is taken by
//! nobody without one of these, so everything that says "the last
//! submission has run" under that lock still holds. Its lists stay held
//! here until it is finished: a title list reset or released meanwhile
//! leaves its host list to us (`crate::list::restart`), as for a list a
//! queue holds.
//!
//! Left out of the fork's change: refreshing the next submission's buffer
//! copies while the one before runs. There `Shadow::update` goes around the
//! ranges a running submission's shaders write; here it copies under them
//! and into their snapshot, which is right only once that submission has
//! been copied back (`crate::shadow::Shadow::update_tracked`).

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release};

use super::*;

/// Whether the title's submissions may return before the host GPU has run
/// them (`VICTORY_ASYNC_SUBMIT=1`).
pub(crate) fn on() -> bool {
    #[cfg(test)]
    if tests::FORCED.load(Relaxed) {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = crate::switch::on("VICTORY_ASYNC_SUBMIT");
        if on {
            tracing::info!(
                "VICTORY_ASYNC_SUBMIT: the title's submissions return before the host GPU has run them, \
                 and are finished when something waits for them"
            );
        }
        on
    })
}

/// Whether every Signal and every wait that does not hold finishes the
/// submission left, whatever queue it is of and whatever it owes
/// (`VICTORY_ASYNC_SETTLE_ALL=1`): the fork's rule, to compare with.
fn all() -> bool {
    static ALL: OnceLock<bool> = OnceLock::new();
    *ALL.get_or_init(|| {
        let all = crate::switch::on("VICTORY_ASYNC_SETTLE_ALL");
        if all {
            tracing::info!(
                "VICTORY_ASYNC_SETTLE_ALL: every queue Signal and every wait that does not hold \
                 finishes the submission left to the host GPU"
            );
        }
        all
    })
}

/// Whether a prepared submission of the title's may be left: not one whose
/// lists run one by one (`VICTORY_GPU_TIMING`), nor in a listed frame, nor
/// while frames are captured (diagnostics builds; the dumps submit lists of
/// their own under their own locks).
pub(super) fn may(groups: usize, traced: bool) -> bool {
    #[cfg(feature = "diagnostics")]
    if crate::capture::folder().is_some() {
        return false;
    }
    on() && groups == 1 && !traced
}

/// What finished a submission left to the host GPU.
#[derive(Clone, Copy, Debug)]
pub(crate) enum By {
    /// The title's next submission.
    Next,
    /// A submission of ours.
    Ours,
    /// A queue's Signal.
    Signal,
    /// A queue's head whose wait did not hold.
    Wait,
    /// A present.
    Present,
    /// A queue's page mapping.
    Mapping,
    /// A view heap being made (the wait for the GPU alone).
    Quiet,
    /// Nothing: our thread, once the host GPU had run it.
    Idle,
}

const WAYS: usize = 8;

/// A submission given to the host GPU and not waited for.
struct Left {
    /// Its value on the host fence.
    value: u64,
    /// Whether the fence was told to reach it (else nothing is waited for,
    /// as for a waited submission whose signal failed).
    signalled: bool,
    /// The host GPU was already waited for ([`Host::stand`]).
    stood: bool,
    /// The title's lists, held until it is finished.
    lists: Vec<std::sync::Arc<HostList>>,
    /// Where their bottom-of-pipe writes go (address, bytes).
    writes: Vec<(u64, u64)>,
    ran: Ran,
}

/// The submission left, under the submission lock.
static LEFT: Mutex<Option<Left>> = Mutex::new(None);

/// Its fence value while it is not finished, else 0: read without a lock.
static VALUE: AtomicU64 = AtomicU64::new(0);

/// The title queue it is of (meaningful while [`VALUE`] is not 0).
static QUEUE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Since the last frame line: Signals of other queues, and waits it owed
/// no write to, that went on without finishing it.
static SPARED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Lists of finished submissions, to give back (`crate::list::recycle`) by
/// a thread that does not hold the submission lock, as before.
static DONE: Mutex<Vec<std::sync::Arc<HostList>>> = Mutex::new(Vec::new());

/// Submissions left and finished so far.
static FINISHED: AtomicU64 = AtomicU64::new(0);

/// Since the last frame line: submissions left; then, by what finished
/// them ([`By`]), how many and the microseconds it waited for the host GPU.
static LEAVES: AtomicU64 = AtomicU64::new(0);
/// And the microseconds it took to give them to the host GPU (the calls
/// that submit, which a runtime may do much of its own work in).
static GIVING_US: AtomicU64 = AtomicU64::new(0);
static FINISHES: [AtomicU64; WAYS] = [const { AtomicU64::new(0) }; WAYS];
static WAITED_US: [AtomicU64; WAYS] = [const { AtomicU64::new(0) }; WAYS];

fn take_done() -> Vec<std::sync::Arc<HostList>> {
    std::mem::take(&mut *DONE.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Submissions left and finished so far ([`settle_waits`]).
pub(crate) fn finished() -> u64 {
    FINISHED.load(Acquire)
}

/// Finishes the submission left to the host GPU, if there is one: waits
/// for it and does what it owes. Whether there was one. Nothing on a thread
/// that holds the submission lock (inside a submission none is left).
pub(crate) fn settle(by: By) -> bool {
    if VALUE.load(Acquire) == 0 || HOLDS_SUBMISSIONS.with(std::cell::Cell::get) {
        return false;
    }
    let Some(host) = host() else { return false };
    drop(host.submissions(by));
    take_done().into_iter().for_each(crate::list::recycle);
    true
}

/// Before `queue`'s Signal moves the title's fence: [`settle`], if the
/// submission left is that queue's. A fence says the queue's own work
/// before the Signal has run, and all of it but the submission left has
/// been finished; another queue's is not what the fence is about.
pub(crate) fn settle_signal(queue: usize) -> bool {
    if VALUE.load(Acquire) == 0 {
        return false;
    }
    // Read after the value: a submission of `queue` cannot be left
    // meanwhile, as its Signal and its submissions come one at a time.
    if QUEUE.load(Acquire) != queue && !all() {
        SPARED[0].fetch_add(1, Relaxed);
        return false;
    }
    settle(By::Signal)
}

/// For a queue's head whose `waits` do not all hold: [`settle`], if the
/// submission left owes a bottom-of-pipe write where one of them reads.
/// Whether any submission left was finished since `seen` ([`finished`],
/// brought up to date), by this call or by another thread: the waits are
/// then to be looked at again.
///
/// A wait for memory the submission changes some other way (what its
/// shaders write, a read-back) is not seen here: its head is held until
/// the submission is finished by something else ([`idle`] at the latest),
/// and runs at the queues' next look (their timer's, within 2 ms).
pub(crate) fn settle_waits(waits: &[crate::hold::ValueWait], seen: &mut u64) -> bool {
    if VALUE.load(Acquire) != 0 {
        let owed = match LEFT.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            Some(left) => left
                .writes
                .iter()
                .any(|&(at, n)| waits.iter().any(|w| w.reads(at, n))),
            // Being finished right now: `settle` waits for that.
            None => true,
        };
        if owed || all() {
            settle(By::Wait);
        } else {
            SPARED[1].fetch_add(1, Relaxed);
        }
    }
    let now = finished();
    std::mem::replace(seen, now) != now
}

/// The thread that finishes a submission left to the host GPU when nothing
/// else has by the time the GPU has run it; `None` if it could not be made
/// (then only the callers of [`settle`] and the submission lock finish it).
fn idler() -> Option<&'static std::thread::Thread> {
    static IDLER: OnceLock<Option<std::thread::Thread>> = OnceLock::new();
    IDLER
        .get_or_init(|| {
            std::thread::Builder::new()
                .name("victory unwaited".into())
                .spawn(idle)
                .map(|made| made.thread().clone())
                .inspect_err(|e| tracing::warn!(error = %e, "no thread to finish host submissions nothing waits for"))
                .ok()
        })
        .as_ref()
}

/// [`idler`]'s work: sleeps while no submission is left; while one is,
/// looks at the host fence every millisecond and finishes the submission
/// once the GPU has run it.
fn idle() {
    loop {
        let value = VALUE.load(Acquire);
        if value == 0 {
            // Woken by `Host::leave`, after it set the value.
            std::thread::park();
            continue;
        }
        let Some(host) = host() else { return };
        // SAFETY: a plain query of our fence.
        if unsafe { host.fence.GetCompletedValue() } >= value {
            settle(By::Idle);
        } else {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

/// The frame line of submissions left to the host GPU (`frames` since the
/// last one).
pub(super) fn log(frames: f64) {
    if !on() {
        return;
    }
    let left = LEAVES.swap(0, Relaxed) as f64 / frames;
    let giving = GIVING_US.swap(0, Relaxed) as f64 / 1000.0 / frames;
    let by: [f64; WAYS] = std::array::from_fn(|i| FINISHES[i].swap(0, Relaxed) as f64 / frames);
    let ms: [f64; WAYS] =
        std::array::from_fn(|i| WAITED_US[i].swap(0, Relaxed) as f64 / 1000.0 / frames);
    let spared: [f64; 2] = std::array::from_fn(|i| SPARED[i].swap(0, Relaxed) as f64 / frames);
    tracing::info!(
        "unwaited submissions: {left:.1}/frame left to the host GPU, which took {giving:.1} ms/frame to give them \
         to it; finished by (the title's next submission, \
         one of ours, a queue's signal, a held queue's wait, the present, a page mapping, a view heap being made, \
         nothing but our thread once it had run): \
         {by:.1?} per frame, after waiting {ms:.1?} ms/frame for the host GPU (the frame line's GPU time is these, \
         the giving, and our own submissions' waits); {spared:.1?} per frame went on without finishing it \
         (Signals of other queues, waits it owed no write to)"
    );
}

impl Host {
    /// The submission lock, once the submission left to the host GPU (if
    /// any) is finished: the holder prepares or submits with none running.
    pub(super) fn submissions(&self, by: By) -> std::sync::MutexGuard<'_, u64> {
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        if VALUE.load(Acquire) != 0 {
            self.finish_left(&mut submitted, by);
        }
        submitted
    }

    /// Runs the title's own lists (a queue's piece, `crate::queue`), and
    /// returns the lists that have run, to give back: these when they were
    /// waited for, else those of submissions finished since.
    pub(crate) fn execute_title(
        &self,
        queue: usize,
        lists: Vec<std::sync::Arc<HostList>>,
    ) -> Vec<std::sync::Arc<HostList>> {
        let runs: Vec<&HostList> = lists.iter().map(|l| &**l).collect();
        if !on() {
            self.execute(&runs);
            drop(runs);
            return lists;
        }
        let left = self.run(&runs, Some((queue, &lists)));
        drop(runs);
        let mut done = take_done();
        if !left {
            done.extend(lists);
        }
        done
    }

    /// Gives a prepared submission (`lists`, then the mark) to the host
    /// GPU and leaves it there. Under the submission lock, with none left.
    pub(super) fn leave(
        &self,
        submitted: &mut u64,
        lists: &[Option<ID3D12CommandList>],
        mark: Option<&Mark>,
        (title_queue, title): (usize, &[std::sync::Arc<HostList>]),
        ran: Ran,
    ) {
        *submitted += 1;
        let started = Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::resource::life::note(0, crate::resource::life::SUBMITTED, *submitted, 0);
        // Last, the list that shows the submission ran.
        let mut group = lists.to_vec();
        if let Some(mark) = mark {
            mark.arm(*submitted);
            group.push(mark.list());
        }
        let queue = self.queue();
        // SAFETY: closed lists; the fence is ours.
        let signalled = unsafe {
            queue.ExecuteCommandLists(&group);
            queue.Signal(&self.fence, *submitted)
        };
        match &signalled {
            // The event is asked for now, as for a waited submission, and
            // only wakes the wait: the fence's value decides
            // ([`Host::wait_left`]).
            Ok(()) => {
                // SAFETY: the fence and event are ours.
                let asked = unsafe { self.fence.SetEventOnCompletion(*submitted, self.event) };
                if let Err(e) = asked {
                    tracing::warn!(error = %e, "no event for a host submission: its fence is polled");
                }
            }
            Err(e) => tracing::warn!(error = %e, "host submission not waited for"),
        }
        // Time on the host GPU's side, as when the submission is waited
        // for: the frame line's GPU time.
        let giving = started.elapsed().as_micros() as u64;
        GIVING_US.fetch_add(giving, Relaxed);
        SUBMISSION_US[1].fetch_add(giving, Relaxed);
        LEAVES.fetch_add(1, Relaxed);
        let writes = title
            .iter()
            .flat_map(|l| {
                let writes = &l.deferred().writes;
                writes
                    .iter()
                    .map(|&(at, _, bytes)| (at as u64, bytes as u64))
                    .collect::<Vec<_>>()
            })
            .collect();
        *LEFT.lock().unwrap_or_else(|e| e.into_inner()) = Some(Left {
            value: *submitted,
            signalled: signalled.is_ok(),
            stood: false,
            lists: title.to_vec(),
            writes,
            ran,
        });
        QUEUE.store(title_queue, Release);
        VALUE.store(*submitted, Release);
        if let Some(idler) = idler() {
            idler.unpark();
        }
    }

    /// Waits until the host fence reaches `value`: how long. As
    /// [`Host::wait_for`], the fence decides and the event only wakes; the
    /// event was asked for when the submission was left, so it may be set
    /// already, or (under D3DMetal, #50) never: the fence is looked at
    /// first and then every 2 ms.
    fn wait_left(&self, value: u64) -> Duration {
        let started = Instant::now();
        let mut seconds = 1;
        // SAFETY: our fence and event.
        while unsafe { self.fence.GetCompletedValue() } < value {
            // SAFETY: as above.
            unsafe { WaitForSingleObject(self.event, 2) };
            if started.elapsed().as_secs() >= seconds {
                seconds *= 2;
                // SAFETY: as above.
                let completed = unsafe { self.fence.GetCompletedValue() };
                tracing::warn!(
                    seconds = started.elapsed().as_secs(),
                    value,
                    completed,
                    "still waiting for the host GPU (a submission left to it)"
                );
            }
        }
        started.elapsed()
    }

    /// [`Host::quiet`]'s part: the host GPU has run the submission left to
    /// it, which stays to be finished. Under the submission lock.
    pub(super) fn stand(&self) {
        if VALUE.load(Acquire) == 0 {
            return;
        }
        let left = || LEFT.lock().unwrap_or_else(|e| e.into_inner());
        let value = left()
            .as_ref()
            .filter(|l| l.signalled && !l.stood)
            .map(|l| l.value);
        if let Some(value) = value {
            let waited = self.wait_left(value).as_micros() as u64;
            // The same one: it is only finished under the lock we hold.
            if let Some(left) = left().as_mut() {
                left.stood = true;
            }
            WAITED_US[By::Quiet as usize].fetch_add(waited, Relaxed);
            FINISHES[By::Quiet as usize].fetch_add(1, Relaxed);
            SUBMISSION_US[1].fetch_add(waited, Relaxed);
        }
    }

    /// Waits for the submission left to the host GPU and finishes it, as
    /// [`Host::run`] does one it waited for. Under the submission lock.
    fn finish_left(&self, submitted: &mut u64, by: By) {
        let Some(left) = LEFT.lock().unwrap_or_else(|e| e.into_inner()).take() else {
            // None is being finished either: that is done under our lock.
            VALUE.store(0, Release);
            return;
        };
        let _holding = Holding::enter();
        let Left {
            value,
            signalled,
            stood,
            lists,
            mut ran,
            ..
        } = left;
        if signalled && !stood {
            ran.gpu = self.wait_left(value);
            WAITED_US[by as usize].fetch_add(ran.gpu.as_micros() as u64, Relaxed);
            SUBMISSION_US[1].fetch_add(ran.gpu.as_micros() as u64, Relaxed);
        }
        if !stood {
            FINISHES[by as usize].fetch_add(1, Relaxed);
        }
        if signalled
            && let Some(mark) = self.mark()
            && !mark.saw(value)
            // SAFETY: a plain query.
            && unsafe { self.device.GetDeviceRemovedReason() }.is_ok()
        {
            self.lost(submitted, mark);
        }
        #[cfg(feature = "diagnostics")]
        crate::resource::life::note(0, crate::resource::life::RAN, value, 0);
        {
            let runs: Vec<&HostList> = lists.iter().map(|l| &**l).collect();
            self.finish(None, &runs, ran);
        }
        DONE.lock().unwrap_or_else(|e| e.into_inner()).extend(lists);
        FINISHED.fetch_add(1, AcqRel);
        // Last: until now a thread that asks ([`settle`]) waits for our lock.
        VALUE.store(0, Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Turns the switch on for the tests here, whatever the environment.
    pub(super) static FORCED: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);

    /// A title submission left to the host GPU owes its bottom-of-pipe
    /// write until something waits for it; then the write is there, the
    /// mark shows it ran, and its list comes back.
    #[test]
    fn a_submission_left_is_finished_by_what_waits_for_it() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        FORCED.store(true, Relaxed);
        let word = Box::new(AtomicU64::new(0));
        let list = std::sync::Arc::new(host.new_list().unwrap());
        list.defer_write(word.as_ptr().cast(), crate::queue::Written::Value(7), 8);
        list.close();
        let back = host.execute_title(1, vec![list.clone()]);
        // Left (or waited for, in a diagnostics build that captures): a
        // Signal finishes it, if another test's submission has not.
        settle(By::Signal);
        FORCED.store(false, Relaxed);
        assert_eq!(word.load(Acquire), 7, "the bottom-of-pipe write was made");
        // Its list is held by nobody else once the finished ones are given
        // up (our thread may be giving it up just now).
        drop(back);
        drop(take_done());
        let asked = Instant::now();
        while std::sync::Arc::strong_count(&list) > 1 && asked.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(std::sync::Arc::strong_count(&list), 1, "the list came back");
        // Under the lock nothing is left, and the last submission ran.
        let submitted = host.submissions(By::Ours);
        assert_eq!(VALUE.load(Acquire), 0);
        if let Some(mark) = host.mark() {
            assert!(
                mark.saw(*submitted),
                "the mark shows submission {}",
                *submitted
            );
        }
    }
}
