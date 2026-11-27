//! mac: the plain arithmetic of the flights (`crate::host::unwaited::flights`,
//! `VICTORY_FLIGHTS=1`) and of their frame lines: plain std, so its tests
//! run everywhere.

/// Title memory as ranges `[from, to)`: after [`Spans::join`], in order,
/// none touching the next.
#[derive(Debug, Default, Clone)]
pub(crate) struct Spans(Vec<(usize, usize)>);

impl Spans {
    /// Adds `[from, to)` (nothing if empty); [`Spans::join`] after the last.
    pub(crate) fn add(&mut self, from: usize, to: usize) {
        if from < to {
            self.0.push((from, to));
        }
    }

    /// Puts the ranges in order, those that overlap or touch as one.
    pub(crate) fn join(&mut self) {
        self.0.sort_unstable();
        let mut out: Vec<(usize, usize)> = Vec::with_capacity(self.0.len());
        for &(from, to) in &self.0 {
            match out.last_mut() {
                Some(last) if from <= last.1 => last.1 = last.1.max(to),
                _ => out.push((from, to)),
            }
        }
        self.0 = out;
    }

    /// Whether `[from, from + len)` shares a byte with one of them (joined).
    pub(crate) fn touches(&self, from: usize, len: usize) -> bool {
        let to = from.saturating_add(len);
        if from >= to {
            return false;
        }
        let at = self.0.partition_point(|r| r.1 <= from);
        self.0.get(at).is_some_and(|r| r.0 < to)
    }

    /// Whether they hold nothing.
    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[cfg(test)]
    fn ranges(&self) -> &[(usize, usize)] {
        &self.0
    }
}

/// What the `bytes` (1 to 8) at `address` will hold once `writes` are made
/// in their order (each: where, the value it stores if that is known now,
/// how many of its low bytes it stores, 4 or 8, little-endian):
///
/// - `None`: no write reaches those bytes, memory decides;
/// - `Some(Some(value))`: the last write that reaches them stores all of
///   them, and its value is known;
/// - `Some(None)`: a write reaches them that cannot be told now (a time
///   stamp, or one that stores only a part of them and leaves the rest to
///   memory or to an earlier write).
pub(crate) fn after_writes(writes: &[(usize, Option<u64>, usize)], address: usize, bytes: usize) -> Option<Option<u64>> {
    let end = address.checked_add(bytes)?;
    if bytes == 0 || bytes > 8 {
        return None;
    }
    let mut seen = None;
    for &(at, value, n) in writes {
        let Some(stop) = at.checked_add(n) else { continue };
        if stop <= address || end <= at {
            continue;
        }
        seen = Some(match value {
            Some(v) if at <= address && end <= stop => {
                // The low `n` bytes of `v`, then the ones asked for.
                let stored = if n >= 8 { v } else { v & ((1u64 << (8 * n)) - 1) };
                let shifted = stored >> (8 * (address - at));
                Some(if bytes >= 8 { shifted } else { shifted & ((1u64 << (8 * bytes)) - 1) })
            }
            _ => None,
        });
    }
    seen
}

/// mac: the names `VICTORY_FLIGHTS_WITH=<list>` takes, each building on the
/// ones before it (`notes/overlap-stages-design.md` §6).
pub(crate) const WITH: [&str; 6] = ["alias", "writes", "copies", "ring", "relay", "present"];

/// `VICTORY_FLIGHTS_WITH`'s rule: of the comma list `value`, the names taken
/// (bit `i` for [`WITH`]`[i]`), and a line for each one refused: unknown, not
/// in this build (`built`, the bits it has), asked without every name before
/// it, or refused by `refuse(i)` for a reason of its own.
pub(crate) fn with_names(value: &str, built: u32, refuse: impl Fn(usize) -> Option<&'static str>) -> (u32, Vec<String>) {
    let mut asked = 0u32;
    let mut refused = Vec::new();
    for name in value.split(',').map(str::trim).filter(|n| !n.is_empty()) {
        match WITH.iter().position(|w| w.eq_ignore_ascii_case(name)) {
            Some(i) => asked |= 1 << i,
            None => refused.push(format!("{name}: not a name it takes ({})", WITH.join(","))),
        }
    }
    let mut taken = 0u32;
    for (i, name) in WITH.iter().enumerate().filter(|&(i, _)| asked & 1 << i != 0) {
        let why = if built & 1 << i == 0 {
            Some("not in this build".to_owned())
        } else if taken != (1 << i) - 1 {
            Some(format!("it needs {} first", WITH[..i].join(",")))
        } else {
            refuse(i).map(str::to_owned)
        };
        match why {
            Some(why) => refused.push(format!("{name}: {why}")),
            None => taken |= 1 << i,
        }
    }
    (taken, refused)
}

/// mac: E1 of `notes/overlap-stages-design.md` (lines only): per thread that
/// calls us, the microseconds inside our queue calls (the outermost one),
/// waiting for the held-work lock and waiting for the submission lock, since
/// the last frame line. A thread takes its slot once, without a lock; past
/// [`threads::SLOTS`] threads share the last one (id 0).
pub(crate) mod threads {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
    use std::time::Instant;

    pub(crate) const SLOTS: usize = 16;

    /// What a thread's time went on.
    #[derive(Clone, Copy)]
    pub(crate) enum Spent {
        Inside = 1,
        Held = 2,
        Lock = 3,
    }

    /// Per slot: the thread's id, then microseconds by [`Spent`].
    static TABLE: [[AtomicU64; 4]; SLOTS] = [const { [const { AtomicU64::new(0) }; 4] }; SLOTS];
    static CLAIMED: AtomicUsize = AtomicUsize::new(0);

    thread_local! {
        static SLOT: Cell<usize> = const { Cell::new(usize::MAX) };
        static DEPTH: Cell<u32> = const { Cell::new(0) };
    }

    /// The thread's id as the table shows it (the Windows thread id).
    pub(crate) fn id() -> u64 {
        #[cfg(windows)]
        // SAFETY: a plain query.
        return u64::from(unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() });
        #[cfg(not(windows))]
        {
            static NEXT: AtomicU64 = AtomicU64::new(1);
            thread_local!(static ID: u64 = NEXT.fetch_add(1, Relaxed));
            ID.with(|i| *i)
        }
    }

    fn slot() -> usize {
        SLOT.with(|s| {
            if s.get() == usize::MAX {
                let i = CLAIMED.fetch_add(1, Relaxed).min(SLOTS - 1);
                TABLE[i][0].store(if i < SLOTS - 1 { id() } else { 0 }, Relaxed);
                s.set(i);
            }
            s.get()
        })
    }

    pub(crate) fn add(spent: Spent, us: u64) {
        TABLE[slot()][spent as usize].fetch_add(us, Relaxed);
    }

    /// Times the thread's outermost call of ours while it lives.
    pub(crate) struct Inside(Option<Instant>);

    impl Inside {
        pub(crate) fn enter() -> Inside {
            let outermost = DEPTH.with(|d| {
                d.set(d.get() + 1);
                d.get() == 1
            });
            Inside(outermost.then(Instant::now))
        }
    }

    impl Drop for Inside {
        fn drop(&mut self) {
            DEPTH.with(|d| d.set(d.get() - 1));
            if let Some(at) = self.0 {
                add(Spent::Inside, at.elapsed().as_micros() as u64);
            }
        }
    }

    /// Since the last call, per thread with anything: its id and the
    /// microseconds inside, held, lock (then zero).
    pub(crate) fn take() -> Vec<(u64, [u64; 3])> {
        (0..CLAIMED.load(Relaxed).min(SLOTS))
            .filter_map(|i| {
                let us = [1, 2, 3].map(|k| TABLE[i][k].swap(0, Relaxed));
                (us != [0; 3]).then(|| (TABLE[i][0].load(Relaxed), us))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `VICTORY_FLIGHTS_WITH`: names in their order, each needing those
    /// before it; unknown, unbuilt and otherwise refused names named.
    #[test]
    fn flights_with_takes_each_name_after_the_ones_before_it() {
        let none = |_: usize| None;
        assert_eq!(with_names("", 1, none), (0, vec![]));
        assert_eq!(with_names(" alias , ", 1, none), (1, vec![]));
        assert_eq!(with_names("ALIAS", 1, none).0, 1);
        let (taken, refused) = with_names("alias,writes", 1, none);
        assert_eq!(taken, 1);
        assert_eq!(refused, ["writes: not in this build"]);
        let (taken, refused) = with_names("writes", 0b11, none);
        assert_eq!(taken, 0);
        assert_eq!(refused, ["writes: it needs alias first"]);
        assert_eq!(with_names("writes,alias", 0b11, none), (0b11, vec![]));
        let (taken, refused) = with_names("alias,copies", 0b111, none);
        assert_eq!(taken, 1);
        assert_eq!(refused, ["copies: it needs alias,writes first"]);
        let (taken, refused) = with_names("alias,bogus", 1, none);
        assert_eq!(taken, 1);
        assert!(refused[0].starts_with("bogus: not a name"), "{refused:?}");
        let (taken, refused) = with_names("alias,writes", 0b11, |i| (i == 0).then_some("the heap worker"));
        assert_eq!(taken, 0);
        assert_eq!(refused, ["alias: the heap worker", "writes: it needs alias first"]);
    }

    /// E1's table: each thread its own slot, only the outermost call timed,
    /// the waits added as they come, zero after a take.
    #[test]
    fn each_thread_counts_its_outermost_call_and_its_waits() {
        use threads::{Inside, Spent, add, id, take};
        let ids: Vec<u64> = (0..2u64)
            .map(|k| {
                std::thread::spawn(move || {
                    let outer = Inside::enter();
                    {
                        let _inner = Inside::enter();
                        std::thread::sleep(std::time::Duration::from_millis(20));
                    }
                    drop(outer);
                    add(Spent::Held, 1000 * (k + 1));
                    add(Spent::Lock, 7);
                    id()
                })
                .join()
                .unwrap()
            })
            .collect();
        assert_ne!(ids[0], ids[1]);
        let seen = take();
        for (k, me) in ids.iter().enumerate() {
            let &(_, [inside, held, lock]) = seen.iter().find(|(i, _)| i == me).expect("the thread's slot");
            assert!((20_000..1_000_000).contains(&inside), "once, not twice: {inside}");
            assert_eq!((held, lock), (1000 * (k as u64 + 1), 7));
        }
        assert!(take().iter().all(|(i, _)| !ids.contains(i)), "zero after a take");
    }

    #[test]
    fn spans_join_where_they_touch_and_find_what_they_share() {
        let mut s = Spans::default();
        s.add(100, 200);
        s.add(300, 400);
        s.add(200, 250);
        s.add(500, 500);
        s.add(350, 360);
        s.join();
        assert_eq!(s.ranges(), &[(100, 250), (300, 400)]);
        assert!(s.touches(249, 1));
        assert!(!s.touches(250, 50));
        assert!(s.touches(250, 51));
        assert!(!s.touches(0, 100));
        assert!(s.touches(0, 101));
        assert!(!s.touches(400, 1000));
        assert!(!s.touches(120, 0));
        assert!(!Spans::default().touches(0, usize::MAX));
    }

    #[test]
    fn a_wait_reads_what_the_last_write_over_it_stores() {
        // Nothing there: memory decides.
        assert_eq!(after_writes(&[(0x100, Some(7), 8)], 0x200, 4), None);
        // Whole: an 8-byte value read as 8 and as its two halves.
        let w = [(0x100, Some(0x1111_2222_3333_4444), 8)];
        assert_eq!(after_writes(&w, 0x100, 8), Some(Some(0x1111_2222_3333_4444)));
        assert_eq!(after_writes(&w, 0x100, 4), Some(Some(0x3333_4444)));
        assert_eq!(after_writes(&w, 0x104, 4), Some(Some(0x1111_2222)));
        // A 4-byte write stores the low half of its value only.
        let w = [(0x100, Some(0xdead_beef_0000_0005), 4)];
        assert_eq!(after_writes(&w, 0x100, 4), Some(Some(5)));
        // Read wider than it stores: the rest is memory's, not known.
        assert_eq!(after_writes(&w, 0x100, 8), Some(None));
        // A time stamp: not known.
        assert_eq!(after_writes(&[(0x100, None, 8)], 0x100, 8), Some(None));
        // In order: the later write decides, known or not.
        let w = [(0x100, Some(1), 8), (0x100, Some(2), 8)];
        assert_eq!(after_writes(&w, 0x100, 8), Some(Some(2)));
        let w = [(0x100, Some(1), 8), (0x100, None, 8)];
        assert_eq!(after_writes(&w, 0x100, 8), Some(None));
        let w = [(0x100, None, 8), (0x100, Some(3), 8)];
        assert_eq!(after_writes(&w, 0x100, 8), Some(Some(3)));
        // A later write over a part of it: not known.
        let w = [(0x100, Some(1), 8), (0x104, Some(9), 4)];
        assert_eq!(after_writes(&w, 0x100, 8), Some(None));
        assert_eq!(after_writes(&w, 0x104, 4), Some(Some(9)));
        assert_eq!(after_writes(&w, 0x100, 4), Some(Some(1)));
        // Nothing asked, or more than a value holds.
        assert_eq!(after_writes(&[(0x100, Some(1), 8)], 0x100, 0), None);
        assert_eq!(after_writes(&[(0x100, Some(1), 8)], 0x100, 9), None);
    }
}
