//! mac: `VICTORY_CENSUS_FAST=1` (`crate::mac_switches::census_fast`, off by
//! default): a view heap's census (`crate::bindless::Viewed`) held under its
//! lock for less time and taken less often.
//!
//! The census lock was the largest serial resource of a frame at the prison
//! cell (`notes/profile-first-run.md`: per frame 1,025 waits for 243.9 ms by
//! the title's threads, 1,405 for 117.5 ms by heap syncs). The user that held
//! it longest: the ranges a heap's unordered views write
//! (`bindless::widened_writes`) were found by walking every range of the
//! census (130,000 to 150,000 in the world, five in six of them not
//! unordered) under the lock, once for every heap-writing draw of a list at
//! its next barrier. Here the census keeps its unordered ranges apart
//! ([`Unordered`], changed exactly where its ranges come and go) and the
//! last answer ([`Fast::kept`]), valid while the census's unordered ranges
//! stay as they were: a generation counted up under the lock at each change
//! of them ([`Fast::changed`]). A query whose generation is the kept
//! answer's does not take the census lock at all.
//!
//! Plain std, outside the Windows-only census, so that the tests run on any
//! host.

#![cfg_attr(not(windows), allow(dead_code))]

use std::collections::BTreeMap;
use std::sync::atomic::{
    AtomicU64,
    Ordering::{Acquire, Relaxed, Release},
};
use std::sync::{Arc, Mutex};

/// A range key as the census has them: copy, start, end, unordered access.
pub(crate) type Range = (usize, u64, u64, bool);

/// A range unordered views write: the copy, the title address, the bytes
/// (what `bindless::widened_writes` gives).
pub(crate) type Write<S> = (Arc<S>, usize, usize);

/// The census's unordered ranges alone, with their copies, kept under the
/// census lock where its range counts change (`bindless::Inner::count`).
pub(crate) struct Unordered<S> {
    ranges: BTreeMap<Range, Arc<S>>,
}

impl<S> Unordered<S> {
    pub(crate) fn new() -> Self {
        Unordered { ranges: BTreeMap::new() }
    }

    /// Unordered range `key` of `copy` is viewed now (it was not).
    pub(crate) fn came(&mut self, key: Range, copy: &Arc<S>) {
        self.ranges.insert(key, copy.clone());
    }

    /// Unordered range `key` is viewed by no slot any more.
    pub(crate) fn went(&mut self, key: &Range) {
        self.ranges.remove(key);
    }

    /// What `bindless::widened_writes` finds by walking every range: the
    /// unordered ones that are not empty, as copy, start and bytes. In key
    /// order, the order the census's B-tree form (`VICTORY_QUICK_SYNC=0`)
    /// gives too; its hash form gives the same set in no set order.
    pub(crate) fn writes(&self) -> Vec<Write<S>> {
        self.ranges
            .iter()
            .filter(|((_, address, end, _), _)| address < end)
            .map(|(&(_, address, end, _), copy)| (copy.clone(), address as usize, (end - address) as usize))
            .collect()
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.ranges.len()
    }
}

/// The parts of a census outside its lock.
pub(crate) struct Fast<S> {
    /// Counted up, under the census lock, at every change of the census's
    /// unordered ranges ([`Unordered`]).
    generation: AtomicU64,
    /// The last answer of [`Fast::answer`] and the generation it is of;
    /// none since the last change.
    kept: Mutex<Option<(u64, Arc<[Write<S>]>)>>,
}

impl<S> Default for Fast<S> {
    fn default() -> Self {
        Fast { generation: AtomicU64::new(0), kept: Mutex::new(None) }
    }
}

impl<S> Fast<S> {
    /// Under the census lock: its unordered ranges changed. Every change of
    /// them must be told before the lock is let go, or [`Fast::kept`] goes
    /// on answering what they were. The kept answer goes with it, so that
    /// it holds no copy the census let go of (a copy no slot views any more
    /// is given back when its last holder goes, `crate::shadow`).
    pub(crate) fn changed(&self) {
        self.generation.fetch_add(1, Release);
        *self.kept.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// The last answer, if the census's unordered ranges are still what it
    /// was made from. Takes no census lock: the generation is read once,
    /// and the answer kept for it is what the census held at it (a change
    /// told after the read is one this query came before).
    pub(crate) fn kept(&self) -> Option<Arc<[Write<S>]>> {
        let now = self.generation.load(Acquire);
        match &*self.kept.lock().unwrap_or_else(|e| e.into_inner()) {
            Some((of, writes)) if *of == now => Some(writes.clone()),
            _ => None,
        }
    }

    /// Under the census lock, whose unordered ranges are `unordered`: the
    /// answer, kept for the queries after. (All changes are told under the
    /// lock, so the generation read here is exact, and a kept answer is of
    /// it or older.)
    pub(crate) fn answer(&self, unordered: &Unordered<S>) -> Arc<[Write<S>]> {
        let now = self.generation.load(Relaxed);
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((of, writes)) = &*kept
            && *of == now
        {
            return writes.clone();
        }
        let writes: Arc<[Write<S>]> = unordered.writes().into();
        *kept = Some((now, writes.clone()));
        writes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A host copy, as far as the census cares.
    #[derive(Debug)]
    struct Copy(#[allow(dead_code)] u32);

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// A census as `bindless::Inner` counts it, with the fast parts: the
    /// slots, how many slots view each range (and its copy), the unordered
    /// ranges apart.
    struct Census {
        slots: HashMap<u32, (Range, Arc<Copy>)>,
        counts: HashMap<Range, (u32, Arc<Copy>)>,
        unordered: Unordered<Copy>,
        /// Whether a change of the unordered ranges is told (false: the
        /// negative control of the tests below).
        tell: bool,
    }

    impl Census {
        fn new() -> Self {
            Census { slots: HashMap::new(), counts: HashMap::new(), unordered: Unordered::new(), tell: true }
        }

        /// `bindless::Inner::count`: whether the unordered ranges changed.
        fn count(&mut self, key: Range, copy: &Arc<Copy>, add: bool) -> bool {
            let e = self.counts.entry(key).or_insert_with(|| (0, copy.clone()));
            let was = e.0;
            e.0 = if add { e.0 + 1 } else { e.0.saturating_sub(1) };
            let now = e.0;
            if now == 0 {
                self.counts.remove(&key);
            }
            if (was == 0) != (now == 0) && key.3 {
                if now > 0 {
                    self.unordered.came(key, copy);
                } else {
                    self.unordered.went(&key);
                }
                return true;
            }
            false
        }

        /// `bindless::Viewed::set_seen` with the switch.
        fn set(&mut self, fast: &Fast<Copy>, index: u32, seen: Option<(Range, Arc<Copy>)>) {
            let mut changed = false;
            if let Some((key, copy)) = self.slots.remove(&index) {
                changed |= self.count(key, &copy, false);
            }
            if let Some((key, copy)) = seen {
                changed |= self.count(key, &copy, true);
                self.slots.insert(index, (key, copy));
            }
            if changed && self.tell {
                fast.changed();
            }
        }

        /// The old walk of every range (`bindless::widened_writes`), sorted.
        fn walked(&self) -> Vec<(usize, usize, usize)> {
            let mut w: Vec<_> = self
                .counts
                .iter()
                .filter(|((_, address, end, uav), _)| *uav && address < end)
                .map(|((_, address, end, _), (_, copy))| (Arc::as_ptr(copy) as usize, *address as usize, (end - address) as usize))
                .collect();
            w.sort_unstable();
            w
        }
    }

    /// An answer as (copy, start, bytes), sorted.
    fn plain(writes: &[Write<Copy>]) -> Vec<(usize, usize, usize)> {
        let mut w: Vec<_> = writes.iter().map(|(c, a, l)| (Arc::as_ptr(c) as usize, *a, *l)).collect();
        w.sort_unstable();
        w
    }

    /// A query as `bindless::unordered_writes` makes it: the kept answer, or
    /// one made under the lock. Whether it was kept.
    fn query(census: &Mutex<Census>, fast: &Fast<Copy>) -> (Arc<[Write<Copy>]>, bool) {
        if let Some(kept) = fast.kept() {
            return (kept, true);
        }
        let census = census.lock().unwrap();
        (fast.answer(&census.unordered), false)
    }

    /// Slots rewritten at random (the same range again, another range, an
    /// unordered one, an empty one, none): after every change a query
    /// answers what the old walk finds, kept or made anew; a change that
    /// leaves the unordered ranges as they were keeps the answer.
    fn random_rewrites(tell: bool) -> Result<(usize, usize), String> {
        let copies: Vec<Arc<Copy>> = (0..5).map(|i| Arc::new(Copy(i))).collect();
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let census = Mutex::new(Census::new());
        census.lock().unwrap().tell = tell;
        let fast = Fast::default();
        let (mut kept, mut made) = (0, 0);
        for step in 0..60_000u64 {
            let slot = (rng.next() % 300) as u32;
            let seen = (step % 9 != 0).then(|| {
                let c = (rng.next() % 5) as usize;
                let start = 0x1000 + (rng.next() % 40) * 16;
                // Now and then an empty range (end at the start).
                let len = if rng.next() % 50 == 0 { 0 } else { 16 + rng.next() % 64 };
                let key = (Arc::as_ptr(&copies[c]) as usize, start, start + len, rng.next() % 3 == 0);
                (key, copies[c].clone())
            });
            census.lock().unwrap().set(&fast, slot, seen);
            // Queries between some changes only, two in a row now and then.
            for _ in 0..(step % 3) {
                let (answer, was_kept) = query(&census, &fast);
                let walked = census.lock().unwrap().walked();
                if plain(&answer) != walked {
                    return Err(format!("step {step}: answered {} ranges, the walk finds {}", answer.len(), walked.len()));
                }
                if was_kept {
                    kept += 1;
                } else {
                    made += 1;
                }
            }
        }
        Ok((kept, made))
    }

    #[test]
    fn answers_follow_every_change_of_the_unordered_ranges() {
        let (kept, made) = random_rewrites(true).unwrap();
        assert!(kept > 1_000 && made > 1_000, "both kept and new answers are tested: {kept} kept, {made} made");
    }

    /// The negative control: if a change of the unordered ranges is not
    /// told, the kept answer goes stale and the test above sees it.
    #[test]
    fn an_untold_change_gives_a_stale_answer() {
        assert!(random_rewrites(false).is_err());
    }

    /// The cases one by one: a range that comes, one that goes, a slot made
    /// again from the same range, an empty range, a change of ranges that
    /// are not unordered, and two heaps, each with its own answer.
    #[test]
    fn each_kind_of_change_is_seen() {
        let a = Arc::new(Copy(0));
        let pa = Arc::as_ptr(&a) as usize;
        let census = Mutex::new(Census::new());
        let fast = Fast::default();
        let set = |slot: u32, seen: Option<(u64, u64, bool)>| {
            census.lock().unwrap().set(&fast, slot, seen.map(|(s, e, u)| ((pa, s, e, u), a.clone())));
        };
        let answer = || plain(&query(&census, &fast).0);
        assert_eq!(answer(), []);
        set(0, Some((0x100, 0x200, true)));
        assert_eq!(answer(), [(pa, 0x100, 0x100)], "a range came");
        let first = query(&census, &fast);
        assert!(first.1, "and the answer is kept");
        set(1, Some((0x400, 0x480, false)));
        let second = query(&census, &fast);
        assert!(second.1 && Arc::ptr_eq(&first.0, &second.0), "a range that is not unordered keeps the answer");
        set(2, Some((0x800, 0x900, true)));
        assert_eq!(answer(), [(pa, 0x100, 0x100), (pa, 0x800, 0x100)], "a second range came");
        set(0, Some((0x100, 0x200, true)));
        assert_eq!(answer(), [(pa, 0x100, 0x100), (pa, 0x800, 0x100)], "slot 0 made again from its range");
        set(3, Some((0x100, 0x200, true)));
        set(0, None);
        assert_eq!(answer(), [(pa, 0x100, 0x100), (pa, 0x800, 0x100)], "slot 3 still views it");
        set(3, Some((0x100, 0x100, true)));
        assert_eq!(answer(), [(pa, 0x800, 0x100)], "the range went; an empty one is none");
        assert_eq!(census.lock().unwrap().unordered.len(), 2, "the empty range is kept, only not answered");
        set(2, Some((0x800, 0x900, false)));
        assert_eq!(answer(), [], "a slot from unordered to shader view");
        // Two heaps: each answers for itself.
        let other = Mutex::new(Census::new());
        let other_fast = Fast::default();
        other.lock().unwrap().set(&other_fast, 0, Some(((pa, 0x10, 0x20, true), a.clone())));
        assert_eq!(plain(&query(&other, &other_fast).0), [(pa, 0x10, 0x10)]);
        assert_eq!(answer(), [], "the first heap's answer is its own");
        set(5, Some((0x30, 0x40, true)));
        assert_eq!(plain(&query(&other, &other_fast).0), [(pa, 0x10, 0x10)], "and the other's its own");
        assert_eq!(answer(), [(pa, 0x30, 0x10)]);
    }

    /// A kept answer holds the copies of the ranges it names; once the
    /// census lets go of a range, the answer lets go of its copy too.
    #[test]
    fn a_change_lets_go_of_the_kept_answer() {
        let (a, b) = (Arc::new(Copy(0)), Arc::new(Copy(1)));
        let (pa, pb) = (Arc::as_ptr(&a) as usize, Arc::as_ptr(&b) as usize);
        let census = Mutex::new(Census::new());
        let fast = Fast::default();
        let set = |slot: u32, seen: Option<(Range, Arc<Copy>)>| census.lock().unwrap().set(&fast, slot, seen);
        set(0, Some(((pa, 0x100, 0x200, true), a.clone())));
        set(1, Some(((pb, 0x100, 0x200, true), b.clone())));
        assert_eq!(query(&census, &fast).0.len(), 2);
        assert_eq!(Arc::strong_count(&b), 5, "the test's, the slot's, the range count's, the unordered ranges' and the answer's");
        set(1, None);
        assert_eq!(Arc::strong_count(&b), 1, "only the test's: the census and its kept answer let go");
        assert_eq!(plain(&query(&census, &fast).0), [(pa, 0x100, 0x100)]);
        set(0, Some(((pa, 0x100, 0x200, false), a.clone())));
        assert_eq!(Arc::strong_count(&a), 3, "the test's, the slot's, the range count's");
    }

    /// While one thread enters ranges, others query: every answer is what
    /// the census held at some moment (here: the first n ranges entered, for
    /// an n that never falls between one thread's queries), and once the
    /// thread is done the answer is all of them.
    #[test]
    fn queries_beside_changes_see_a_census_that_was() {
        let copy = Arc::new(Copy(0));
        let pc = Arc::as_ptr(&copy) as usize;
        let census = Mutex::new(Census::new());
        let fast = Fast::default();
        const N: u64 = 3_000;
        std::thread::scope(|s| {
            s.spawn(|| {
                for k in 0..N {
                    census.lock().unwrap().set(&fast, k as u32, Some(((pc, 16 * k, 16 * k + 8, true), copy.clone())));
                }
            });
            for _ in 0..3 {
                s.spawn(|| {
                    let mut seen = 0;
                    for _ in 0..20_000 {
                        let answer = plain(&query(&census, &fast).0);
                        let n = answer.len();
                        let expected: Vec<_> = (0..n as u64).map(|k| (pc, 16 * k as usize, 8)).collect();
                        assert_eq!(answer, expected, "the first {n} ranges");
                        assert!(n >= seen, "an answer older than the one before: {n} after {seen}");
                        seen = n;
                    }
                });
            }
        });
        assert_eq!(query(&census, &fast).0.len(), N as usize);
    }
}
