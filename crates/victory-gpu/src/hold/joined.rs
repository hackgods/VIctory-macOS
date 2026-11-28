//! mac: from VIctory-macOS #419 e678253e (gdemontalivet), "coalesce
//! cross-queue and bottom-of-pipe dependent held pieces"
//! (`VICTORY_COALESCE_PIECES=1`, off by default here; on upstream).
//!
//! Without it, each queue's ready heads go to the host GPU as a piece of
//! their own, and a head whose wait reads what another queue's piece
//! writes at its end runs only once that piece has run and been finished:
//! one host submission per queue switch, and with the title's submissions
//! left to the host GPU (`crate::host::unwaited`) a drain before each held
//! head. Here the ready heads of every queue gather into one piece, which
//! is given when nothing more can join it:
//!
//! - a head whose wait the piece's own bottom-of-pipe writes meet (the last
//!   write of the piece that reaches the wait's bytes decides) joins it,
//!   behind a barrier;
//! - a Signal behind work of the piece is delivered once the piece has run,
//!   and a fence Wait it meets lets its queue join, behind a barrier;
//! - a head that would join another queue's work, or join through the
//!   piece's writes or Signals, joins only where the submission can keep
//!   the order of what it does on the CPU (`Joining::can_join`): else the
//!   piece is given first. Heads of one queue that follow each other stay
//!   one piece, as without the switch.
//!
//! Changed from upstream: a wait is first judged by the last write of the
//! piece that reaches it, and by memory only where none does, so a later
//! write of another value, or a write that clears what memory holds, keeps
//! the head waiting for the piece to run (the two cases evan-kolberg found
//! in the pull request). A wait memory does not meet is put to `World::owed`
//! as before, so a submission left to the host GPU that owes it is finished.

// Its callers are Windows only (`crate::queue`); the tests run everywhere.
#![cfg_attr(not(windows), allow(dead_code))]

use std::sync::OnceLock;

use super::{Held, Item, Pumped, ValueWait, World};

/// Whether the pump joins pieces across queues (`VICTORY_COALESCE_PIECES=1`;
/// off unless set, read once).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = rule(std::env::var("VICTORY_COALESCE_PIECES").ok().as_deref());
        if on {
            tracing::info!(
                cap = cap(),
                "VICTORY_COALESCE_PIECES=1: ready heads of every queue join one host submission, and a wait the \
                 piece's own bottom-of-pipe writes or Signals meet joins it behind a barrier (VIctory-macOS #419)"
            );
        }
        on
    })
}

/// `VICTORY_COALESCE_PIECES`'s rule: on only for `1`.
fn rule(value: Option<&str>) -> bool {
    value.map(str::trim) == Some("1")
}

/// The most host lists a piece is joined to across queues
/// (`VICTORY_JOIN_LISTS`, default 64 as upstream; one queue's heads in a
/// row are never split for it).
pub(crate) fn cap() -> usize {
    static CAP: OnceLock<usize> = OnceLock::new();
    *CAP.get_or_init(|| cap_rule(std::env::var("VICTORY_JOIN_LISTS").ok().as_deref()))
}

fn cap_rule(value: Option<&str>) -> usize {
    value.and_then(|v| v.trim().parse::<usize>().ok()).filter(|&n| n >= 1).unwrap_or(64)
}

/// What a segment does on the CPU when its submission is prepared and
/// finished, as far as joining it to work of another queue goes
/// (upstream's `BatchFlags`, with our steps).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct JoinFlags {
    /// Copies into title memory while prepared (`Step::Copy`, `Move`).
    pub(crate) title_copy: bool,
    /// Reads of title memory while prepared (`Copy`, `Move`, `Fill`, an
    /// `Own` step with a range).
    pub(crate) cpu_read: bool,
    /// Shader writes declared into host copies, copied back once run
    /// (`Step::GpuWrite`).
    pub(crate) gpu_write: bool,
    /// Work for after it has run (`defer_after`).
    pub(crate) after: bool,
    /// mac: title memory its work for after writes from what the GPU made
    /// (textures read back into buffers, textures copied back): a later
    /// list's refresh, made while prepared, would not see it.
    pub(crate) lands: bool,
    /// Uploads made before every list of the submission: textures read
    /// through tables (`Step::Texture`) and, mac, view heaps' textures
    /// (`Step::Heap`).
    pub(crate) uploads: bool,
}

impl JoinFlags {
    pub(crate) fn merge(self, o: Self) -> Self {
        Self {
            title_copy: self.title_copy || o.title_copy,
            cpu_read: self.cpu_read || o.cpu_read,
            gpu_write: self.gpu_write || o.gpu_write,
            after: self.after || o.after,
            lands: self.lands || o.lands,
            uploads: self.uploads || o.uploads,
        }
    }
}

/// Whether a segment (`next`) may join a piece whose segments together do
/// `piece`, keeping the order their submissions would have had: the
/// uploads are made before any list's copies, and every list's CPU work
/// before the GPU runs any, the copy-back and the work for after once all
/// have run (upstream's `can_join_batch`, plus `lands`).
pub(crate) fn may_join(piece: JoinFlags, next: JoinFlags) -> bool {
    !((piece.title_copy || piece.after) && next.uploads)
        && !((piece.gpu_write || piece.after) && next.cpu_read)
        && !piece.lands
}

/// What a head's waits say, against the piece gathered so far and memory.
enum Verdict {
    /// All hold: `via` of them by the piece's writes, `unknown` with a
    /// comparison we do not know (taken as holding).
    Ready { via: u32, unknown: u32 },
    /// The piece's last write that reaches one of them leaves a value it
    /// does not take (or one not known now): the piece runs first.
    Piece,
    /// One of them reads memory no write of the piece reaches, and memory
    /// does not meet it.
    Memory,
}

fn verdict<S>(
    waits: &[ValueWait],
    piece: &[(S, bool)],
    leaves: &dyn Fn(&S, &ValueWait) -> Option<Option<bool>>,
    read: &dyn Fn(u64, u8) -> Option<u64>,
) -> Verdict {
    let (mut via, mut unknown, mut memory) = (0, 0, false);
    for wait in waits {
        match piece.iter().rev().find_map(|(s, _)| leaves(s, wait)) {
            Some(Some(true)) => via += 1,
            Some(_) => return Verdict::Piece,
            None => match wait.holds(read) {
                Some(true) => {}
                Some(false) => memory = true,
                None => unknown += 1,
            },
        }
    }
    if memory { Verdict::Memory } else { Verdict::Ready { via, unknown } }
}

/// What the pump does with a queue's head.
enum Decision {
    Blocked,
    /// The head may run, but not in the piece: the piece goes first.
    GiveFirst,
    /// The segment joins, with a barrier before it or not; `via` of its
    /// waits met by the piece.
    Segment { barrier: bool, via: u32 },
    Signal,
    /// The fence Wait holds, by a Signal behind the piece or not.
    FenceWait { by_piece: bool },
}

/// The piece being gathered.
struct Piece<S> {
    segments: Vec<(S, bool)>,
    /// Signals behind it, delivered once it has run.
    signals: Vec<(usize, u64)>,
    /// Its queue, `None` once it holds several (meaningful when not empty).
    of: Option<usize>,
    /// The queue of its last segment.
    last: Option<usize>,
}

impl<S> Piece<S> {
    /// Gives it to the host GPU, then its Signals.
    fn give(&mut self, w: &mut World<S>, done: &mut Pumped) {
        if !self.segments.is_empty() {
            let run = &mut w.joining.as_mut().expect("joining").run;
            run(self.of, std::mem::take(&mut self.segments));
            done.pieces += 1;
        }
        self.last = None;
        for (fence, value) in self.signals.drain(..) {
            (w.signal)(fence, value);
        }
    }
}

impl<S> Held<S> {
    /// [`Held::pump`] with pieces joined across queues.
    pub(super) fn pump_joined(&mut self, w: &mut World<S>) -> Pumped {
        let mut done = Pumped::default();
        let mut piece = Piece { segments: Vec::new(), signals: Vec::new(), of: None, last: None };
        loop {
            let mut moved = false;
            for (queue, items) in &mut self.queues {
                let queue = *queue;
                // The queue's last item taken was a fence Wait a Signal
                // behind the piece met: its next segment joins behind a
                // barrier.
                let mut fenced = false;
                loop {
                    let mut behind = items.iter_mut();
                    let decision = match behind.next() {
                        None => Decision::Blocked,
                        Some((Item::Segment(head, waits), since, fresh)) => {
                            let joining = w.joining.as_ref().expect("joining");
                            let (leaves, can_join) = (joining.leaves, joining.can_join);
                            let mut v = verdict(waits, &piece.segments, leaves, w.read);
                            // What a piece left to the host GPU owes may be
                            // what they wait for: finished first, as
                            // without the switch, and looked at again.
                            if matches!(v, Verdict::Memory) {
                                let next: Vec<&S> = if w.ahead {
                                    let after = behind.map_while(|(item, ..)| match item {
                                        Item::Segment(s, none) if none.is_empty() => Some(&*s),
                                        _ => None,
                                    });
                                    piece.segments.iter().map(|(s, _)| s).chain([&*head]).chain(after).collect()
                                } else {
                                    Vec::new()
                                };
                                if (w.owed)(waits.as_slice(), &next) {
                                    v = verdict(waits, &piece.segments, leaves, w.read);
                                }
                            }
                            match v {
                                // Not found waiting: the piece runs in this pump.
                                Verdict::Piece => Decision::Blocked,
                                Verdict::Memory => {
                                    *fresh = false;
                                    Decision::Blocked
                                }
                                Verdict::Ready { via, unknown } => {
                                    let crosses = !piece.segments.is_empty()
                                        && (piece.last != Some(queue) || via > 0 || fenced);
                                    if crosses && !can_join(&piece.segments, head) {
                                        Decision::GiveFirst
                                    } else {
                                        done.unknown += unknown;
                                        if !*fresh {
                                            done.waited.push((queue, since.elapsed()));
                                        }
                                        Decision::Segment { barrier: via > 0 || fenced, via }
                                    }
                                }
                            }
                        }
                        Some((Item::Signal(..), ..)) => Decision::Signal,
                        Some((Item::FenceWait(fence, value), _, fresh)) => {
                            if (w.fence)(*fence) >= *value {
                                Decision::FenceWait { by_piece: false }
                            } else if piece.signals.iter().any(|&(f, v)| f == *fence && v >= *value) {
                                Decision::FenceWait { by_piece: true }
                            } else {
                                *fresh = false;
                                Decision::Blocked
                            }
                        }
                    };
                    match decision {
                        Decision::Blocked => break,
                        Decision::GiveFirst => {
                            done.splits += 1;
                            piece.give(w, &mut done);
                            fenced = false;
                            moved = true;
                        }
                        Decision::Segment { barrier, via } => {
                            moved = true;
                            if let Some((Item::Segment(s, _), ..)) = items.pop_front() {
                                if piece.segments.is_empty() {
                                    piece.of = Some(queue);
                                } else {
                                    if piece.last != Some(queue) {
                                        done.crossed += 1;
                                    }
                                    if piece.of != Some(queue) {
                                        piece.of = None;
                                    }
                                }
                                piece.segments.push((s, barrier));
                                piece.last = Some(queue);
                                done.segments += 1;
                                done.met_by_piece += via;
                            }
                            fenced = false;
                        }
                        Decision::Signal => {
                            moved = true;
                            if let Some((Item::Signal(fence, value), ..)) = items.pop_front() {
                                if piece.segments.is_empty() {
                                    (w.signal)(fence, value);
                                } else {
                                    piece.signals.push((fence, value));
                                }
                            }
                        }
                        Decision::FenceWait { by_piece } => {
                            moved = true;
                            items.pop_front();
                            if by_piece {
                                done.fences_by_piece += 1;
                                fenced = true;
                            }
                        }
                    }
                }
            }
            if !moved {
                if piece.segments.is_empty() && piece.signals.is_empty() {
                    break;
                }
                piece.give(w, &mut done);
            }
        }
        self.queues.retain(|(_, items)| !items.is_empty());
        done
    }
}

#[cfg(test)]
mod tests {
    use super::super::{GREATER_EQUAL, Joining};
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;

    /// Memory, fences and a log of what ran, for a recording executor.
    #[derive(Default)]
    struct Fake {
        memory: HashMap<u64, u64>,
        fences: HashMap<usize, u64>,
        log: Vec<String>,
        /// Bottom-of-pipe writes of each segment, in order: (address, value).
        writes: HashMap<&'static str, Vec<(u64, u64)>>,
        /// Segments that may not join a piece across a queue or a wait.
        alone: Vec<&'static str>,
        /// Pieces are not waited for: their writes are owed until finished.
        unwaited: bool,
        owed: Vec<(u64, u64)>,
    }

    #[test]
    fn joining_is_on_only_when_asked_for() {
        assert!(rule(Some("1")));
        assert!(rule(Some(" 1 ")));
        for off in [None, Some(""), Some("0"), Some("2"), Some("on")] {
            assert!(!rule(off), "{off:?}");
        }
        assert_eq!(cap_rule(None), 64);
        assert_eq!(cap_rule(Some("8")), 8);
        assert_eq!(cap_rule(Some("0")), 64);
        assert_eq!(cap_rule(Some("x")), 64);
    }

    #[test]
    fn a_join_keeps_the_order_of_the_cpu_work() {
        let clean = JoinFlags::default();
        let gpu_write = JoinFlags { gpu_write: true, ..clean };
        let title_copy = JoinFlags { title_copy: true, cpu_read: true, ..clean };
        let cpu_read = JoinFlags { cpu_read: true, ..clean };
        let after = JoinFlags { after: true, ..clean };
        let lands = JoinFlags { lands: true, after: true, ..clean };
        let uploads = JoinFlags { uploads: true, ..clean };
        assert!(may_join(clean, clean));
        assert!(may_join(gpu_write, uploads), "a shader write, then uploads");
        assert!(may_join(gpu_write, gpu_write));
        assert!(may_join(title_copy, cpu_read), "copies keep their order in the submission");
        assert!(may_join(cpu_read, title_copy));
        assert!(!may_join(gpu_write, cpu_read), "a read of what goes back after the run");
        assert!(!may_join(after, cpu_read));
        assert!(!may_join(title_copy, uploads), "uploads are made before the copies");
        assert!(!may_join(after, uploads));
        assert!(!may_join(lands, clean), "a refresh would not see what lands");
        assert!(may_join(clean.merge(gpu_write), clean));
        assert_eq!(clean.merge(title_copy).merge(after), JoinFlags { title_copy: true, cpu_read: true, after: true, ..clean });
    }

    fn ge(address: u64, reference: u64) -> ValueWait {
        ValueWait { address, bytes: 8, comparison: GREATER_EQUAL, reference, mask: u64::MAX }
    }

    fn pump(held: &mut Held<&'static str>, fake: &RefCell<Fake>, cap: usize) -> Pumped {
        let read = |a: u64, _| Some(*fake.borrow().memory.get(&a).unwrap_or(&0));
        let fence = |f: usize| *fake.borrow().fences.get(&f).unwrap_or(&0);
        let leaves = |s: &&'static str, wait: &ValueWait| {
            let f = fake.borrow();
            let &(at, value) = f.writes.get(s)?.iter().rev().find(|&&(at, _)| wait.reads(at, 8))?;
            Some(if at == wait.address { wait.holds_for(value) } else { None })
        };
        let can_join = |piece: &[(&'static str, bool)], next: &&'static str| {
            piece.len() < cap && !fake.borrow().alone.contains(next)
        };
        let mut run_one = |_: usize, _: Vec<&'static str>| unreachable!("pieces are joined");
        let mut run = |of: Option<usize>, segments: Vec<(&'static str, bool)>| {
            let mut f = fake.borrow_mut();
            let mut names = Vec::new();
            for (s, barrier) in &segments {
                for (a, v) in f.writes.get(s).cloned().unwrap_or_default() {
                    if f.unwaited {
                        f.owed.push((a, v));
                    } else {
                        f.memory.insert(a, v);
                    }
                }
                names.push(if *barrier { format!("|{s}") } else { (*s).to_owned() });
            }
            let of = of.map_or("q*".to_owned(), |q| format!("q{q}"));
            f.log.push(format!("{of}: {}", names.join(" ")));
        };
        let mut owed = |_: &[ValueWait], _: &[&&'static str]| {
            let mut f = fake.borrow_mut();
            let owed = std::mem::take(&mut f.owed);
            for &(a, v) in &owed {
                f.memory.insert(a, v);
            }
            if !owed.is_empty() {
                f.log.push("finished".into());
            }
            !owed.is_empty()
        };
        let mut signal = |fence: usize, value: u64| {
            let mut f = fake.borrow_mut();
            f.fences.insert(fence, value);
            f.log.push(format!("signal {fence} {value}"));
        };
        held.pump(&mut World {
            read: &read,
            fence: &fence,
            run: &mut run_one,
            signal: &mut signal,
            owed: &mut owed,
            ahead: false,
            joining: Some(Joining { leaves: &leaves, can_join: &can_join, run: &mut run }),
        })
    }

    /// The chain of `hold`'s `chained_waits_across_two_queues` (five pieces
    /// without the switch) is one piece, a barrier before each segment
    /// whose wait the piece's writes meet.
    #[test]
    fn chained_waits_across_two_queues_are_one_piece() {
        let fake = RefCell::new(Fake::default());
        {
            let mut f = fake.borrow_mut();
            f.writes.insert("g0", vec![(0x08, 1)]);
            f.writes.insert("c1", vec![(0x10, 1)]);
            f.writes.insert("g1", vec![(0x20, 1)]);
            f.writes.insert("c2", vec![(0x30, 1)]);
        }
        let mut held = Held::default();
        held.push(2, Item::Segment("c1", vec![ge(0x08, 1)]));
        held.push(2, Item::Segment("c2", vec![ge(0x20, 1)]));
        assert_eq!(pump(&mut held, &fake, 64).pieces, 0);
        held.push(1, Item::Segment("g0", vec![]));
        held.push(1, Item::Segment("g1", vec![ge(0x10, 1)]));
        held.push(1, Item::Segment("g2", vec![ge(0x30, 1)]));
        let p = pump(&mut held, &fake, 64);
        assert_eq!(fake.borrow().log, ["q*: g0 |c1 |g1 |c2 |g2"]);
        assert_eq!((p.pieces, p.segments, p.met_by_piece, p.crossed, p.splits), (1, 5, 4, 4, 0));
        assert_eq!(p.waited.len(), 4, "c1, g1, c2, g2 were found waiting first");
        assert!(held.is_empty());
    }

    /// Ready heads of two queues that wait for nothing are one piece, with
    /// no barrier; one queue's alone keeps its queue.
    #[test]
    fn ready_heads_of_several_queues_join_without_a_barrier() {
        let fake = RefCell::new(Fake::default());
        let mut held = Held::default();
        held.push(1, Item::Segment("a", vec![]));
        held.push(1, Item::Segment("b", vec![]));
        held.push(2, Item::Segment("c", vec![]));
        let p = pump(&mut held, &fake, 64);
        assert_eq!(fake.borrow().log, ["q*: a b c"]);
        assert_eq!((p.pieces, p.crossed), (1, 1));
        held.push(3, Item::Segment("d", vec![]));
        pump(&mut held, &fake, 64);
        assert_eq!(fake.borrow().log[1], "q3: d");
    }

    /// The piece's last write to the word decides: a 7 then a 0 leave 0,
    /// so the head waiting for 7 or more waits for the piece to run (and
    /// then for memory), as without the switch.
    #[test]
    fn the_last_write_of_the_piece_decides() {
        let fake = RefCell::new(Fake::default());
        {
            let mut f = fake.borrow_mut();
            f.writes.insert("p1", vec![(0x100, 7)]);
            f.writes.insert("p2", vec![(0x100, 0)]);
        }
        let mut held = Held::default();
        held.push(2, Item::Segment("consumer", vec![ge(0x100, 7)]));
        pump(&mut held, &fake, 64);
        held.push(1, Item::Segment("p1", vec![]));
        held.push(1, Item::Segment("p2", vec![]));
        let p = pump(&mut held, &fake, 64);
        assert_eq!(fake.borrow().log, ["q1: p1 p2"]);
        assert_eq!(p.met_by_piece, 0);
        assert!(held.holds_work(2), "memory reads 0 after the piece");
        // A later write of 7 lets it go, in that one's piece.
        fake.borrow_mut().writes.insert("p3", vec![(0x100, 7)]);
        held.push(1, Item::Segment("p3", vec![]));
        pump(&mut held, &fake, 64);
        assert_eq!(fake.borrow().log, ["q1: p1 p2", "q*: p3 |consumer"]);
    }

    /// Memory holds 7, but the piece clears it: the head waits for the
    /// piece, and stays held after it.
    #[test]
    fn a_piece_that_clears_the_word_holds_the_head() {
        let fake = RefCell::new(Fake::default());
        fake.borrow_mut().memory.insert(0x100, 7);
        fake.borrow_mut().writes.insert("clear", vec![(0x100, 0)]);
        let mut held = Held::default();
        held.push(1, Item::Segment("clear", vec![]));
        held.push(2, Item::Segment("consumer", vec![ge(0x100, 7)]));
        pump(&mut held, &fake, 64);
        assert_eq!(fake.borrow().log, ["q1: clear"]);
        assert!(held.holds_work(2));
    }

    /// A write of the piece over part of the word, or of a value not known
    /// now, gives the piece first: memory decides after it.
    #[test]
    fn a_write_the_piece_cannot_judge_gives_it_first() {
        let fake = RefCell::new(Fake::default());
        fake.borrow_mut().writes.insert("part", vec![(0x104, 9)]);
        let mut held = Held::default();
        held.push(1, Item::Segment("part", vec![]));
        held.push(2, Item::Segment("consumer", vec![ge(0x100, 1)]));
        fake.borrow_mut().memory.insert(0x100, 1);
        let p = pump(&mut held, &fake, 64);
        assert_eq!(fake.borrow().log, ["q1: part", "q2: consumer"]);
        assert_eq!((p.pieces, p.met_by_piece), (2, 0));
    }

    /// A head that may not join (an order the submission cannot keep) has
    /// the piece and its Signals given first.
    #[test]
    fn a_head_that_may_not_join_gives_the_piece_and_its_signals_first() {
        let fake = RefCell::new(Fake::default());
        fake.borrow_mut().writes.insert("producer", vec![(0x100, 1)]);
        fake.borrow_mut().alone.push("reader");
        let mut held = Held::default();
        held.push(2, Item::Segment("reader", vec![ge(0x100, 1)]));
        pump(&mut held, &fake, 64);
        held.push(1, Item::Segment("producer", vec![]));
        held.push(1, Item::Signal(7, 42));
        let p = pump(&mut held, &fake, 64);
        assert_eq!(fake.borrow().log, ["q1: producer", "signal 7 42", "q2: reader"]);
        assert_eq!((p.pieces, p.splits, p.met_by_piece), (2, 1, 0));
    }

    /// The cap splits only across queues: one queue's heads stay together.
    #[test]
    fn the_cap_splits_across_queues_only() {
        let fake = RefCell::new(Fake::default());
        let mut held = Held::default();
        for s in ["a", "b", "c"] {
            held.push(1, Item::Segment(s, vec![]));
        }
        held.push(2, Item::Segment("d", vec![]));
        held.push(2, Item::Segment("e", vec![]));
        let p = pump(&mut held, &fake, 2);
        assert_eq!(fake.borrow().log, ["q1: a b c", "q2: d e"]);
        assert_eq!((p.pieces, p.splits), (2, 1));
    }

    /// A Signal behind the piece is delivered once it has run; another
    /// queue's fence Wait it meets lets that queue join behind a barrier.
    #[test]
    fn a_signal_behind_the_piece_meets_a_fence_wait() {
        let fake = RefCell::new(Fake::default());
        let mut held = Held::default();
        held.push(1, Item::Segment("work", vec![]));
        held.push(1, Item::Signal(5, 3));
        held.push(1, Item::Segment("more", vec![]));
        held.push(2, Item::FenceWait(5, 3));
        held.push(2, Item::Segment("consumer", vec![]));
        let p = pump(&mut held, &fake, 64);
        assert_eq!(fake.borrow().log, ["q*: work more |consumer", "signal 5 3"]);
        assert_eq!((p.pieces, p.fences_by_piece), (1, 1));
        // A Signal with nothing gathered goes at once.
        held.push(3, Item::Signal(6, 1));
        pump(&mut held, &fake, 64);
        assert_eq!(fake.borrow().log.last().map(String::as_str), Some("signal 6 1"));
    }

    /// A queue keeps its order: a ready segment does not pass a held one.
    #[test]
    fn a_queue_keeps_its_order() {
        let fake = RefCell::new(Fake::default());
        let mut held = Held::default();
        held.push(2, Item::Segment("first", vec![ge(0x100, 1)]));
        held.push(2, Item::Segment("second", vec![]));
        held.push(1, Item::Segment("other", vec![]));
        pump(&mut held, &fake, 64);
        assert_eq!(fake.borrow().log, ["q1: other"]);
        fake.borrow_mut().memory.insert(0x100, 1);
        pump(&mut held, &fake, 64);
        assert_eq!(fake.borrow().log, ["q1: other", "q2: first second"]);
    }

    /// A piece left to the host GPU still owes its write: a wait memory
    /// does not meet is put to `owed`, which finishes it, and the head runs
    /// in the same pump without having waited.
    #[test]
    fn a_wait_an_unwaited_piece_owes_finishes_it_first() {
        let fake = RefCell::new(Fake { unwaited: true, ..Fake::default() });
        fake.borrow_mut().writes.insert("draw", vec![(0x100, 7)]);
        let mut held = Held::default();
        held.push(1, Item::Segment("draw", vec![]));
        let p = pump(&mut held, &fake, 64);
        assert_eq!(p.pieces, 1);
        held.push(2, Item::Segment("depth", vec![ge(0x100, 7)]));
        let p = pump(&mut held, &fake, 64);
        assert_eq!(fake.borrow().log, ["q1: draw", "finished", "q2: depth"]);
        assert_eq!((p.pieces, p.waited.len()), (1, 0));
    }

    /// Joined in one pump, a wait the piece meets needs no finish of what
    /// it owes: the two run in one piece.
    #[test]
    fn a_wait_the_piece_meets_is_not_put_to_owed() {
        let fake = RefCell::new(Fake { unwaited: true, ..Fake::default() });
        fake.borrow_mut().writes.insert("draw", vec![(0x100, 7)]);
        let mut held = Held::default();
        held.push(1, Item::Segment("draw", vec![]));
        held.push(2, Item::Segment("depth", vec![ge(0x100, 7)]));
        let p = pump(&mut held, &fake, 64);
        assert_eq!(fake.borrow().log, ["q*: draw |depth"]);
        assert_eq!((p.pieces, p.met_by_piece, p.waited.len()), (1, 1, 0));
    }
}
