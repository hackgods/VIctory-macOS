//! A dispatch counted in threads (the console's `DispatchX` with its
//! thread-dimensions flag, 0x20; `emit::Options::thread_counts`): the lanes
//! past the count start with EXEC off.
//!
//! The console runs the groups that hold the threads asked for; in the last
//! group of a dimension the threads at or past the count are not in EXEC.
//! The title's ray-tracing structure builder dispatches its box unpackers
//! (`28918b7150a7aad5`, `caa78b8ca62bc3d6`) this way and counts their nodes
//! with `s_bcnt1_i32_b64 exec` at entry. Run as group counts they ran 64
//! times the threads asked for and wrote empty nodes past the structure
//! (LakshanSivakumar/Native-Victory-macOS 0ae6cfcd, measured there).
//!
//! Here the host runs whole groups, rounded up, and every host thread runs,
//! as the console's lanes exist with EXEC off: so the group's barriers and
//! the hand-overs between 32-lane halves ([`super::exchange`]) still meet
//! every thread. Only the structure pass sees a difference: EXEC on entry
//! is a lane mask of its own ([`crate::ir::Program::thread_counts`]), whose
//! bit this module gives each lane. Ballots, counts of EXEC and the guards
//! of stores follow from it as from any other mask. Threads leaving early
//! instead (VIctory-macOS #527, fxceless/victory-fork #646 where nothing
//! waits) would be cheaper and is not done: the builder's unpackers have
//! group barriers and hand-overs.

use super::*;

impl Lower<'_> {
    /// The lane's bit of EXEC on entry in a program for dispatches counted
    /// in threads: below the thread count in each dimension wider than one
    /// thread a group, which the host passes in user-data dwords past the
    /// console's own ([`super::super::USER_DISPATCH_THREADS`]). A count of 0
    /// puts every lane on (the host passes none: a dispatch of groups).
    pub(super) fn launched(&mut self) {
        let p = self.x.p;
        let (Some(&(_, exec)), Some(threads)) = (
            p.inputs.iter().find(|i| i.0 == Reg::S(EXEC)),
            self.s.threads,
        ) else {
            return;
        };
        // Past every user-data dword the console's program has.
        let layout = self.x.native.layout.iter();
        let ends = layout.map(|rp| u32::from(rp.user_data) + rp.sgprs);
        let user = match self.x.native.io {
            StageIo::Cs(cs) => u32::from(cs.user_sgprs()),
            _ => 0,
        };
        let free = ends.chain([user]).max().unwrap_or(0);
        let e = self.s.entry;
        let mut on = self.s.m.bool(true);
        let mut word = 0;
        // A dimension of one thread a group has no partial group.
        for d in (0..3u8).filter(|&d| threads[usize::from(d)] > 1) {
            let (Val::I32(t), Val::I32(g)) = (self.thread_id(d), self.compute_id("groupId.i32", 94, d)) else {
                unreachable!("compute ids are numbers")
            };
            let n = self.s.m.i32(threads[usize::from(d)]);
            let at = self.s.m.bin(e, Bin::Mul, g, n);
            let at = self.s.m.bin(e, Bin::Add, at, t);
            let count = self.user_at(UserData {
                index: free + word,
                slot: 0,
                root: RootType::Other(super::super::USER_DISPATCH_THREADS),
                dword: d.into(),
            });
            word += 1;
            let zero = self.s.m.i32(0);
            let all = self.s.m.cmp(e, pred::EQ, count, zero);
            let below = self.s.m.cmp(e, pred::IULT, at, count);
            let here = self.s.m.bin(e, Bin::Or, all, below);
            on = self.s.m.bin(e, Bin::And, on, here);
        }
        self.lane_bits.insert(exec, on);
    }
}
