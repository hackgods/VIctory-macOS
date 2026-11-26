//! An exact answer for a read across the two 32-lane halves of a console
//! wave64 ([`super::halves`]), through thread-group shared memory
//! ([`Mac::exchange`], `emit::Options::halves_exchange`;
//! `notes/ray-tracing-plan.md`).
//!
//! In halves a console wave is two host waves, and what one lane reads of a
//! lane in the other half has no host wave operation: such a read is
//! refused, or with half-local reads answered from the reader's own half,
//! which is not the console's value. Here the lanes hand the value over
//! instead. Every thread of the group stores its own value in a slot of its
//! own, the group meets at a barrier, and every thread then loads the slot
//! of the lane it reads:
//!
//! ```text
//! slots[set][thread] = value        every thread, its own slot
//! barrier (group sync, shared-memory fence)
//! result = slots[set][wave's first thread + lane]
//! ```
//!
//! `thread` is the relabelled flat thread id ([`Lower::relabelled_flat`]),
//! whose low six bits are the console lane and whose other bits name the
//! console wave, so a group of several console waves has all its waves'
//! lanes side by side and each wave reads its own.
//!
//! Where: only in a block every lane runs exactly once
//! ([`lds_order::once`]), where a group barrier is legal and where every
//! thread of the group stores before any loads. A read anywhere else is left
//! to the halves' other rules. First user: `v_readlane_b32`, as the console
//! compiler ends a prefix sum over the wave's 64 lanes (the lower half's
//! total, lane 31, added in the upper half): the form of five programs of
//! the title's ray-tracing structure builder.
//!
//! Why the result is the console's:
//!
//! 1. Each host lane is one console lane and holds that lane's value: a
//!    register a lane did not write keeps its old value in the per-lane
//!    form as under the console's EXEC, and `v_readlane_b32` reads the
//!    register whatever EXEC is. So `slots[set][t]` gets what console thread
//!    `t` holds at the instruction.
//! 2. The block is run once by every thread, so each thread stores once,
//!    then waits at the barrier, then loads. The barrier lets no thread on
//!    before all have arrived, and fences shared memory: every load sees
//!    every store of this exchange.
//! 3. Nothing else writes the set meanwhile. The slots are the exchange's
//!    own memory (the program's LDS is another array). Exchange `n` uses set
//!    `n & 1`. Its loads lie between its own barrier and the next
//!    exchange's barrier; the stores to the same set are its own, before its
//!    barrier, and those of exchange `n + 2`, after the barrier of exchange
//!    `n + 1`. A barrier divides the group's work into all of before and all
//!    of after, so no store of the set falls among the loads. (With one set
//!    the next exchange's stores would: some thread still loading while
//!    another already stores. The model test shows both.)
//! 4. A second read of the same value right after (no exchange between)
//!    loads from the same set with no new barrier, by 3.
//!
//! What it rests on, and nothing more: the relabelling (each host wave
//! holds a 32-aligned run of flat thread ids, as every program in halves
//! assumes) and a group barrier that syncs shared memory across the host
//! waves of a group (as [`split_wave_barriers`] and [`lds_order`] assume).
//! No wave operation is asked about other lanes: the host's ballots, lane
//! reads and helper-lane answers are not used.
//!
//! Not exact here, as before: a lane the group does not have (a wave cut
//! short by the group's size) reads a slot nothing stored, where the console
//! reads a register nothing wrote. The loop cap ([`LOOP_LIMIT`]) sends a
//! lane out of the program, which would then miss the barrier; the same
//! holds for the barriers of [`split_wave_barriers`].
//!
//! The exchange through group-shared memory between two barriers is
//! belobug's (Belobug/VIctory-Belobug-AMD 5eac4fa, `Lower::exchange`, two
//! dwords for the halves' ballots) and hyphennn's (fxceless/victory-fork
//! 578d459, `exchange_halves`); both run every lane through every branch
//! to place it anywhere, which hung an NVIDIA card with barriers in loops
//! (e9765a7 there). This form keeps the per-lane control flow, stays out of
//! divergent code, stores per lane instead of per half, and needs one
//! barrier where they need two.

use super::*;

/// D3D12's limit on a compute shader's thread-group shared memory, in bytes.
const SHARED_BYTES: u32 = 32768;

/// The exchanges of one program.
#[derive(Default)]
pub(super) struct State {
    /// Blocks every lane runs exactly once, when first asked.
    once: Option<Vec<bool>>,
    /// The slots: two sets of one dword per thread of the group (rounded
    /// up to whole console waves), group-shared.
    slots: Option<V>,
    /// Exchanges made.
    pub made: u32,
    /// The value the last exchange handed over, and the first slot of its
    /// set.
    last: Option<(V, u32)>,
}

impl Lower<'_> {
    /// The slots per set: the group's threads, in whole console waves; and
    /// whether both sets fit beside the program's LDS.
    fn exchange_slots(&self) -> Option<u32> {
        let threads: u32 = self.s.threads?.iter().product();
        let per_set = threads.div_ceil(64) * 64;
        (per_set * 8 + self.s.lds_dwords * 4 <= SHARED_BYTES).then_some(per_set)
    }

    /// What console lane `lane` of the reader's wave holds in `x`, for every
    /// lane of the group, exact across the halves (the module notes); `None`
    /// where the exchange does not apply: not asked for, not in halves, or
    /// not in a block every lane runs exactly once.
    pub(super) fn exchanged_lane(&mut self, x: V, lane: V) -> Option<V> {
        if !(self.halves && self.x.mac.exchange) {
            return None;
        }
        let per_set = self.exchange_slots()?;
        let b = self.cur_ir;
        let p = self.x.p;
        if !self.exchange.once.get_or_insert_with(|| lds_order::once(p))[b] {
            return None;
        }
        let i32t = self.s.t.i32;
        let slots = match self.exchange.slots {
            Some(g) => g,
            None => {
                let arr = self.s.m.array(2 * per_set, i32t);
                let g = self.s.m.global("exchange", arr, 3);
                self.exchange.slots = Some(g);
                g
            }
        };
        let flat = self.relabelled_flat();
        let c = self.cur;
        let base = match self.exchange.last {
            Some((v, base)) if v == x => base,
            _ => {
                let base = (self.exchange.made & 1) * per_set;
                // The thread's own slot; the relabelled id is below the
                // set's size while host waves hold aligned runs of ids, and
                // is held there otherwise so no store leaves the array.
                let n = self.s.m.i32(per_set);
                let inside = self.s.m.cmp(c, pred::IULT, flat, n);
                let zero = self.s.m.i32(0);
                let own = self.s.m.select(c, inside, flat, zero);
                let k = self.s.m.i32(base);
                let own = self.s.m.bin(c, Bin::Add, own, k);
                let at = self.s.m.element(c, slots, own);
                self.s.m.store(c, at, x);
                self.group_barrier();
                self.exchange.made += 1;
                self.exchange.last = Some((x, base));
                base
            }
        };
        // The slot of that lane of the reader's console wave.
        let c = self.cur;
        let (wave, in_wave) = (self.s.m.i32(!63), self.s.m.i32(63));
        let first = self.s.m.bin(c, Bin::And, flat, wave);
        let lane = self.s.m.bin(c, Bin::And, lane, in_wave);
        let from = self.s.m.bin(c, Bin::Or, first, lane);
        let n = self.s.m.i32(per_set);
        let inside = self.s.m.cmp(c, pred::IULT, from, n);
        let zero = self.s.m.i32(0);
        let from = self.s.m.select(c, inside, from, zero);
        let k = self.s.m.i32(base);
        let from = self.s.m.bin(c, Bin::Add, from, k);
        let at = self.s.m.element(c, slots, from);
        Some(self.s.m.load(c, at))
    }
}
