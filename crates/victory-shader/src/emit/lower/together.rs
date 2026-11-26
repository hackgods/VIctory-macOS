//! Blocks that a thread group runs together: every thread exactly once, or
//! no thread at all ([`Mac::walk`], `emit::Options::halves_walk`;
//! `notes/ray-tracing-plan.md`, "Tiled sun-shadow programs").
//!
//! A hand-over across the halves ([`super::exchange`]) needs a group
//! barrier, and a barrier needs every thread of the group. Blocks every
//! lane runs exactly once ([`lds_order::once`]) have that. So does a block
//! under a branch that every thread of the group takes the same way: the
//! title's programs test a flag of a constant buffer ("is this feature on
//! for the pass?") with `s_cmp` and `s_cbranch_scc`, and count a lane mask
//! inside. Either the whole group comes to the barrier there or none of it.
//!
//! Which branches ([`decided`]): `s_cbranch_scc0` or `scc1` on an SCC that
//! is a group value. A group value ([`values`]) is a scalar that is the same
//! whenever and by whichever thread of the group it is computed:
//!
//! * an SGPR the hardware sets on entry (user data, the group's id), but
//!   not the thread-group info SGPR, whose wave index differs between the
//!   group's waves, and not EXEC;
//! * what a scalar instruction makes of group values and constants alone
//!   (never of EXEC, a lane mask or SCC of one: a first-lane read, a count
//!   of a mask and anything a wave computes are not scalar instructions of
//!   that kind);
//! * what a scalar load reads at an address made of group values, in a
//!   program that stores to no memory by address, and for a buffer whose
//!   descriptor no store or atomic of the program uses.
//!
//! A join of two group values is not one (which of them a thread holds
//! depends on its path), so a group value has no history: its instruction
//! gives the same result each time it runs.
//!
//! Then a block is run together ([`blocks`]) when no cycle goes through it
//! and every branch that can go round it is decided by a group value. Why:
//! take two threads of a group, one that reaches the block and one that
//! does not. Their paths part at a branch. A decided branch sends both the
//! same way, so it is another one, from which the block is reachable; by
//! the rule the block is then on every way from that branch to the end of
//! the program, also on the second thread's.
//!
//! What this rests on, beyond the exchange's own premises: that the memory
//! a scalar load reads does not change while the group runs. The program's
//! own stores are ruled out as far as its descriptors tell (two different
//! descriptors of one buffer would not be seen); other writers do not run
//! during a dispatch. If it failed, part of a group would wait at a barrier
//! for the rest.
//!
//! Only a half's answer is handed over in such a block
//! ([`Lower::exchange_halves_in`]): it has slots of its own. Lane values
//! share two sets of slots in turn, which is right only when every
//! exchange runs ([`super::exchange`], 3), so they stay with the blocks
//! every lane runs.

use super::*;

/// Scalar values that are the same in every thread of a group (the module
/// notes).
fn values(x: &Parts) -> Set<Value> {
    let p = x.p;
    let mut same: Set<Value> = Set::default();
    for &(reg, v) in &p.inputs {
        let info = x
            .io
            .inputs
            .iter()
            .any(|i| i.0 == v && !matches!(i.1, Input::GroupId(_)));
        let set = x.io.initialized_sgprs.unwrap_or(0).min(crate::ir::VCC);
        if matches!(reg, Reg::S(c) if c < set) && !info {
            same.insert(v);
        }
    }
    let insts = || p.blocks.iter().flat_map(|b| &b.insts);
    // Memory the program writes by address, and the buffers it writes.
    let stores = |i: &Inst| i.name().contains("store") || i.name().contains("atomic");
    let by_address = insts().any(|i| matches!(i.fields, Fields::Flat { .. }) && stores(i));
    let written: Vec<&Source> = x
        .res
        .descriptors
        .iter()
        .filter(|d| d.role == Role::Resource && stores(&p.blocks[d.site.0].insts[d.site.1]))
        .map(|d| &d.source)
        .collect();
    let unwritten = |site: Site| {
        x.res
            .descriptors
            .iter()
            .filter(|d| d.site == site && d.role == Role::Resource)
            .all(|d| !written.contains(&&d.source))
    };
    // Reverse postorder: a value's instruction comes before its uses, but
    // for joins, which are no group values.
    for b in crate::ir::reverse_postorder(&p.blocks) {
        for (i, inst) in p.blocks[b].insts.iter().enumerate() {
            let name = inst.name();
            let pure = match inst.fields {
                Fields::Sop { .. } => ![
                    "s_getreg", "s_setreg", "s_movrel", "s_setpc", "s_swappc", "s_call", "s_rfe",
                    "s_subvector", "s_wqm",
                ]
                .iter()
                .any(|n| name.starts_with(n)),
                Fields::Smem { .. } => {
                    !by_address
                        && (name.starts_with("s_load_dword")
                            || name.starts_with("s_buffer_load_dword") && unwritten((b, i)))
                }
                _ => false,
            };
            let group = |r: &(Reg, Value)| {
                !matches!(r.0, Reg::S(c) if c == EXEC || c == EXEC + 1) && same.contains(&r.1)
            };
            if pure && inst.reads.iter().all(group) {
                same.extend(inst.writes.iter().map(|w| w.1));
            }
        }
    }
    same
}

/// Whether the branch block `b` ends in is taken the same way by every
/// thread of a group: on SCC, a group value, and a uniform branch in the
/// per-lane form too.
fn decided(x: &Parts, same: &Set<Value>, b: usize) -> bool {
    let uniform = match x.st.exits[b] {
        LaneExit::Branch { when, .. } => {
            let atoms = x.st.bools.support(when);
            atoms.iter().all(|a| matches!(a, Atom::Uniform(_)))
        }
        _ => false,
    };
    uniform
        && x.p.blocks[b].insts.last().is_some_and(|i| {
            matches!(i.name(), "s_cbranch_scc0" | "s_cbranch_scc1")
                && i.read(Reg::Scc).is_some_and(|v| same.contains(&v))
        })
}

/// For each block, whether a thread group runs it together: every thread
/// exactly once, or none (the module notes).
pub(super) fn blocks(x: &Parts) -> Vec<bool> {
    let p = x.p;
    let n = p.blocks.len();
    if p.blocks.iter().any(|b| b.exit == Exit::Indirect) {
        return vec![false; n];
    }
    let succ = |b: usize| p.blocks[b].exit.successors();
    let reach = |from: usize, stop: &dyn Fn(usize) -> bool| {
        let mut seen = vec![false; n];
        let mut stack = succ(from);
        while let Some(b) = stack.pop() {
            if !stop(b) && !std::mem::replace(&mut seen[b], true) {
                stack.extend(succ(b));
            }
        }
        seen
    };
    // Post-dominators, as bitsets (a virtual exit after every end).
    let words = n.div_ceil(64);
    let full = vec![u64::MAX; words];
    let only = |b: usize| {
        let mut v = vec![0; words];
        v[b / 64] |= 1 << (b % 64);
        v
    };
    let mut pdom: Vec<Vec<u64>> = (0..n)
        .map(|b| if succ(b).is_empty() { only(b) } else { full.clone() })
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..n).rev().filter(|&b| !succ(b).is_empty()) {
            let mut v = full.clone();
            for t in succ(b) {
                for (w, q) in v.iter_mut().zip(&pdom[t]) {
                    *w &= q;
                }
            }
            v[b / 64] |= 1 << (b % 64);
            if v != pdom[b] {
                pdom[b] = v;
                changed = true;
            }
        }
    }
    let same = values(x);
    let mut together: Vec<bool> = (0..n).map(|b| !reach(b, &|_| false)[b]).collect();
    for d in 0..n {
        if !matches!(p.blocks[d].exit, Exit::Branch { .. }) || decided(x, &same, d) {
            continue;
        }
        // The blocks this branch can go round.
        let round = reach(d, &|b| pdom[d][b / 64] >> (b % 64) & 1 == 1);
        for (t, r) in together.iter_mut().zip(round) {
            *t &= !r;
        }
    }
    together
}
