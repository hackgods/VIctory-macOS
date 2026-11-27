//! LDS that a group of one console wave shares round after round of a
//! loop, with lanes that leave on the way ([`Mac::lds_rounds`],
//! `emit::Options::lds_rounds`; `notes/replay-determinism.md`, "HairOnly,
//! 27 November").
//!
//! The console runs such a group as one wave in lockstep, so its compiler
//! puts no barrier between one lane's LDS write and another lane's read.
//! On a host of 32-lane waves the group is two waves, and
//! [`split_wave_barriers`] gives them the console's order with group
//! barriers in blocks every lane runs together ([`in_step`]). It finds none
//! in the title's hair shadow-mask program
//! (`shadowmask:CS_GenerateShadowMasks_HairOnly`, `0a5ecfee6b79e13b`): thread
//! 0 alone writes one LDS dword in each round of a loop over the tile's
//! lights, and every lane reads it right after, where the threads meet
//! again. Two things stand in the way there:
//!
//! 1. Lanes leave the program before the loop (the tile's lanes outside
//!    the hair and the screen), so no block after that point is one every
//!    lane runs.
//! 2. The loops are counted from values the lanes read with
//!    `v_readfirstlane`/`v_readlane`: a light list loaded at the tile's
//!    coordinates, which every lane of the group computes alike. No scalar
//!    rule sees that.
//!
//! This is the same barrier placement ([`split_wave_barriers_in`]) over
//! other blocks: those every thread of the group runs as often as every
//! other **that has not ended** ([`met`]). Which:
//!
//! * A branch one of whose ways leads only to the end of the program (an
//!   early-out, a loop whose exit is the end) need not be taken alike: a
//!   thread that takes it ends, the others go on as if it were a jump.
//! * Every other branch that can go round the block (the block lies between
//!   it and the block where its ways join: [`round`]) must be decided for
//!   the group: `s_cbranch_scc0`/`scc1` on a group value ([`values`]).
//!   That includes the branches that go round a loop holding the block.
//!   A branch whose ways join before the block (a lane's own `if`, a loop
//!   nested in the round) does not change how often the block runs.
//! * A group value is what [`super::together`] takes for one (inputs that
//!   are the same for the group, scalar code of them, scalar loads of memory
//!   the program does not write, joins where the group arrives as one), and
//!   also the first-lane read of a VGPR value every thread of the group
//!   holds alike ([`values_with`]): written from group values by plain
//!   vector arithmetic, loaded at such an address from memory the program
//!   does not write, or the thread id shifted out of a coordinate
//!   (`(s << k) + tid >> k` with the group's size along that axis at most
//!   `2^k`, the title's tile coordinates).
//!
//! Why it is the console's order: between two barriers the threads that
//! have not ended run the same met blocks in the same order (every branch
//! that could make one of them run a met block more often is decided for
//! the group), so the k-th barrier of every such thread is the same one,
//! and [`split_wave_barriers_in`]'s argument holds as with [`in_step`].
//!
//! What it rests on, beyond that argument's own premises:
//!
//! * **A thread that has ended does not hold up a group barrier.** DXIL
//!   leaves a barrier that not every thread of the group reaches undefined.
//!   Measured on D3DMetal (`lds_rounds_tests`, the GPU tests): lanes that
//!   left a host wave, and a host wave whose lanes all left, do not stop
//!   the others' barriers. That is all this form needs of the host.
//! * As in [`super::together`]: memory read by a scalar or vector load that
//!   decides a loop does not change while the group runs. The program's own
//!   stores are ruled out by descriptor.
//! * As for every barrier here: the loop cap ([`LOOP_LIMIT`]) would send a
//!   lane out of the program in the middle of a round.
//!
//! LDS accesses outside the met blocks are left as [`split_wave_barriers`]
//! leaves them: a read there of what code outside them may have written
//! since the last barrier refuses the form, and the program stays as it
//! was. The program's other reads across the halves keep their own rules
//! ([`super::halves`]): with half-local reads they are each half's own, as
//! without this.

use super::*;

/// The barriers that give `x`'s group of `group` threads with
/// `lds_dwords` of LDS the console's order across loop rounds (the module
/// notes), where [`Mac::lds_rounds`] asks for them and the group is one
/// console wave that a host of narrower waves splits; `None` where it does
/// not apply or no such placement is known.
pub(crate) fn wanted(x: &Parts, group: u32, lds_dwords: u32) -> Option<Set<(usize, usize)>> {
    let wave = if x.native.wave64 { 64 } else { 32 };
    let applies = x.mac.lds_rounds
        && (x.narrow_waves || x.host_wave32)
        && group > 32
        && group <= wave
        && lds_dwords > 0;
    if !applies {
        return None;
    }
    split_wave_barriers_in(x.p, &met(x)?, true)
}

/// Per block, whether every thread of the group that has not ended runs it
/// as often as every other (the module notes); `None` where the program's
/// shape is not followed (an indirect branch, a barrier of its own, a block
/// from which no way leads to the end).
pub(super) fn met(x: &Parts) -> Option<Vec<bool>> {
    met_with(x, true)
}

/// [`met`]; without `ends`, a branch with a way that leads only to the end
/// counts as any other (it must be decided for the group), so the blocks
/// are those every thread of the group runs as often as every other, none
/// having ended (mac: [`super::exchange`] with [`Mac::whole_wave`]).
pub(super) fn met_with(x: &Parts, ends: bool) -> Option<Vec<bool>> {
    let p = x.p;
    let n = p.blocks.len();
    let own_barrier = |b: &Block| b.insts.iter().any(|i| i.name() == "s_barrier");
    if p.blocks
        .iter()
        .any(|b| b.exit == Exit::Indirect || own_barrier(b))
    {
        return None;
    }
    let succ = |b: usize| p.blocks[b].exit.successors();
    // Blocks with only the end ahead (no cycle can be one).
    let mut ending: Vec<bool> = (0..n).map(|b| succ(b).is_empty()).collect();
    loop {
        let more: Vec<usize> = (0..n)
            .filter(|&b| !ending[b] && succ(b).iter().all(|&t| ending[t]))
            .collect();
        if more.is_empty() {
            break;
        }
        for b in more {
            ending[b] = true;
        }
    }
    let rounds = round(p)?;
    let branches: Vec<usize> = (0..n)
        .filter(|&d| matches!(p.blocks[d].exit, Exit::Branch { .. }))
        .collect();
    // A way to the end alone, and one on.
    let to_end = |d: usize| {
        let s = succ(d);
        s.iter().any(|&t| ending[t]) && s.iter().any(|&t| !ending[t])
    };
    let mut met: Vec<bool> = ending.iter().map(|e| !e).collect();
    loop {
        let same = values(x, &met);
        let mut next: Vec<bool> = ending.iter().map(|e| !e).collect();
        for &d in &branches {
            if ends && to_end(d) || decided(x, &same, d) {
                continue;
            }
            for (b, r) in rounds[d].iter().enumerate() {
                if *r {
                    next[b] = false;
                }
            }
        }
        // Fewer met blocks, fewer group values: this only shrinks.
        if next == met {
            return Some(met);
        }
        met = next;
    }
}

/// Per block `d`, the blocks a branch at its end can go round: reachable
/// from it without passing a block that post-dominates it (a virtual exit
/// after every end), and `d` itself where a cycle leads back to it (a
/// loop's latch: it runs once a round). `None` if some block has no way to
/// the end.
fn round(p: &Program) -> Option<Vec<Vec<bool>>> {
    let n = p.blocks.len();
    let succ = |b: usize| p.blocks[b].exit.successors();
    let words = n.div_ceil(64);
    let full = vec![u64::MAX; words];
    let only = |b: usize| {
        let mut v = vec![0; words];
        v[b / 64] |= 1 << (b % 64);
        v
    };
    let mut pdom: Vec<Vec<u64>> = (0..n)
        .map(|b| {
            if succ(b).is_empty() {
                only(b)
            } else {
                full.clone()
            }
        })
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
    // A block that cannot reach an end keeps the full set.
    if pdom.iter().any(|v| *v == full) {
        return None;
    }
    Some(
        (0..n)
            .map(|d| {
                let mut seen = vec![false; n];
                if !matches!(p.blocks[d].exit, Exit::Branch { .. }) {
                    return seen;
                }
                let mut stack = succ(d);
                while let Some(b) = stack.pop() {
                    let after = b != d && pdom[d][b / 64] >> (b % 64) & 1 == 1;
                    if !after && !std::mem::replace(&mut seen[b], true) {
                        stack.extend(succ(b));
                    }
                }
                seen
            })
            .collect(),
    )
}

/// Whether the branch block `b` ends in is taken the same way by every
/// thread of the group: on SCC, a group value, and a uniform branch in the
/// per-lane form too (as [`super::together`] has it).
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

/// Values that are the same in every thread of the group that holds them,
/// given the blocks taken to be met (the module notes): scalars, and VGPR
/// values every lane holds alike.
fn values(x: &Parts, met: &[bool]) -> Set<Value> {
    let p = x.p;
    // Joins where the group arrives as one, all taken to hold first.
    let whole = |b: usize| met[b] && p.blocks[b].preds.iter().all(|&q| met[q]);
    let scalar = |r: Reg| matches!(r, Reg::Scc) || matches!(r, Reg::S(c) if c < VCC);
    let mut joins: Vec<&Phi> = (0..p.blocks.len())
        .filter(|&b| whole(b))
        .flat_map(|b| p.blocks[b].phis.iter().filter(|f| scalar(f.reg)))
        .collect();
    loop {
        let same = values_with(x, &joins);
        let held = joins.len();
        joins.retain(|f| f.args.iter().all(|a| same.contains(a)));
        if joins.len() == held {
            return same;
        }
    }
}

/// Vector instructions whose result in a lane is a function of that lane's
/// operands alone, read as numbers (no lane mask, no other lane, no lane
/// index, no carry).
fn plain(name: &str) -> bool {
    ![
        "lane",
        "mbcnt",
        "cmp",
        "cndmask",
        "_co_",
        "addc",
        "subb",
        "div_fmas",
        "div_scale",
        "interp",
        "movrel",
        "swap",
    ]
    .iter()
    .any(|k| name.contains(k))
}

/// [`values`] with `joins` taken as group values.
fn values_with(x: &Parts, joins: &[&Phi]) -> Set<Value> {
    let p = x.p;
    let mut same: Set<Value> = joins.iter().map(|f| f.value).collect();
    let set = x.io.initialized_sgprs.unwrap_or(0).min(VCC);
    for &(reg, v) in &p.inputs {
        let info = (x.io.inputs.iter()).any(|i| i.0 == v && !matches!(i.1, Input::GroupId(_)));
        if matches!(reg, Reg::S(c) if c < set) && !info {
            same.insert(v);
        }
    }
    let StageIo::Cs(cs) = &x.native.io else {
        return same;
    };
    // The thread id's coordinates on entry, with the group's size along each.
    let ids: Map<Value, u32> = (x.io.inputs.iter())
        .filter_map(|&(v, i)| match i {
            Input::ThreadIdInGroup(a) => cs.threads.get(usize::from(a)).map(|&k| (v, k)),
            _ => None,
        })
        .collect();
    // Where each value is written.
    let mut defs: Map<Value, (usize, usize)> = Map::default();
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            defs.extend(inst.writes.iter().map(|w| (w.1, (b, i))));
        }
    }
    let at = |s: (usize, usize)| &p.blocks[s.0].insts[s.1];
    let unguarded = |s: (usize, usize)| x.st.guards[s.0][s.1].is_none();
    // EXEC on entry is every lane where the group is whole console waves.
    let whole_waves = cs.threads.iter().product::<u32>() % 64 == 0;
    let entry: Vec<Value> = (p.inputs.iter())
        .filter(|r| matches!(r.0, Reg::S(c) if c == EXEC || c == EXEC + 1))
        .map(|r| r.1)
        .collect();
    let is_exec = |r: Reg| matches!(r, Reg::S(c) if c == EXEC || c == EXEC + 1);
    // `s_mov_b64 s[k:k+1], exec` of the entry's EXEC: every lane.
    let every_lane = |v: Value| {
        whole_waves
            && defs.get(&v).is_some_and(|&s| {
                let i = at(s);
                i.name() == "s_mov_b64"
                    && entry.len() == 2
                    && (i.reads.iter()).all(|r| is_exec(r.0) && entry.contains(&r.1))
            })
    };
    // Memory the program writes by address, and the buffers and images it
    // writes (as `together` tells them).
    let insts = || p.blocks.iter().flat_map(|b| &b.insts);
    let stores = |i: &Inst| i.name().contains("store") || i.name().contains("atomic");
    let by_address = insts().any(|i| matches!(i.fields, Fields::Flat { .. }) && stores(i));
    let written: Vec<&Source> = (x.res.descriptors.iter())
        .filter(|d| d.role == Role::Resource && stores(at(d.site)))
        .map(|d| &d.source)
        .collect();
    let unwritten = |site: Site| {
        let here =
            || (x.res.descriptors.iter()).filter(|d| d.site == site && d.role == Role::Resource);
        here().next().is_some() && here().all(|d| !written.contains(&&d.source))
    };
    // Reverse postorder: a value's instruction comes before its uses, but
    // for joins.
    for b in crate::ir::reverse_postorder(&p.blocks) {
        for (i, inst) in p.blocks[b].insts.iter().enumerate() {
            let name = inst.name();
            let group = |r: &(Reg, Value)| !is_exec(r.0) && same.contains(&r.1);
            // Its operands, EXEC aside (a vector instruction's lane mask).
            let of_group = inst.reads.iter().filter(|r| !is_exec(r.0)).all(group);
            // A vector instruction also reads what it writes over (the old
            // value lanes outside EXEC keep); unguarded, every lane holding
            // the result ran it, so only its operands count.
            let operands = |v: &crate::gfx10::Valu| {
                let named = |r: Reg| {
                    v.src.iter().any(|o| match *o {
                        Operand::Vector { reg, dwords } => {
                            matches!(r, Reg::V(k) if (reg..reg + dwords.max(1)).contains(&k))
                        }
                        Operand::Scalar { code, dwords } => {
                            matches!(r, Reg::S(k) if (code..code + dwords.max(1)).contains(&k))
                        }
                        _ => false,
                    })
                };
                let old = |r: Reg| inst.writes.iter().any(|w| w.0 == r) && !named(r);
                inst.reads
                    .iter()
                    .filter(|r| !is_exec(r.0) && !old(r.0))
                    .all(group)
            };
            let vgprs_only = inst.writes.iter().all(|w| matches!(w.0, Reg::V(_)));
            let made = match &inst.fields {
                Fields::Sop { .. } => {
                    let own = [
                        "s_getreg",
                        "s_setreg",
                        "s_movrel",
                        "s_setpc",
                        "s_swappc",
                        "s_call",
                        "s_rfe",
                        "s_subvector",
                        "s_wqm",
                    ];
                    !own.iter().any(|n| name.starts_with(n)) && inst.reads.iter().all(group)
                }
                Fields::Smem { .. } => {
                    !by_address
                        && (name.starts_with("s_load_dword")
                            || name.starts_with("s_buffer_load_dword") && unwritten((b, i)))
                        && inst.reads.iter().all(group)
                }
                Fields::Valu(v) if unguarded((b, i)) && v.dpp.is_none() && v.sdwa.is_none() => {
                    match name {
                        // The first lane of EXEC holds the value as every lane does.
                        "v_readfirstlane_b32" => operands(v),
                        "v_readlane_b32" => {
                            readlane_of_exec(inst, (b, i), &defs, &at, &every_lane)
                                && of_group_but_lane(inst, &same)
                        }
                        "v_lshrrev_b32" => {
                            operands(v) || id_shifted_out(inst, &same, &ids, &defs, &at, &unguarded)
                        }
                        _ => plain(name) && vgprs_only && operands(v),
                    }
                }
                Fields::Image { .. } | Fields::Buffer { .. } => {
                    let load = name.starts_with("image_load")
                        || name.starts_with("buffer_load")
                        || name.starts_with("tbuffer_load");
                    load && unguarded((b, i))
                        && !by_address
                        && unwritten((b, i))
                        && vgprs_only
                        && of_group
                }
                _ => false,
            };
            if made {
                same.extend(inst.writes.iter().map(|w| w.1));
            }
        }
    }
    same
}

/// Whether `v_readlane_b32` `inst` (at `site`) reads a lane inside EXEC:
/// its lane is `s_ff1_i32_b64` of `s_and_b64` of EXEC and a mask of every
/// lane, all three in its block with EXEC unchanged between them. That
/// lane is at the read, so it holds the value the read names.
fn readlane_of_exec<'a>(
    inst: &Inst,
    site: (usize, usize),
    defs: &Map<Value, (usize, usize)>,
    at: &dyn Fn((usize, usize)) -> &'a Inst,
    every_lane: &dyn Fn(Value) -> bool,
) -> bool {
    let Fields::Valu(v) = &inst.fields else {
        return false;
    };
    let Some(Operand::Scalar { code, .. }) = v.src.get(1) else {
        return false;
    };
    let Some(&ff1) = inst.read(Reg::S(*code)).and_then(|l| defs.get(&l)) else {
        return false;
    };
    let first = at(ff1);
    let and: Vec<(usize, usize)> = first
        .reads
        .iter()
        .filter_map(|r| defs.get(&r.1).copied())
        .collect();
    if first.name() != "s_ff1_i32_b64" || and.len() != 2 || and[0] != and[1] {
        return false;
    }
    let mask = at(and[0]);
    let exec = |r: &(Reg, Value)| matches!(r.0, Reg::S(c) if c == EXEC || c == EXEC + 1);
    let others: Vec<Value> = mask
        .reads
        .iter()
        .filter(|r| !exec(r))
        .map(|r| r.1)
        .collect();
    let in_block = and[0].0 == site.0 && ff1.0 == site.0 && and[0].1 < ff1.1 && ff1.1 < site.1;
    let exec_kept = |from: usize, to: usize| {
        // No instruction between rewrites EXEC.
        (from + 1..to).all(|k| {
            let i = at((site.0, k));
            !i.writes
                .iter()
                .any(|w| matches!(w.0, Reg::S(c) if c == EXEC || c == EXEC + 1))
        })
    };
    mask.name() == "s_and_b64"
        && mask.reads.iter().filter(|r| exec(r)).count() == 2
        && others.len() == 2
        && others.iter().all(|&o| every_lane(o))
        && in_block
        && exec_kept(and[0].1, site.1)
}

/// Whether `v_readlane_b32` `inst` reads a group value but for its lane
/// operand.
fn of_group_but_lane(inst: &Inst, same: &Set<Value>) -> bool {
    let Fields::Valu(v) = &inst.fields else {
        return false;
    };
    match v.src.first() {
        Some(&Operand::Vector { reg, .. }) => {
            inst.read(Reg::V(reg)).is_some_and(|x| same.contains(&x))
        }
        _ => false,
    }
}

/// Whether `v_lshrrev_b32 d, k, x` `inst` shifts the thread id out of `x`:
/// `x` is `v_lshl_add_u32 x, s, k, tid` (unguarded) of a group value `s`
/// and a coordinate of the thread id along which the group has at most
/// `2^k` threads. The id is then below `2^k`, adds no carry into bit `k`,
/// and `d` is `s` shifted back: a group value.
fn id_shifted_out<'a>(
    inst: &Inst,
    same: &Set<Value>,
    ids: &Map<Value, u32>,
    defs: &Map<Value, (usize, usize)>,
    at: &dyn Fn((usize, usize)) -> &'a Inst,
    unguarded: &dyn Fn((usize, usize)) -> bool,
) -> bool {
    let konst = |o: &Operand| match *o {
        Operand::Const(c) => crate::resource::inline(c),
        Operand::Literal(k) => Some(k),
        _ => None,
    };
    let Fields::Valu(v) = &inst.fields else {
        return false;
    };
    let (Some(k), Some(&Operand::Vector { reg, .. })) =
        (v.src.first().and_then(konst), v.src.get(1))
    else {
        return false;
    };
    let Some(&site) = inst.read(Reg::V(reg)).and_then(|x| defs.get(&x)) else {
        return false;
    };
    let add = at(site);
    let Fields::Valu(a) = &add.fields else {
        return false;
    };
    let read = |o: Option<&Operand>| match o {
        Some(&Operand::Scalar { code, .. }) => add.read(Reg::S(code)),
        Some(&Operand::Vector { reg, .. }) => add.read(Reg::V(reg)),
        _ => None,
    };
    let (s, tid) = (read(a.src.first()), read(a.src.get(2)));
    add.name() == "v_lshl_add_u32"
        && unguarded(site)
        && a.dpp.is_none()
        && a.sdwa.is_none()
        && (a.neg | a.abs) == 0
        && k < 32
        && a.src.get(1).and_then(konst) == Some(k)
        && s.is_some_and(|s| same.contains(&s))
        && tid
            .and_then(|t| ids.get(&t))
            .is_some_and(|&size| u64::from(size) <= 1 << k)
}
