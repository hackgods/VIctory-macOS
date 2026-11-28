//! Hand-overs across the 32-lane halves in code only some lanes ran
//! (`Options::halves_walk`, with `Options::halves_exchange`;
//! `notes/ray-tracing-plan.md`, "Tiled sun-shadow programs").
//!
//! A console wave64 that runs as two host waves can hand a value across its
//! halves only in a block every lane runs exactly once
//! (`lower::exchange`). The title's tiled sun-shadow programs count lane
//! masks ("does any lane of the wave still need the wide filter?", "did any
//! lane see a caster?") behind their early-out and inside `if` bodies,
//! where the per-lane form sends the lanes outside EXEC round the code. So
//! each half answered for itself: refused, or with half-local reads two
//! answers where the console has one.
//!
//! Here such a program is translated a second way: with the skips around
//! those places left out ([`crate::ir::walk`]), so every lane walks the
//! skipped code, EXEC bit clear, and the places are blocks every lane runs.
//! The second way is taken only if it is the same program for every lane
//! and exact across the halves:
//!
//! 1. Only skips the console takes when no lane of the wave is left
//!    (`s_cbranch_execz`, and the early-out's long form), around a region
//!    with one way in and one way out whose scalars nothing after it reads
//!    (`ir::walk::skip`).
//! 2. A lane outside the EXEC the skip tested does nothing in the region
//!    ([`idle`], checked on the new program's lane masks): no vector
//!    instruction of the region can run in it, and it leaves the region
//!    outside EXEC. Its registers, the memory and the masks it is counted
//!    in are then what they were when it went round.
//! 3. The new program translates with no read across the halves answered
//!    by a half alone (half-local reads off): every such read is handed
//!    over, or was exact before.
//!
//! If any of the three fails the program is translated as before, bit for
//! bit. So is a program that is exact already, and one that reads nothing
//! across its halves where not every lane runs.
//!
//! What it costs the GPU: the walked code's scalar instructions and its
//! unguarded arithmetic run in every lane of the group, where the lanes
//! outside EXEC used to leave. Regions that hold no hand-over are still
//! gone round.
//!
//! The first round opens the places the program as it stands asks for; a
//! hand-over made there can uncover the next (a count that decided a branch
//! around another count), so the rounds repeat a few times.
//!
//! # Whole-wave code
//!
//! Mac (`Options::halves_whole_wave`, off unless asked for;
//! `notes/far-ray.md`). The console's compiler writes a wave's sum or
//! prefix sum inside an `if` with EXEC set to every lane
//! (`s_or_saveexec_b64 s, -1` ... `s_mov_b64 exec, s`): the lanes outside
//! the `if` run that code too, adding nothing (`v_cndmask_b32` of the saved
//! EXEC gives them 0). The title's ray and shader bin offsets
//! (`CS_GenerateRayBinOffsets`, `CS_GenerateShaderBinOffsets`) count their
//! packets that way, one bin per lane, and rule 2 refuses them: a lane
//! outside the tested EXEC runs vector code there. With the switch rule 2
//! lets such a lane run vector arithmetic (no memory, no LDS) of the
//! region, where no lane reads what that left in its VGPRs after the region
//! ([`unread_after`]). Why each lane then does what it did on the console:
//!
//! * a console wave with some lane inside ran the region in every lane, the
//!   whole-wave code included: the walked program is that program with a
//!   branch left out that this wave did not take;
//! * a console wave with no lane inside went round the region. Walked, its
//!   lanes run only the region's whole-wave arithmetic (rule 2's other
//!   instructions need a lane inside), which changes their VGPRs and
//!   scalars only: no lane reads those VGPRs after the region before
//!   writing them, and no scalar the region changed is read after it
//!   (rule 1). EXEC leaves as it came (rule 2's last check).
//!
//! What a whole-wave instruction left in a VGPR counts as read where a later
//! instruction names the register as an operand ([`ir::walk::operand_vgprs`]:
//! not the old value a vector write keeps in the lanes it skips), and as
//! gone where an instruction every lane that reaches it runs writes it; a
//! write under a guard ends it only for a later read under the same guard.
//! Over the program's edges and its per-lane form's, so over every way a
//! lane goes.
//!
//! LDS that a group of one console wave shares with no barrier the halves
//! can use (the title's ray budgeting with ray bins,
//! `sr_tracing:CS_RayBudgeting_Cull_RayBinning`: threads 0 to 7 clear eight
//! counters, every lane with a ray adds to one, threads 0 to 7 read them;
//! all behind the early-out) asks for its blocks to be opened as a
//! hand-over does (`lower::exchange::want_lds`). In the walked form the
//! accesses are in blocks every lane runs, where `split_wave_barriers`
//! orders them. The form the walk keeps must then order its LDS: it is
//! lowered without the race (`Options::halves_shared_lds`), and only the
//! lowering that finds the places to open may race.
//!
//! The reads across the halves in such code are then handed over where the
//! group runs together (`lower::exchange`): with the switch also in blocks
//! every thread runs as often as every other on a loop's body
//! (`lower::lds_rounds::met_with`), and a first-lane read under a narrowed
//! EXEC (the base the bin's atomic returned) by the console's EXEC as a
//! ballot and that lane's value.

use super::*;
use crate::gfx10::Fields;
use crate::ir::walk::Skip;
use crate::ir::{EXEC, Reg};
use crate::structure::{Atom, Bool, Map, Structured};

/// How often a program is opened further before it is left as it was.
const ROUNDS: usize = 4;

/// A program in another form, with everything the emitter needs of it.
pub(super) struct Form {
    pub p: ir::Program,
    pub st: Structured,
    pub io: io::Io,
    pub res: resource::Resources,
}

/// `input`'s program translated with the skips around its hand-overs left
/// out, if that makes every read across the halves exact (the module
/// notes); `None`: translate it as it is.
pub(super) fn exact(input: &EmitInput<'_>, options: Options) -> Option<HostShader> {
    let native = input.native;
    if !(options.halves_walk
        && options.halves_exchange
        && options.host_wave32
        && native.wave64
        && native.stage == HwStage::Cs)
    {
        return None;
    }
    let start = (input.p, input.st, input.io, input.res);
    open(native, start, options.halves_whole_wave, |(p, st, io, res), local| {
        let form = EmitInput {
            c: input.c,
            native,
            p,
            st,
            io,
            res,
        };
        // Mac (`whole_wave`): the form the walk keeps (translated without
        // half-local reads) must not leave LDS to race between the halves
        // either; the one that only finds the places to open (with them,
        // never kept) may, so that it gets to every place (the module
        // notes, "Whole-wave code").
        let race = if options.halves_whole_wave && !options.halves_whole_wave_no_lds {
            local
        } else {
            options.halves_shared_lds
        };
        let options = Options {
            halves_local: local,
            halves_shared_lds: race,
            ..options
        };
        emit_form(&form, options, None)
    })
}

/// A program as the stages before the emitter leave it.
pub(super) type Stages<'a> = (
    &'a ir::Program,
    &'a Structured,
    &'a io::Io,
    &'a resource::Resources,
);

/// [`exact`] with the lowering given: `lower` translates a form of the
/// program, with half-local reads (its second argument) or without.
///
/// With `whole_wave` (`Options::halves_whole_wave`) it also opens a region
/// whose whole-wave code runs in lanes outside the tested EXEC ([`idle`]).
pub(super) fn open<T>(
    native: &NativeProgram,
    start: Stages<'_>,
    whole_wave: bool,
    mut lower: impl FnMut(Stages<'_>, bool) -> Result<T, ShaderError>,
) -> Option<T> {
    let mut form: Option<Form> = None;
    for _ in 0..ROUNDS {
        let next = {
            let cur = match &form {
                Some(f) => (&f.p, &f.st, &f.io, &f.res),
                None => start,
            };
            // Where this form asks for a hand-over and not every lane runs.
            let (made, wanted) = lower::exchanges_wanted(|| lower(cur, true));
            made.ok()?;
            // Nothing to open, or the program is exact as it stands (what
            // was asked for there is a value nothing ends up needing).
            if wanted.is_empty() || form.is_none() && lower(cur, false).is_ok() {
                return None;
            }
            let mut around = vec![false; cur.0.blocks.len()];
            for b in wanted {
                around[b] = true;
            }
            let (p, skips) = ir::walk::walked(cur.0, &around)?;
            let st = structure::structure(&p).ok()?;
            if !idle(&p, &st, &skips, whole_wave) {
                return None;
            }
            let res = resource::track(&p, &native.layout, native.stage, Some(&st)).ok()?;
            let io = io::raise(&p, native).ok()?;
            Form { p, st, io, res }
        };
        if let Ok(made) = lower((&next.p, &next.st, &next.io, &next.res), false) {
            return Some(made);
        }
        form = Some(next);
    }
    None
}

/// Whether, in the program `p` that [`ir::walk::walked`] made, a lane
/// outside the EXEC a left-out skip tested does nothing in the skip's
/// region (the module notes, 2).
///
/// A loop in the region joins EXEC at its head, and the join is a mask of
/// its own to the structure pass. Such joins are taken by induction over
/// the lane's steps ([`inside`]): a join holds no lane outside the tested
/// EXEC if no mask it takes in does, the other joins being granted.
///
/// With `whole_wave` (`Options::halves_whole_wave`; the module notes,
/// "Whole-wave code") such a lane may also run vector arithmetic of the
/// region, where nothing after the region reads what it leaves in its
/// VGPRs ([`unread_after`]).
fn idle(p: &ir::Program, st: &Structured, skips: &[Skip], whole_wave: bool) -> bool {
    let mask = |v| {
        let known = st.masks.get(&v).or_else(|| st.other.get(&v));
        known.or_else(|| st.joined.get(&v)).copied()
    };
    skips.iter().all(|s| {
        let Some(tested) = mask(s.exec) else {
            return false;
        };
        let joins = inside(p, st, s, tested, &mask);
        // Whether a lane at block `at` can meet `runs` outside that EXEC.
        let outside = |at: usize, runs: Bool| {
            let mut all = vec![(st.reach[at], true), (runs, true), (tested, false)];
            all.extend(joins.iter().map(|j| (*j, false)));
            st.bools.meet(&all)
        };
        // VGPRs whole-wave code of the region writes in such a lane.
        let mut written = Vgprs::default();
        for (x, block) in p.blocks.iter().enumerate().filter(|(x, _)| s.inside[*x]) {
            for (i, inst) in block.insts.iter().enumerate() {
                if matches!(
                    inst.fields,
                    Fields::Sop { .. } | Fields::Sopp { .. } | Fields::Smem { .. }
                ) {
                    continue;
                }
                if inst.read(Reg::S(EXEC)).is_none() {
                    // (`v_writelane_b32` writes its lane whatever EXEC is)
                    if inst.writes.iter().any(|w| matches!(w.0, Reg::V(_))) {
                        return false;
                    }
                    continue;
                }
                if outside(x, st.guards[x][i].unwrap_or(Bool::TRUE)) {
                    // Only arithmetic: no memory, no LDS.
                    if !(whole_wave && matches!(inst.fields, Fields::Valu(_))) {
                        return false;
                    }
                    for &(reg, _) in &inst.writes {
                        if let Reg::V(v) = reg {
                            written.insert(v);
                        }
                    }
                }
            }
        }
        let unread = written.is_empty()
            || s.join.is_none_or(|j| unread_after(p, st, j, &written));
        unread
            && s.leaves
                .iter()
                .all(|&(x, e)| mask(e).is_some_and(|m| !outside(x, m)))
    })
}

/// The lane-mask joins in the region of `s` (each as its own mask, an
/// [`Atom::Phi`]) that hold no lane outside `tested`: all of them taken to
/// hold first, then those taken out one of whose incoming masks is unknown
/// or can, on its edge and with the others still granted, hold a lane
/// outside `tested`, until none is. What is left holds at every step of a
/// lane: a join's value is one of its incoming masks, made of masks that
/// held it before.
fn inside(
    p: &ir::Program,
    st: &Structured,
    s: &Skip,
    tested: Bool,
    mask: &dyn Fn(ir::Value) -> Option<Bool>,
) -> Vec<Bool> {
    let own = |f: &ir::Phi| {
        mask(f.value).filter(|m| {
            matches!(st.bools.node(*m), Some((Atom::Phi(v), Bool::FALSE, Bool::TRUE)) if v == f.value)
        })
    };
    let mut joins: Vec<(usize, &ir::Phi, Bool)> = (0..p.blocks.len())
        .filter(|b| s.inside[*b])
        .flat_map(|b| p.blocks[b].phis.iter().map(move |f| (b, f)))
        .filter_map(|(b, f)| own(f).map(|m| (b, f, m)))
        .collect();
    loop {
        let granted: Vec<(Bool, bool)> = joins.iter().map(|j| (j.2, false)).collect();
        let holds = |(b, f, _): &(usize, &ir::Phi, Bool)| {
            p.blocks[*b].preds.iter().zip(&f.args).all(|(&q, &a)| {
                mask(a).is_some_and(|m| {
                    let mut all = vec![(st.reach[q], true), (m, true), (tested, false)];
                    all.extend(&granted);
                    !st.bools.meet(&all)
                })
            })
        };
        let kept: Vec<_> = joins.iter().filter(|j| holds(j)).copied().collect();
        if kept.len() == joins.len() {
            return joins.iter().map(|j| j.2).collect();
        }
        joins = kept;
    }
}

/// A set of VGPRs, by number.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct Vgprs([u64; 4]);

impl Vgprs {
    fn insert(&mut self, v: u8) {
        self.0[usize::from(v / 64)] |= 1 << (v % 64);
    }

    fn contains(&self, v: u8) -> bool {
        self.0[usize::from(v / 64)] >> (v % 64) & 1 == 1
    }

    fn is_empty(&self) -> bool {
        self.0 == [0; 4]
    }
}

/// Whether no lane reads, from block `join` on, a VGPR of `written` before
/// it writes it again itself (the module notes, "Whole-wave code"): what
/// whole-wave code left there in a lane that went round the region before
/// is never seen. A VGPR is read where an instruction names it as an
/// operand ([`ir::walk::operand_vgprs`]); a lane writes it where an
/// instruction runs in every lane that reaches it (no guard). A guarded
/// write keeps the old value in the lanes it skips, so it ends nothing.
/// Over every edge of the program and of its per-lane form, so over every
/// way a lane can go.
fn unread_after(p: &ir::Program, st: &Structured, join: usize, written: &Vgprs) -> bool {
    use crate::structure::LaneExit;
    let n = p.blocks.len();
    let mut uses = vec![Vgprs::default(); n];
    let mut ends = vec![Vgprs::default(); n];
    for (b, block) in p.blocks.iter().enumerate() {
        // VGPRs last written in this block under a guard, with the guard: a
        // read under the same guard reads what those lanes wrote.
        let mut by: Map<u8, Bool> = Map::default();
        for (i, inst) in block.insts.iter().enumerate() {
            if matches!(
                inst.fields,
                Fields::Sop { .. } | Fields::Sopp { .. } | Fields::Smem { .. }
            ) {
                continue;
            }
            let Some(operands) = ir::walk::operand_vgprs(inst, p.wave64) else {
                return false;
            };
            let guard = st.guards[b][i].filter(|g| *g != Bool::TRUE);
            for v in operands {
                if !ends[b].contains(v) && (guard.is_none() || by.get(&v) != guard.as_ref()) {
                    uses[b].insert(v);
                }
            }
            for &(reg, _) in &inst.writes {
                if let Reg::V(v) = reg {
                    match guard {
                        None => {
                            ends[b].insert(v);
                            by.remove(&v);
                        }
                        Some(g) => {
                            by.insert(v, g);
                        }
                    }
                }
            }
        }
    }
    let successors = |b: usize| {
        let mut s = p.blocks[b].exit.successors();
        match st.exits.get(b) {
            Some(LaneExit::Jump(t)) => s.push(*t),
            Some(LaneExit::Branch {
                taken, not_taken, ..
            }) => s.extend([*taken, *not_taken]),
            _ => {}
        }
        s
    };
    let mut live = vec![Vgprs::default(); n];
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..n).rev() {
            let mut out = Vgprs::default();
            for t in successors(b) {
                for (o, l) in out.0.iter_mut().zip(live[t].0) {
                    *o |= l;
                }
            }
            let mut inn = Vgprs::default();
            for w in 0..4 {
                inn.0[w] = uses[b].0[w] | out.0[w] & !ends[b].0[w];
            }
            if inn != live[b] {
                live[b] = inn;
                changed = true;
            }
        }
    }
    (0..4).all(|w| live[join].0[w] & written.0[w] == 0)
}
