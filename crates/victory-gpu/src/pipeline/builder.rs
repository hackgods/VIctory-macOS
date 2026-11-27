//! The title's acceleration-structure builder runs whole or not at all.
//!
//! `XboxBvhBuilder.dll` builds and refits the ray-tracing structures in
//! title memory every frame through some fifty compute programs, one
//! dispatch after another. A dispatch whose host pipeline is missing is
//! dropped on its own (`State::dispatch`), and some of the builder's are
//! missing for a long time or for good: refused (64-lane waves on a host of
//! 32-lane waves), not translated, or still being made. The stages that do
//! run then work on what the missing ones never wrote, and write the result
//! into title memory: on fxceless/victory-fork, garbage bounds that crashed
//! the title. So the builder's dispatches run only in frames where every
//! pipeline it has created has its host pipeline; otherwise none of them
//! run (the structures keep what they held).
//!
//! Its pipelines are known by where they come from: created by its code (a
//! return address in its image on the stack) or from bytecode in its image.
//!
//! Mac, `VICTORY_BVH_BUILDER=1` (off unless set; `notes/rt-builder.md`):
//! the builder may run. Each of its compute pipelines then gets two forms
//! of its own ([`forms`]), translated with no half of a console wave
//! answering on its own (`shader_cache::exact_options`): the plain one, and
//! one for dispatches counted in threads (`DispatchX` with flag 0x20,
//! `emit::Options::thread_counts`), which the box unpackers need. The
//! builder runs in a frame only when every plain form is made and every
//! thread form is made or refused, and never again after a dispatch
//! counted in threads found no thread form ([`trip`]). Without the switch
//! nothing here changes: the builder stays held while a halves switch is on.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use super::Shader;
use super::make::{self, Pending};

/// The builder's image, `[start, end)`, once mapped.
static IMAGE: Mutex<Option<(usize, usize)>> = Mutex::new(None);

/// The builder's file name.
const FILE: &str = "XboxBvhBuilder.dll";

/// A title DLL was mapped at `base`, `size` bytes: noted if it is the
/// builder.
pub fn note_title_dll(file: &str, base: usize, size: usize) {
    if file.eq_ignore_ascii_case(FILE) {
        tracing::info!(base = format_args!("{base:#x}"), size, "BVH builder mapped");
        *IMAGE.lock().unwrap_or_else(|e| e.into_inner()) = Some((base, base + size));
    }
}

fn image() -> Option<(usize, usize)> {
    *IMAGE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether the pipeline being created is the builder's: `code` (the title's
/// bytecode pointers) lies in its image, or its code is among our callers.
#[inline(never)]
pub(crate) fn created_by_builder(code: &[*const u8]) -> bool {
    let Some((start, end)) = image() else {
        return false;
    };
    let inside = |a: usize| (start..end).contains(&a);
    if code.iter().any(|&c| inside(c as usize)) {
        return true;
    }
    callers().into_iter().any(inside)
}

/// Return addresses on the stack, ours first.
fn callers() -> Vec<usize> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace;
        let mut frames = [std::ptr::null_mut(); 24];
        // SAFETY: room for that many frames.
        let n = unsafe {
            RtlCaptureStackBackTrace(
                1,
                frames.len() as u32,
                frames.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        frames[..usize::from(n)]
            .iter()
            .map(|&f| f as usize)
            .collect()
    }
    #[cfg(not(windows))]
    Vec::new()
}

/// `VICTORY_BVH_BUILDER=1`: the builder may run, from forms of its own
/// (the module notes). Unset or anything else: off.
pub(crate) fn on() -> bool {
    #[cfg(test)]
    if FORCED.load(Ordering::Relaxed) {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("VICTORY_BVH_BUILDER").is_ok_and(|v| v.trim() == "1");
        if on {
            tracing::info!(
                "VICTORY_BVH_BUILDER=1: the BVH builder runs from forms of its own with no half answering alone, once all are made; DispatchX thread counts are honoured"
            );
        }
        on
    })
}

/// What a pipeline of the builder is to [`runs`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// The pipeline its dispatches of groups run: made from the shared
    /// shader as this run's settings say (switch off, or no forms), or its
    /// own plain form (`exact`).
    Plain { exact: bool },
    /// Its form for dispatches counted in threads.
    Threads,
}

/// The builder's pipelines (alive while the title holds them).
static PIPELINES: Mutex<Vec<(Weak<Pending>, Kind)>> = Mutex::new(Vec::new());

/// The builder created pipeline `p`: it counts towards [`runs`], and is
/// made at once (no frame's wait budget holds it back). `exact`: it is a
/// plain form of [`forms`].
pub(crate) fn note(p: &Arc<Pending>, exact: bool) {
    note_kind(p, Kind::Plain { exact });
}

fn note_kind(p: &Arc<Pending>, kind: Kind) {
    PIPELINES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((Arc::downgrade(p), kind));
    make::begin(p);
}

/// A compute pipeline's form for dispatches counted in threads, and its
/// program's group size (`numthreads`), which such a dispatch divides by.
pub(crate) struct ThreadForm {
    pub(crate) pending: Arc<Pending>,
    pub(crate) group: [u32; 3],
}

/// The two forms of one of the builder's compute pipelines.
pub(crate) struct Forms {
    pub(crate) plain: Arc<Pending>,
    pub(crate) threads: ThreadForm,
}

/// Under [`on`], the forms of a compute pipeline the builder creates from
/// `shaders` (one compute stage) whose bytecode the title holds at `code`:
/// read from there, and only if it still hashes to the shader. None for
/// anything else; such a pipeline of the builder keeps the builder held
/// ([`runs`]).
pub(crate) fn forms(shaders: &[Arc<Shader>], code: &[*const u8]) -> Option<Forms> {
    if !on() {
        return None;
    }
    let ([shader], [at]) = (shaders, code) else {
        return None;
    };
    if shader.stage != "compute" || at.is_null() || shader.size == 0 {
        return None;
    }
    // SAFETY: the title passes `size` bytes of shader code at `at` for the
    // call that creates the pipeline, which this runs within.
    let bytes = unsafe { std::slice::from_raw_parts(*at, shader.size) };
    if *blake3::hash(bytes).as_bytes() != shader.full_hash {
        tracing::warn!(hash = %shader.hash, "a BVH builder pipeline's bytecode is not its shader's: no forms");
        return None;
    }
    let group = group_size(bytes)?;
    let form = |thread_counts| {
        let own = Shader::own(shader.stage, bytes, crate::shader_cache::exact_options(thread_counts));
        Arc::new(Pending::new(vec![own], None, super::skip_hashes()))
    };
    Some(Forms {
        plain: form(false),
        threads: ThreadForm {
            pending: form(true),
            group,
        },
    })
}

/// A compute program's group size, from its container.
fn group_size(container: &[u8]) -> Option<[u32; 3]> {
    use victory_shader::native::StageIo;
    use victory_shader::{Container, FourCC};
    let c = Container::parse(container).ok()?;
    let part = c.part(FourCC::DXIL).or(c.part(FourCC::ACCC))?;
    victory_shader::native_programs(part.data).ok()?.into_iter().find_map(|p| match p.io {
        StageIo::Cs(cs) if cs.threads.iter().all(|&n| n > 0) => Some(cs.threads),
        _ => None,
    })
}

/// The builder's thread form `t` counts towards [`runs`] and is made at
/// once.
pub(crate) fn note_threads(t: &ThreadForm) {
    note_kind(&t.pending, Kind::Threads);
}

/// Why the builder stopped for good, once it did ([`trip`]).
static TRIPPED: OnceLock<String> = OnceLock::new();

/// The builder runs no more in this run: `why` (logged once). From the
/// next frame on; what this frame recorded of it stays.
pub(crate) fn trip(why: String) {
    if TRIPPED.set(why.clone()).is_ok() {
        tracing::warn!(why, "BVH builder held back for the rest of the run");
    }
}

/// The frame [`runs`] last decided for, and its decision.
static DECIDED: Mutex<(u64, bool)> = Mutex::new((u64::MAX, true));

/// Whether the builder's dispatches run in the current frame: decided once
/// per frame, so that a frame never runs part of the builder.
pub(crate) fn runs() -> bool {
    static SAID: AtomicBool = AtomicBool::new(false);
    let frame = crate::frame::current();
    let mut decided = DECIDED.lock().unwrap_or_else(|e| e.into_inner());
    if decided.0 == frame {
        return decided.1;
    }
    let all: Vec<(Option<bool>, Kind)> = {
        let mut pipelines = PIPELINES.lock().unwrap_or_else(|e| e.into_inner());
        pipelines.retain(|p| p.0.strong_count() > 0);
        pipelines
            .iter()
            .filter_map(|(p, kind)| Some((make::state(&*p.upgrade()?), *kind)))
            .collect()
    };
    let states: Vec<Option<bool>> = all
        .iter()
        .filter(|(_, k)| matches!(k, Kind::Plain { .. }))
        .map(|(s, _)| *s)
        .collect();
    let halves = crate::shader_cache::halves_local_here();
    let tripped = TRIPPED.get();
    let open = if on() {
        tripped.is_none() && opens_exact(&all)
    } else {
        opens(&states, halves)
    };
    if open != decided.1 || !SAID.swap(true, Ordering::Relaxed) {
        let (made, refused) = (
            states.iter().filter(|s| **s == Some(true)).count(),
            states.iter().filter(|s| **s == Some(false)).count(),
        );
        let threads: Vec<Option<bool>> = all
            .iter()
            .filter(|(_, k)| *k == Kind::Threads)
            .map(|(s, _)| *s)
            .collect();
        let (threads_made, threads_refused) = (
            threads.iter().filter(|s| **s == Some(true)).count(),
            threads.iter().filter(|s| **s == Some(false)).count(),
        );
        let shared = all
            .iter()
            .filter(|(_, k)| *k == Kind::Plain { exact: false })
            .count();
        tracing::info!(
            frame,
            pipelines = states.len(),
            made,
            refused,
            threads_made,
            threads_refused,
            "BVH builder {}",
            if open {
                "runs"
            } else if let Some(why) = tripped {
                why.as_str()
            } else if on() && shared > 0 {
                "held back: a pipeline of it has no exact form of its own (VICTORY_BVH_BUILDER=1)"
            } else if on() {
                "held back: not all its exact forms are made or refused (VICTORY_BVH_BUILDER=1)"
            } else if halves {
                "held back: its 64-lane programs would run as 32-lane halves on their own (VICTORY_HALVES_LOCAL, VICTORY_HALVES_SHARED_LDS)"
            } else {
                "held back: not all its pipelines are made"
            }
        );
        if on() && threads_refused > 0 {
            static NAMED: AtomicBool = AtomicBool::new(false);
            if !NAMED.swap(true, Ordering::Relaxed) {
                tracing::info!(
                    threads_refused,
                    "BVH builder: some of its programs have no form for dispatches counted in threads (see their `shader does not translate` lines); a DispatchX counted in threads with one stops the builder"
                );
            }
        }
    }
    *decided = (frame, open);
    open
}

/// Whether every pipeline is made (`Some(true)`); refused ones are
/// `Some(false)`, those still being made `None`.
fn all_made(states: &[Option<bool>]) -> bool {
    states.iter().all(|s| *s == Some(true))
}

/// Whether the builder runs: every pipeline made, and its 64-lane programs
/// not run as two 32-lane halves that each act as a wave (`halves`:
/// `VICTORY_HALVES_LOCAL=1` or `VICTORY_HALVES_SHARED_LDS=1` on a host of
/// 32-lane waves). The builder's
/// programs exchange across the halves; run that way they hung the GPU
/// within two minutes on an RTX 4070 SUPER (upstream PR #2711, from which
/// this rule is taken, keyed there on the host's wave size alone).
fn opens(states: &[Option<bool>], halves: bool) -> bool {
    !halves && all_made(states)
}

/// Under [`on`]: whether the builder runs, its pipelines' states and kinds
/// given. Every plain pipeline is an exact form of its own and made; every
/// thread form is made or refused (a refused one stops the builder only if
/// a dispatch counted in threads asks for it, [`trip`]). The halves
/// switches do not matter: no form has a half answering on its own.
fn opens_exact(all: &[(Option<bool>, Kind)]) -> bool {
    all.iter().all(|(state, kind)| match kind {
        Kind::Plain { exact } => *exact && *state == Some(true),
        Kind::Threads => state.is_some(),
    })
}

/// Tests: [`on`] whatever the variable says, while this lives.
#[cfg(test)]
static FORCED: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
pub(crate) struct ForcedOn;

#[cfg(test)]
impl ForcedOn {
    pub(crate) fn new() -> ForcedOn {
        FORCED.store(true, Ordering::Relaxed);
        ForcedOn
    }
}

#[cfg(test)]
impl Drop for ForcedOn {
    fn drop(&mut self) {
        FORCED.store(false, Ordering::Relaxed);
    }
}

/// Tests: [`runs`] decides again in this frame.
#[cfg(test)]
pub(crate) fn forget_decision() {
    *DECIDED.lock().unwrap_or_else(|e| e.into_inner()) = (u64::MAX, true);
}

/// Tests that set the builder's image, one at a time; each starts with
/// no decision made.
#[cfg(test)]
pub(crate) fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    *DECIDED.lock().unwrap_or_else(|e| e.into_inner()) = (u64::MAX, true);
    guard
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The builder runs only when every one of its pipelines is made: one
    /// refused or still being made holds all of them back.
    #[test]
    fn the_builder_runs_only_with_every_pipeline_made() {
        assert!(all_made(&[]), "no builder: nothing to hold back");
        assert!(all_made(&[Some(true), Some(true)]));
        assert!(!all_made(&[Some(true), Some(false)]), "one refused");
        assert!(!all_made(&[Some(true), None]), "one still being made");
        assert!(opens(&[Some(true)], false));
        assert!(!opens(&[Some(true), Some(false)], false));
        assert!(!opens(&[Some(true)], true), "64-lane programs run as halves");
        assert!(!opens(&[], true));
    }

    /// Under `VICTORY_BVH_BUILDER=1`: every plain pipeline must be an exact
    /// form of its own and made; thread forms must be made or refused.
    #[test]
    fn the_exact_builder_runs_with_every_form_settled() {
        let (exact, shared) = (Kind::Plain { exact: true }, Kind::Plain { exact: false });
        assert!(opens_exact(&[]));
        assert!(opens_exact(&[(Some(true), exact), (Some(true), Kind::Threads)]));
        assert!(opens_exact(&[(Some(true), exact), (Some(false), Kind::Threads)]), "a refused thread form");
        assert!(!opens_exact(&[(Some(true), exact), (None, Kind::Threads)]), "a thread form being made");
        assert!(!opens_exact(&[(None, exact)]), "a plain form being made");
        assert!(!opens_exact(&[(Some(false), exact)]), "a plain form refused");
        assert!(!opens_exact(&[(Some(true), shared)]), "a pipeline without its own exact form");
    }

    /// Pipelines are the builder's by their bytecode in its image, or by
    /// its code among the callers; not otherwise.
    #[test]
    fn builder_pipelines_are_known_by_their_origin() {
        let _one = test_lock();
        // A range that holds this test's own code (a caller of the check).
        let here = builder_pipelines_are_known_by_their_origin as *const () as usize;
        let elsewhere = [0x7_0000_0000usize as *const u8];
        *IMAGE.lock().unwrap() = None;
        assert!(!created_by_builder(&elsewhere), "no builder mapped");
        // Bytecode inside the image.
        *IMAGE.lock().unwrap() = Some((0x7_0000_0000, 0x7_0001_0000));
        assert!(created_by_builder(&elsewhere));
        assert!(!created_by_builder(&[0x8_0000_0000usize as *const u8]));
        // A caller inside the image: this function's own code.
        *IMAGE.lock().unwrap() = Some((here - 0x10_0000, here + 0x10_0000));
        assert!(created_by_builder(&[]));
        *IMAGE.lock().unwrap() = None;
    }
}
