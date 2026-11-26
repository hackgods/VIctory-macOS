//! The Mac line's translation switches (`victory_shader::emit::Options`,
//! `lower::mac` there): read once from the environment, and part of the
//! shader cache key ([`key`], used by `shader_cache`), so translations made
//! with and without one never mix. The rules are here, outside the
//! Windows-only shader cache, so that their tests run on any host.

#![cfg_attr(not(windows), allow(dead_code))]

use std::sync::OnceLock;

/// Whether pixel programs that ask for the depth test before the shader
/// get DXIL's early depth-stencil flag
/// (`victory_shader::emit::Options::early_depth`): by default under Wine
/// (`wine`), where the title's see-through pre-pass is the cost of a frame
/// and hidden fragments ran it. `VICTORY_EARLY_DEPTH=0` or `1` overrides.
pub(crate) fn early_depth(wine: bool) -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = setting(std::env::var("VICTORY_EARLY_DEPTH").ok().as_deref(), wine);
        tracing::info!(
            on,
            "VICTORY_EARLY_DEPTH: pixel programs that ask for the depth test first get the early depth-stencil flag"
        );
        on
    })
}

/// Which lanes stay out of the lock loop of a 64-bit image atomic
/// (`victory_shader::emit::Options::atomic_lock_helpers` and
/// `atomic_lock_keys`; they act only where atomic locks are on): by default
/// under Wine (`wine`) both the helper lanes of pixel programs and the lanes
/// whose `umax` key is below the texel's. `VICTORY_ATOMIC_LOCK_WRITERS=0`:
/// every lane takes the lock, as before; `helpers`: only helper lanes stay
/// out; `1`: both, anywhere.
pub(crate) fn lock_writers(wine: bool) -> [bool; 2] {
    static ON: OnceLock<[bool; 2]> = OnceLock::new();
    *ON.get_or_init(|| {
        let value = std::env::var("VICTORY_ATOMIC_LOCK_WRITERS").ok();
        let [helpers, keys] = writers(value.as_deref(), wine);
        tracing::info!(
            helpers,
            keys,
            "VICTORY_ATOMIC_LOCK_WRITERS: lanes that cannot change the texel stay out of the lock loop of 64-bit image atomics"
        );
        [helpers, keys]
    })
}

/// Which parts of the quicker heap sync are on (`crate::descriptor::sync_heaps`
/// makes 25,000 to 40,000 host descriptor slots again in a world frame, on
/// the submitting thread): the heap census in hash maps
/// (`crate::census_maps`), and buffer shader views made from one lookup
/// (`crate::descriptor::host`, "quick"). By default both under Wine (`wine`
/// says whether; asked once), where a frame's CPU part is the cost.
/// `VICTORY_QUICK_SYNC=0`: neither, as before; `maps` or `views`: that part
/// alone; `1`: both, anywhere. Neither changes what a slot shows.
pub(crate) fn quick_sync(wine: impl FnOnce() -> bool) -> [bool; 2] {
    static ON: OnceLock<[bool; 2]> = OnceLock::new();
    *ON.get_or_init(|| {
        let value = std::env::var("VICTORY_QUICK_SYNC").ok();
        let [maps, views] = quick(value.as_deref(), wine());
        tracing::info!(
            maps,
            views,
            "VICTORY_QUICK_SYNC: the heap census in hash maps, buffer shader views of a heap sync made from one lookup"
        );
        [maps, views]
    })
}

/// [`quick_sync`]'s rule: census maps, views.
fn quick(value: Option<&str>, default: bool) -> [bool; 2] {
    match value {
        Some("maps") => [true, false],
        Some("views") => [false, true],
        value => [setting(value, default); 2],
    }
}

/// Whether a compute group of one console wave that shares LDS in a way
/// barriers at its common blocks cannot order by themselves gets the finer
/// order of `victory_shader::emit::lower::lds_order`
/// (`victory_shader::emit::Options::lds_order`; `notes/cross-half-programs.md`,
/// "Shared LDS"): only with `VICTORY_HALVES_LDS_ORDER=1`. Exact where it
/// applies; a program it cannot order stays refused, or runs unsynced with
/// `VICTORY_HALVES_SHARED_LDS=1`.
pub(crate) fn lds_order() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = setting(std::env::var("VICTORY_HALVES_LDS_ORDER").ok().as_deref(), false);
        if on {
            tracing::info!(
                "VICTORY_HALVES_LDS_ORDER=1: one-wave64 groups that share LDS get group barriers where every lane runs, when that orders the LDS exactly"
            );
        }
        on
    })
}

/// Whether a lane read across the two 32-lane halves of a console wave64,
/// in a block every lane runs once, is handed over through group-shared
/// memory at a group barrier (`victory_shader::emit::lower::exchange`,
/// `victory_shader::emit::Options::halves_exchange`;
/// `notes/ray-tracing-plan.md`): only with `VICTORY_HALVES_EXCHANGE=1`.
/// Exact where it applies; a read it does not cover stays refused, or is
/// answered by the reader's half with `VICTORY_HALVES_LOCAL=1`.
pub(crate) fn exchange() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = setting(std::env::var("VICTORY_HALVES_EXCHANGE").ok().as_deref(), false);
        if on {
            tracing::info!(
                "VICTORY_HALVES_EXCHANGE=1: lane reads across 32-lane halves are handed over through group-shared memory where every lane runs"
            );
        }
        on
    })
}

/// [`lock_writers`]' rule: helper lanes, low keys.
fn writers(value: Option<&str>, default: bool) -> [bool; 2] {
    match value {
        Some("helpers") => [true, false],
        value => [setting(value, default); 2],
    }
}

/// `VICTORY_ASYNC_HEAP_SYNC`'s rule (`crate::descriptor::ahead`): off when
/// unset, empty or `0`; else on, and the milliseconds between a round of
/// the heap sync and the next while the GPU still runs: 8 for `1` (or
/// anything that is no number), a larger number for itself.
pub(crate) fn heap_sync_pause(value: Option<&str>) -> Option<u64> {
    match value {
        None | Some("" | "0") => None,
        Some(v) => Some(v.parse::<u64>().ok().filter(|&ms| ms > 1).unwrap_or(8)),
    }
}

/// A switch's state: `0` off, `1` on, anything else (and unset) `default`.
fn setting(value: Option<&str>, default: bool) -> bool {
    match value {
        Some("0") => false,
        Some("1") => true,
        _ => default,
    }
}

/// The cache key `build` for translations made with these settings. A
/// setting that is off adds nothing, so the entries made without it stay
/// the ones read.
pub(crate) fn key(
    build: &[u8; 32],
    early_depth: bool,
    lock_writers: [bool; 2],
    lds_order: bool,
    exchange: bool,
) -> [u8; 32] {
    if !early_depth && lock_writers == [false; 2] && !lds_order && !exchange {
        return *build;
    }
    let mut h = blake3::Hasher::new();
    h.update(build);
    if early_depth {
        h.update(b"early depth");
    }
    if lock_writers[0] {
        h.update(b"atomic lock helpers");
    }
    if lock_writers[1] {
        h.update(b"atomic lock keys");
    }
    if lds_order {
        h.update(b"lds order");
    }
    if exchange {
        h.update(b"halves exchange");
    }
    *h.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::{heap_sync_pause, key, quick, setting, writers};

    #[test]
    fn the_heap_sync_ahead_is_off_unless_asked_and_takes_its_pause() {
        assert_eq!(heap_sync_pause(None), None);
        assert_eq!(heap_sync_pause(Some("")), None);
        assert_eq!(heap_sync_pause(Some("0")), None);
        assert_eq!(heap_sync_pause(Some("1")), Some(8));
        assert_eq!(heap_sync_pause(Some("yes")), Some(8));
        assert_eq!(heap_sync_pause(Some("4")), Some(4));
        assert_eq!(heap_sync_pause(Some("1000")), Some(1000));
    }

    #[test]
    fn the_quick_sync_switch_names_its_parts() {
        for default in [false, true] {
            assert_eq!(quick(Some("0"), default), [false, false]);
            assert_eq!(quick(Some("maps"), default), [true, false]);
            assert_eq!(quick(Some("views"), default), [false, true]);
            assert_eq!(quick(Some("1"), default), [true, true]);
            assert_eq!(quick(None, default), [default, default]);
            assert_eq!(quick(Some("both"), default), [default, default]);
        }
    }

    #[test]
    fn a_switch_is_off_at_0_on_at_1_and_else_the_default() {
        for default in [false, true] {
            assert!(!setting(Some("0"), default));
            assert!(setting(Some("1"), default));
            assert_eq!(setting(None, default), default);
            assert_eq!(setting(Some(""), default), default);
            assert_eq!(setting(Some("yes"), default), default);
        }
    }

    #[test]
    fn the_lock_writers_switch_has_three_states() {
        for default in [false, true] {
            assert_eq!(writers(Some("0"), default), [false, false]);
            assert_eq!(writers(Some("helpers"), default), [true, false]);
            assert_eq!(writers(Some("1"), default), [true, true]);
            assert_eq!(writers(None, default), [default, default]);
            assert_eq!(writers(Some("keys"), default), [default, default]);
        }
    }

    /// Every combination of the settings is a key of its own; all off is
    /// the key without them; early depth alone is what it was before the
    /// lock settings came, the three of them without the LDS order what
    /// they were before it came, and all four without the exchange what
    /// they were before that came.
    #[test]
    fn each_combination_of_settings_has_cache_entries_of_its_own() {
        let build = [7u8; 32];
        assert_eq!(key(&build, false, [false, false], false, false), build);
        let of = |build: &[u8; 32], n: u8| {
            key(
                build,
                n & 1 != 0,
                [n & 2 != 0, n & 4 != 0],
                n & 8 != 0,
                n & 16 != 0,
            )
        };
        let mut keys = std::collections::HashSet::new();
        for n in 0..32u8 {
            let k = of(&build, n);
            assert_eq!(k, of(&build, n));
            keys.insert(k);
            // Over another build (other settings, another translator).
            assert!(n == 0 || k != of(&[8u8; 32], n));
        }
        assert_eq!(keys.len(), 32);
        let mut early = blake3::Hasher::new();
        early.update(&build);
        early.update(b"early depth");
        assert_eq!(
            key(&build, true, [false, false], false, false),
            *early.finalize().as_bytes()
        );
        let mut four = blake3::Hasher::new();
        four.update(&build);
        four.update(b"early depth");
        four.update(b"atomic lock helpers");
        four.update(b"atomic lock keys");
        four.update(b"lds order");
        assert_eq!(of(&build, 15), *four.finalize().as_bytes());
        let mut three = blake3::Hasher::new();
        three.update(&build);
        three.update(b"early depth");
        three.update(b"atomic lock helpers");
        three.update(b"atomic lock keys");
        assert_eq!(of(&build, 7), *three.finalize().as_bytes());
    }
}
