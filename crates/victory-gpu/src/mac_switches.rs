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

/// [`lock_writers`]' rule: helper lanes, low keys.
fn writers(value: Option<&str>, default: bool) -> [bool; 2] {
    match value {
        Some("helpers") => [true, false],
        value => [setting(value, default); 2],
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
pub(crate) fn key(build: &[u8; 32], early_depth: bool, lock_writers: [bool; 2]) -> [u8; 32] {
    if !early_depth && lock_writers == [false; 2] {
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
    *h.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::{key, setting, writers};

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
    /// lock settings came.
    #[test]
    fn each_combination_of_settings_has_cache_entries_of_its_own() {
        let build = [7u8; 32];
        assert_eq!(key(&build, false, [false, false]), build);
        let mut keys = std::collections::HashSet::new();
        for n in 0..8u8 {
            let k = key(&build, n & 1 != 0, [n & 2 != 0, n & 4 != 0]);
            assert_eq!(k, key(&build, n & 1 != 0, [n & 2 != 0, n & 4 != 0]));
            keys.insert(k);
            // Over another build (other settings, another translator).
            assert!(n == 0 || k != key(&[8u8; 32], n & 1 != 0, [n & 2 != 0, n & 4 != 0]));
        }
        assert_eq!(keys.len(), 8);
        let mut early = blake3::Hasher::new();
        early.update(&build);
        early.update(b"early depth");
        assert_eq!(
            key(&build, true, [false, false]),
            *early.finalize().as_bytes()
        );
    }
}
