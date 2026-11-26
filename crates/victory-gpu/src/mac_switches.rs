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
pub(crate) fn key(build: &[u8; 32], early_depth: bool) -> [u8; 32] {
    if !early_depth {
        return *build;
    }
    let mut h = blake3::Hasher::new();
    h.update(build);
    h.update(b"early depth");
    *h.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::{key, setting};

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
    fn early_depth_translations_have_cache_entries_of_their_own() {
        let build = [7u8; 32];
        // Off: the key is the one without the setting.
        assert_eq!(key(&build, false), build);
        let on = key(&build, true);
        assert_ne!(on, build);
        assert_eq!(on, key(&build, true));
        // Over another build (other settings, another translator): another key.
        assert_ne!(on, key(&[8u8; 32], true));
    }
}
