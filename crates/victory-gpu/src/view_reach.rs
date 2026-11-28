//! How far into their host buffers the buffer views made for the title start
//! (a counter, always on: a few atomic adds per view made).
//!
//! Under D3DMetal (M5 Pro, the in-place work's view probe) a shader reads
//! zeros through a buffer view whose first byte lies 2 GiB or more into its
//! buffer (1 GiB in reads), and through any view of a reserved buffer of
//! 8 GiB or more, where copies read the right bytes. Host copies of title
//! buffers are as long as the buffers, and an XMemVirtualAlloc pool aliased
//! zero-copy (`crate::alias`) is one buffer of the pool's length (2.15 GiB
//! for one of Grand Theft Auto VI's), so views of the title's may start
//! that deep. Root views start at their copy's first byte and add the root
//! address's offset in the shader; a raw view holds at most 2^27 elements
//! (512 MiB), so a root view into a copy longer than that is cut short.
//! The report line says whether any of this happens.

use std::sync::atomic::{AtomicU64, Ordering};

/// What made the view.
#[derive(Clone, Copy)]
pub(crate) enum Kind {
    /// A view in a descriptor table the title wrote (`crate::descriptor`).
    Table = 0,
    /// A root view: the offset its shader adds (`crate::draw`).
    Root = 1,
}

const GIB: u64 = 1 << 30;

/// Per kind, per buffer (0 a copy, 1 a pool aliased zero-copy): views
/// starting 1 GiB or more in, and 2 GiB or more in.
static DEEP: [[[AtomicU64; 2]; 2]; 2] =
    [const { [const { [const { AtomicU64::new(0) }; 2] }; 2] }; 2];
/// Per kind: views counted, and the deepest start seen.
static SEEN: [[AtomicU64; 2]; 2] = [const { [const { AtomicU64::new(0) }; 2] }; 2];
/// Root view slots whose raw view was cut at 2^27 elements, per buffer kind.
static CLAMPED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A view of `kind` starting `offset` bytes into its host buffer, a pool
/// aliased zero-copy or not (`aliased`).
pub(crate) fn note(kind: Kind, offset: u64, aliased: bool) {
    let k = kind as usize;
    SEEN[k][0].fetch_add(1, Ordering::Relaxed);
    if offset < GIB {
        return;
    }
    SEEN[k][1].fetch_max(offset, Ordering::Relaxed);
    let a = usize::from(aliased);
    DEEP[k][a][0].fetch_add(1, Ordering::Relaxed);
    if offset >= 2 * GIB {
        DEEP[k][a][1].fetch_add(1, Ordering::Relaxed);
    }
}

/// A root view slot made over a copy of `bytes` bytes (`aliased`: a pool).
pub(crate) fn note_root_view(bytes: u64, aliased: bool) {
    if bytes / 4 > 1 << 27 {
        CLAMPED[usize::from(aliased)].fetch_add(1, Ordering::Relaxed);
    }
}

/// One line per report, counts since the start.
pub(crate) fn log() {
    let d = |k: usize, a: usize, i: usize| DEEP[k][a][i].load(Ordering::Relaxed);
    let mb = |k: usize| SEEN[k][1].load(Ordering::Relaxed) >> 20;
    tracing::info!(
        "buffer views deep into their host buffer, since the start: table views {} made, \
         >=1 GiB in {} on copies + {} on aliased pools, >=2 GiB in {} + {}, deepest {} MB; \
         root views {} bound, offset >=1 GiB {} + {}, >=2 GiB {} + {}, deepest {} MB; \
         root view slots cut at 2^27 elements (copy over 512 MiB) {} + {}",
        SEEN[0][0].load(Ordering::Relaxed),
        d(0, 0, 0),
        d(0, 1, 0),
        d(0, 0, 1),
        d(0, 1, 1),
        mb(0),
        SEEN[1][0].load(Ordering::Relaxed),
        d(1, 0, 0),
        d(1, 1, 0),
        d(1, 0, 1),
        d(1, 1, 1),
        mb(1),
        CLAMPED[0].load(Ordering::Relaxed),
        CLAMPED[1].load(Ordering::Relaxed),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn views_are_counted_by_how_deep_they_start() {
        let before = [0, 1].map(|a| [0, 1].map(|i| DEEP[0][a][i].load(Ordering::Relaxed)));
        note(Kind::Table, GIB - 4, false);
        note(Kind::Table, GIB, false);
        note(Kind::Table, 2 * GIB + 64, true);
        let after = [0, 1].map(|a| [0, 1].map(|i| DEEP[0][a][i].load(Ordering::Relaxed)));
        // Other tests may count views meanwhile: at least ours are there.
        assert!(after[0][0] > before[0][0], "a copy's view 1 GiB in");
        assert!(
            after[1][0] > before[1][0] && after[1][1] > before[1][1],
            "a pool's 2 GiB in"
        );
        assert!(SEEN[0][1].load(Ordering::Relaxed) >= 2 * GIB + 64);
        let clamped = CLAMPED[1].load(Ordering::Relaxed);
        note_root_view(512 << 20, true);
        note_root_view((512 << 20) + 4, true);
        assert!(CLAMPED[1].load(Ordering::Relaxed) > clamped);
    }
}
