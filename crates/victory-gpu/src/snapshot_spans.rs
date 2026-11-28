//! From VIctory-macOS #436 (82587a26, Yumaste's Marvisak/VIctory#3211 as
//! ported by Michael McCrimmons), behind `VICTORY_SNAPSHOT_SPANS=1`: a
//! write snapshot that saves only the bytes each declared GPU write newly
//! covers. Spans keep their title-address order; the bytes behind them stay
//! in capture order in one pooled buffer. Off, `Shadow::writing` keeps
//! merging overlapping declarations into one segment, copying the old
//! segments' bytes again each time the covered range grows.
//!
//! mac: the 64 KiB pieces already compared stay in the shadow's snapshot
//! beside the segments (shared by both forms), not in here; and the module
//! sits outside `shadow` so that its tests run on any host.

#[cfg(windows)]
use crate::shadow::{give_back, pooled_preferring};

/// Off Windows (native tests) there is no snapshot pool.
#[cfg(not(windows))]
fn pooled_preferring(_: usize, _: usize) -> Option<Vec<u8>> {
    None
}

#[cfg(not(windows))]
fn give_back(_: Vec<u8>) {}

#[derive(Default)]
pub(crate) struct Spans {
    bytes: Vec<u8>,
    /// The last captured size guides the first pool checkout next submission.
    last_size: usize,
    /// Disjoint spans in address order, pointing into the capture buffer.
    spans: Vec<Span>,
}

struct Span {
    start: usize,
    end: usize,
    offset: usize,
}

impl Spans {
    /// No bytes captured (none since the last [`Spans::clear`]).
    pub(crate) fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// The bytes held (the shadow's tests, which run under Wine).
    #[cfg(all(test, windows))]
    pub(crate) fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Keeps the newly covered bytes of `[start, start + host.len())`, without
    /// rebuilding the ranges captured before. Returns the bytes captured and
    /// the bytes moved when the backing buffer grew.
    pub(crate) fn capture(&mut self, start: usize, host: &[u8]) -> (usize, usize) {
        let end = start + host.len();
        let before = self.bytes.len();
        let mut at = start;
        let mut i = self.spans.partition_point(|span| span.end <= at);
        let mut moved = 0;
        if host.len() > self.bytes.capacity() - before {
            let covered: usize = self.spans[i..]
                .iter()
                .take_while(|span| span.start < end)
                .map(|span| span.end.min(end) - span.start.max(start))
                .sum();
            moved = self.reserve(before + (host.len() - covered));
        }
        while at < end {
            if let Some(span) = self.spans.get(i)
                && span.start <= at
            {
                at = span.end;
                i += 1;
                continue;
            }
            let next = self.spans.get(i).map_or(end, |span| end.min(span.start));
            let offset = self.bytes.len();
            self.bytes.extend_from_slice(&host[at - start..next - start]);
            // Consecutive captures in address order can still use one span.
            if let Some(previous) = i.checked_sub(1).and_then(|j| self.spans.get_mut(j))
                && previous.end == at
                && previous.offset + (previous.end - previous.start) == offset
            {
                previous.end = next;
            } else {
                self.spans.insert(i, Span { start: at, end: next, offset });
                i += 1;
            }
            at = next;
        }
        (self.bytes.len() - before, moved)
    }

    /// Room for `required` bytes; returns the bytes moved.
    fn reserve(&mut self, required: usize) -> usize {
        let capacity = self.bytes.capacity();
        if required <= capacity {
            return 0;
        }
        // Grow by enough to amortize copying, but use a smaller warm
        // buffer if it can hold the new snapshot.
        let growth = required.max(capacity.saturating_mul(2));
        let preferred = if capacity == 0 { growth.max(self.last_size) } else { growth };
        let mut grown = pooled_preferring(required, preferred).unwrap_or_else(|| Vec::with_capacity(growth));
        grown.extend_from_slice(&self.bytes);
        let old = std::mem::replace(&mut self.bytes, grown);
        let moved = old.len();
        if capacity != 0 {
            give_back(old);
        }
        moved
    }

    /// Whether every byte of `[start, end)` is captured.
    pub(crate) fn covers(&self, start: usize, end: usize) -> bool {
        let mut at = start;
        for span in &self.spans[self.spans.partition_point(|span| span.end <= start)..] {
            if span.start > at {
                break;
            }
            at = span.end;
            if at >= end {
                return true;
            }
        }
        false
    }

    /// The original bytes of a piece inside a range [`Spans::covers`] holds,
    /// if they differ from `host` (none: the piece is unchanged). Fragmented
    /// bytes are gathered into `scratch` only if they differ.
    pub(crate) fn changed_piece<'a>(&'a self, start: usize, host: &[u8], scratch: &'a mut Vec<u8>) -> Option<&'a [u8]> {
        let end = start + host.len();
        debug_assert!(self.covers(start, end));
        let first = self.spans.partition_point(|span| span.end <= start);
        let span = &self.spans[first];
        if end <= span.end {
            let offset = span.offset + (start - span.start);
            let old = &self.bytes[offset..offset + host.len()];
            return (host != old).then_some(old);
        }
        let fragments = || {
            self.spans[first..].iter().take_while(|span| span.start < end).map(|span| {
                let from = span.start.max(start);
                let to = span.end.min(end);
                let offset = span.offset + (from - span.start);
                (&self.bytes[offset..offset + to - from], from - start..to - start)
            })
        };
        if fragments().all(|(old, range)| old == &host[range]) {
            return None;
        }
        scratch.clear();
        for (old, _) in fragments() {
            scratch.extend_from_slice(old);
        }
        Some(scratch)
    }

    /// A refresh before the GPU runs changes the baseline of the captured
    /// bytes of `[start, start + host.len())`.
    pub(crate) fn refresh(&mut self, start: usize, host: &[u8]) {
        let end = start + host.len();
        let first = self.spans.partition_point(|span| span.end <= start);
        for span in self.spans[first..].iter().take_while(|span| span.start < end) {
            let from = span.start.max(start);
            let to = span.end.min(end);
            let offset = span.offset + (from - span.start);
            self.bytes[offset..offset + to - from].copy_from_slice(&host[from - start..to - start]);
        }
    }

    /// Forgets the spans; the buffer goes back to the pool.
    pub(crate) fn clear(&mut self) {
        self.spans.clear();
        if !self.bytes.is_empty() {
            self.last_size = self.bytes.len();
            give_back(std::mem::take(&mut self.bytes));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Spans;

    #[test]
    fn overlapping_snapshots_keep_old_bytes_until_a_refresh() {
        let mut snapshot = Spans { bytes: Vec::with_capacity(7), ..Spans::default() };
        let mut scratch = Vec::new();
        assert_eq!(snapshot.capture(4, &[14, 15]), (2, 0));
        let data = snapshot.bytes.as_ptr();
        assert_eq!(snapshot.capture(4, &[0, 0]), (0, 0));
        assert_eq!(snapshot.bytes.as_ptr(), data, "covered bytes need no growth");
        assert_eq!(snapshot.capture(0, &[10, 11]), (2, 0));
        assert!(!snapshot.covers(0, 6), "there is a gap");
        assert_eq!(snapshot.capture(1, &[21, 12, 13, 24, 25, 16]), (3, 0));
        assert_eq!(snapshot.capture(0, &[0; 7]), (0, 0));
        assert!(snapshot.covers(0, 7));
        assert!(!snapshot.covers(0, 8));
        let original = [10, 11, 12, 13, 14, 15, 16];
        assert!(snapshot.changed_piece(0, &original, &mut scratch).is_none());
        assert!(scratch.is_empty());
        assert_eq!(snapshot.changed_piece(0, &[0; 7], &mut scratch).unwrap(), original);
        snapshot.refresh(1, &[21, 22, 23, 24, 25]);
        assert_eq!(snapshot.changed_piece(0, &[0; 7], &mut scratch).unwrap(), [10, 21, 22, 23, 24, 25, 16]);
        assert_eq!(snapshot.changed_piece(5, &[0], &mut scratch).unwrap(), [25]);
        snapshot.clear();
        assert!(snapshot.is_empty());
        assert!(!snapshot.covers(0, 1));
    }

    #[test]
    fn snapshot_traffic_counts_moving_the_saved_bytes_only_on_growth() {
        let mut snapshot = Spans { bytes: Vec::with_capacity(2), ..Spans::default() };
        assert_eq!(snapshot.capture(0, &[1, 2]), (2, 0));
        assert_eq!(snapshot.capture(1, &[9, 3, 4]), (2, 2), "two new, two moved");
        assert_eq!(snapshot.capture(0, &[9; 4]), (0, 0));
        let mut scratch = Vec::new();
        assert!(snapshot.changed_piece(0, &[1, 2, 3, 4], &mut scratch).is_none());
    }
}
