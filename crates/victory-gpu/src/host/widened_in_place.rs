//! mac: `VICTORY_WIDENED_IN_PLACE=1` (`crate::mac_switches::widened_in_place`,
//! from VIctory-macOS #520 4f298a3f): a buffer barrier takes the UAV writes
//! since the last one into the list's widened history without copying the
//! history. Without the switch, [`HostList::sync_widened`] makes a list of
//! the whole history and the new writes at every barrier that has widened
//! views to refresh (hackgods' world frames: about 90,000 writes, 2 MB, ~55
//! times a frame), and a list of the barrier's expanded writes each time.
//! Here the writes go into a buffer the list keeps (`widened_new`), and
//! `crate::widened::sync` gets the history and that buffer as two slices.
//! The same writes reach it in the same order, and a heap's ranges are taken
//! as without the switch (`VICTORY_CENSUS_FAST`'s rules included).

use std::sync::atomic::Ordering;

use windows::core::Interface;

use super::{HostList, WidenedEvent};

/// Room a list's widened buffers keep at their close at least, in writes
/// (24 bytes each), so that small lists never give theirs back.
const KEEP_MIN: usize = 1 << 12;

/// Empties a widened buffer at its list's close, keeping the room for the
/// next use (the world's lists need about the same each frame) but not more
/// than twice what this use needed, `used`: a peak is not kept for good.
fn empty_kept<T>(v: &mut Vec<T>, used: usize) {
    v.clear();
    let keep = used.max(KEEP_MIN).next_power_of_two();
    if v.capacity() > 2 * keep {
        v.shrink_to(keep);
    }
}

impl HostList {
    /// [`HostList::sync_widened`] with the switch.
    pub(crate) fn sync_widened_in_place(&self) {
        let mut pending = self
            .widened_pending
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut new = self.widened_new.lock().unwrap_or_else(|e| e.into_inner());
        let new = &mut *new;
        // As without the switch: with `VICTORY_CENSUS_FAST=1` a heap's
        // ranges once a barrier, where its last draw stood.
        let last = crate::mac_switches::census_fast().then(|| {
            crate::census_fast::last_of_each(pending.iter().map(|event| match event {
                WidenedEvent::Heap(heap) => Some(std::sync::Arc::as_ptr(heap) as usize),
                WidenedEvent::Write(_) => None,
            }))
        });
        for (k, event) in pending.drain(..).enumerate() {
            match event {
                WidenedEvent::Write(write) => new.push(write),
                WidenedEvent::Heap(_) if last.as_ref().is_some_and(|last| !last[k]) => {}
                WidenedEvent::Heap(heap) => match crate::bindless::unordered_writes(&heap) {
                    Some(writes) => new.extend(writes.iter().cloned()),
                    None => crate::bindless::widened_writes_into(&heap, new),
                },
            }
        }
        if !crate::shadow::WIDENED_ANY.load(Ordering::Acquire) {
            self.widened_writes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .append(new);
            return;
        }
        let mut ready = self.widened_ready.lock().unwrap_or_else(|e| e.into_inner());
        if new.is_empty()
            && crate::shadow::widened_all()
                .iter()
                .all(|view| ready.contains(&(view.buffer.as_raw() as usize)))
        {
            return;
        }
        let mut history = self
            .widened_writes
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if history.is_empty() && new.is_empty() {
            return;
        }
        crate::widened::count_history(true, history.len() + new.len());
        // Writes a failed widening leaves stay for the next barrier.
        let Some(views) = crate::widened::sync(self, &history, new, &ready) else {
            return;
        };
        history.append(new);
        ready.extend(views);
    }

    /// Empties the widened history and new writes at the list's close,
    /// keeping their room (with the switch).
    pub(super) fn empty_widened(&self) {
        let mut history = self
            .widened_writes
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let used = history.len();
        empty_kept(&mut history, used);
        drop(history);
        empty_kept(
            &mut self.widened_new.lock().unwrap_or_else(|e| e.into_inner()),
            used,
        );
    }

    /// The widened history (address, bytes per write), where the history
    /// and the buffer of writes not widened yet are, and whether anything
    /// waits for a barrier (tests).
    #[cfg(test)]
    pub(crate) fn widened_state(&self) -> (Vec<(usize, usize)>, *const u8, *const u8, bool) {
        let history = self
            .widened_writes
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let pending = self
            .widened_pending
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let new = self.widened_new.lock().unwrap_or_else(|e| e.into_inner());
        (
            history.iter().map(|&(_, at, len)| (at, len)).collect(),
            history.as_ptr().cast(),
            new.as_ptr().cast(),
            !pending.is_empty() || !new.is_empty(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{KEEP_MIN, empty_kept};

    /// A list's buffer keeps its room up to twice what a use needed, and
    /// never less than the floor.
    #[test]
    fn widened_buffers_keep_their_room_but_not_a_peak() {
        let mut v: Vec<u64> = Vec::with_capacity(100_000);
        v.extend(0..90_000);
        empty_kept(&mut v, 90_000);
        assert!(
            v.is_empty() && v.capacity() >= 90_000,
            "the room a world list needs stays"
        );
        empty_kept(&mut v, 10);
        assert!(
            v.capacity() >= KEEP_MIN && v.capacity() <= 2 * KEEP_MIN,
            "{}",
            v.capacity()
        );
    }
}
