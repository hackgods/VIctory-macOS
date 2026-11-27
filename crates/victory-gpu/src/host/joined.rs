//! mac: from VIctory-macOS #419 e678253e (`VICTORY_COALESCE_PIECES=1`):
//! the host's side of pieces joined across queues (`crate::hold::joined`).
//! What a list's bottom-of-pipe writes leave where a wait reads, what it
//! does on the CPU, the barrier lists put before a segment that joined
//! through the piece's writes or Signals, and the frame line's part.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use super::*;

/// A closed list holding one global UAV barrier, put before a segment that
/// joined its piece through another segment's bottom-of-pipe write or
/// Signal: the shaders before it have written what it reads. Reused: one
/// submission is on the host GPU at a time.
pub(super) struct BarrierList {
    list: ID3D12GraphicsCommandList,
    _allocator: ID3D12CommandAllocator,
}

impl BarrierList {
    fn new(host: &Host) -> windows::core::Result<Self> {
        // SAFETY: plain object creation; the list records one barrier.
        unsafe {
            let allocator: ID3D12CommandAllocator = host.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList =
                host.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
            list.ResourceBarrier(&[uav_barrier()]);
            list.Close()?;
            name(&list, || "joined piece UAV barrier".into());
            Ok(Self { list, _allocator: allocator })
        }
    }
}

impl Host {
    /// The barrier lists for `needed` barriers, made on first use (as many
    /// as the most a submission asked for). Under the submission lock.
    /// `None` (with a warning, once) if they cannot be made: the piece then
    /// runs without them.
    pub(super) fn joined_barriers(&self, needed: usize) -> Vec<Option<ID3D12CommandList>> {
        if needed == 0 {
            return Vec::new();
        }
        let mut pool = self.joined_barriers.lock().unwrap_or_else(|e| e.into_inner());
        while pool.len() < needed {
            match BarrierList::new(self) {
                Ok(made) => pool.push(made),
                Err(error) => {
                    static SAID: std::sync::Once = std::sync::Once::new();
                    SAID.call_once(|| tracing::warn!(%error, "no UAV barrier list for joined pieces"));
                    break;
                }
            }
        }
        pool.iter().take(needed).map(|b| b.list.cast().ok()).collect()
    }
}

impl HostList {
    /// What its bottom-of-pipe writes leave where `wait` reads, by the last
    /// of them that reaches its bytes: `None` if none does; whether the wait
    /// holds if that one stores all its bytes as a known value (`None` in
    /// it for a timestamp, part of the bytes, or a comparison we do not
    /// know).
    pub(crate) fn leaves(&self, wait: &crate::hold::ValueWait) -> Option<Option<bool>> {
        let d = self.deferred();
        let &(at, what, bytes) = d.writes.iter().rev().find(|&&(at, _, n)| wait.reads(at as u64, n as u64))?;
        Some(match what {
            crate::queue::Written::Value(v) if at as u64 == wait.address && bytes >= usize::from(wait.bytes) => {
                wait.holds_for(v)
            }
            _ => None,
        })
    }

    /// What it does on the CPU when its submission is prepared and
    /// finished, for joining (`crate::hold::joined::may_join`).
    pub(crate) fn join_flags(&self) -> crate::hold::joined::JoinFlags {
        let d = self.deferred();
        let mut f = crate::hold::joined::JoinFlags {
            after: !d.after.is_empty(),
            lands: !d.lands.is_empty() || !d.texture_backs.is_empty(),
            ..Default::default()
        };
        for step in &d.copies {
            match step {
                Step::Copy(..) | Step::Move(..) => {
                    f.title_copy = true;
                    f.cpu_read = true;
                }
                Step::Fill(..) => f.cpu_read = true,
                &Step::Own(_, _, bytes) => f.cpu_read |= bytes > 0,
                Step::GpuWrite(..) => f.gpu_write = true,
                Step::Texture(..) | Step::Heap(_) => f.uploads = true,
                Step::Refresh(..) | Step::Slot(..) | Step::Host(_) | Step::Memory(_) => {}
            }
        }
        f
    }
}

/// A joined piece as the host submissions it runs in ([`copy_order_parts`]),
/// each with its segments' barriers (none before a part's first: the part
/// before has run and been finished).
pub(crate) fn copy_order_parts(
    segments: Vec<(std::sync::Arc<HostList>, bool)>,
) -> Vec<(Vec<std::sync::Arc<HostList>>, Vec<bool>)> {
    let (lists, barriers): (Vec<_>, Vec<bool>) = segments.into_iter().unzip();
    let mut at = 0;
    super::copy_order_parts(lists)
        .into_iter()
        .map(|part| {
            let mut b = barriers[at..at + part.len()].to_vec();
            at += part.len();
            if let Some(first) = b.first_mut() {
                *first = false;
            }
            (part, b)
        })
        .collect()
}

/// Since the last frame line: pieces the queues gave and their segments;
/// with the switch, segments that joined another queue's piece, waits met
/// by the piece's writes, fence waits met by its Signals, pieces given
/// early, barrier lists run; host submissions the pieces ran in.
static JOINED: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// Counts a pump (`crate::queue`).
pub(crate) fn count(p: &crate::hold::Pumped) {
    for (i, n) in [p.pieces, p.segments, p.crossed, p.met_by_piece, p.fences_by_piece, p.splits].into_iter().enumerate() {
        JOINED[i].fetch_add(u64::from(n), Relaxed);
    }
}

/// Counts what a piece ran in: host submissions, barrier lists.
pub(crate) fn count_run(submissions: usize, barriers: usize) {
    JOINED[7].fetch_add(submissions as u64, Relaxed);
    JOINED[6].fetch_add(barriers as u64, Relaxed);
}

/// The frame line's part for the queues' pieces (`frames` since the last
/// one), always: with `VICTORY_COALESCE_PIECES=1` also what joining did.
pub(super) fn frame_line(frames: f64) -> String {
    let [pieces, segments, crossed, met, fences, splits, barriers, submissions] =
        JOINED.each_ref().map(|n| n.swap(0, Relaxed) as f64 / frames.max(1.0));
    let mut line = format!(
        "; queue pieces: {pieces:.1}/frame of {segments:.1} segments, in {submissions:.1} host submissions"
    );
    if crate::hold::joined::on() {
        line += &format!(
            " (VICTORY_COALESCE_PIECES: {crossed:.1} segments joined another queue's piece, {met:.1} waits met by the \
             piece's writes and {fences:.1} fence waits by its Signals, {barriers:.1} barriers; {splits:.1} pieces \
             given early for the order of their CPU work or the cap of {})",
            crate::hold::joined::cap()
        );
    }
    line
}

#[cfg(test)]
mod tests {
    use crate::hold::ValueWait;
    use crate::hold::joined::JoinFlags;
    use crate::queue::Written;

    /// Upstream's `host_list_will_write_and_batch_flags_reflect_deferred_steps`,
    /// for `leaves` (the last write that reaches the wait decides) and our
    /// steps; then the two lists run as one piece with a barrier between.
    #[test]
    fn leaves_and_join_flags_follow_the_lists_steps() {
        let Some(host) = super::host() else { return };
        let (Ok(producer), Ok(consumer)) = (host.new_list(), host.new_list()) else { return };
        let mut word = 0u64;
        let addr = (&raw mut word) as u64;
        let ge = |reference| ValueWait { address: addr, bytes: 8, comparison: 7, reference, mask: u64::MAX };
        assert_eq!(producer.leaves(&ge(5)), None, "no write");
        producer.defer_write(addr as *mut u8, Written::Value(7), 8);
        assert_eq!(producer.leaves(&ge(5)), Some(Some(true)));
        assert_eq!(producer.leaves(&ge(9)), Some(Some(false)));
        let low = ValueWait { bytes: 4, mask: 0xffff_ffff, ..ge(7) };
        assert_eq!(producer.leaves(&low), Some(Some(true)), "a 64-bit write over a 32-bit wait");
        producer.defer_write(addr as *mut u8, Written::GpuTimestamp, 8);
        assert_eq!(producer.leaves(&ge(5)), Some(None), "a timestamp is not known now");
        producer.defer_write((addr + 4) as *mut u8, Written::Value(1), 4);
        assert_eq!(producer.leaves(&ge(5)), Some(None), "part of the bytes");
        producer.defer_write(addr as *mut u8, Written::Value(9), 8);
        assert_eq!(producer.leaves(&ge(9)), Some(Some(true)));
        assert_eq!(producer.leaves(&ValueWait { address: addr + 8, ..ge(1) }), None, "the next word");

        assert_eq!(producer.join_flags(), JoinFlags::default());
        let src = [0x22u8; 16];
        let mut dst = [0u8; 16];
        producer.defer_copy(dst.as_mut_ptr(), src.as_ptr(), 16);
        let p = producer.join_flags();
        assert!(p.title_copy && p.cpu_read && !p.gpu_write && !p.after && !p.uploads && !p.lands);
        consumer.defer_after(|| {});
        let c = consumer.join_flags();
        assert!(c.after);
        assert!(crate::hold::joined::may_join(p, c));
        assert!(!crate::hold::joined::may_join(c, p));

        producer.close();
        consumer.close();
        host.run(&[&producer, &consumer], None, &[false, true]);
        assert_eq!(word, 9);
        assert_eq!(dst, src);
        assert!(!host.joined_barriers(1).is_empty(), "the barrier list was made");
    }

    #[test]
    fn a_joined_piece_keeps_its_barriers_by_part() {
        let Some(host) = super::host() else { return };
        let lists: Vec<_> = (0..3).filter_map(|_| host.new_list().ok()).map(std::sync::Arc::new).collect();
        if lists.len() < 3 {
            return;
        }
        let parts = super::copy_order_parts(lists.iter().cloned().zip([true, true, false]).collect());
        // Without `VICTORY_COPY_ORDER=split` in the test, one part: its first
        // list has no barrier before it.
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].1, [false, true, false]);
        for l in &lists {
            l.close();
        }
    }
}
