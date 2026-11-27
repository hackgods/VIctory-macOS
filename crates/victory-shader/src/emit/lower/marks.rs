//! Write marks only where a store changes what is there (`emit`, "Write
//! marks", [`crate::emit::Options::precise_marks`]).
//!
//! A store through a buffer view marks the 64 KiB pages it writes, and the
//! host compares every marked page with title memory to put back what the
//! shaders changed. Programs that store the values a buffer already holds
//! (a prefix sum over a scene that did not change, a clear of what is
//! clear) marked every page they stored to: in a world frame 22,600 pages
//! compared for 920 that had changed. With the option, a store of dwords
//! first loads the dwords it will overwrite and marks only if one differs.
//!
//! No change goes unmarked. Take the stores to one dword in the order they
//! land. The first of them that changes the dword was sent by a lane whose
//! load ran before it, so before any change: it loaded the dword's first
//! value, stores another, and marks. A store that puts the first value
//! back, after one that changed it, either loaded the changed value (and
//! marks) or loaded the first value and does not mark, but the store that
//! changed it has. Marks are per page, so a page any of whose bytes ends
//! other than it began is marked.
//!
//! Atomics, the byte and half stores made of two atomics, and stores
//! through typed views mark as before, whatever they write.

use super::*;

thread_local! {
    /// Whether stores mark only where they change what is there
    /// ([`crate::emit::Options::precise_marks`]), for the lowering running
    /// on this thread.
    pub(crate) static PRECISE_MARKS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

impl Lower<'_> {
    /// Whether a raw or structured store of `count` dwords with the
    /// arguments `args` (handle, two coordinates, the dwords, mask and
    /// alignment, as `rawBufferStore` takes them) changes a dword of what
    /// is there, loaded in the current block (call it before the store):
    /// the lanes to mark in. `None` to mark as before: without precise
    /// marks, or through a handle whose writes are not marked.
    pub(super) fn store_changes(&mut self, args: &[V], count: u32) -> Option<V> {
        if !PRECISE_MARKS.get() || !self.x.view_marks || !self.s.view_index.contains_key(&args[0]) {
            return None;
        }
        let c = self.cur;
        let count = count.min(4);
        let (mask, align) = (self.s.m.i8((1u8 << count) - 1), self.s.m.i32(4));
        let there = self.raw_op(c, &[args[0], args[1], args[2], mask, align]);
        let mut changes = None;
        for k in 0..count {
            let x = self.s.m.extract(c, there, k);
            let x = self.as_int(x);
            let differs = self.s.m.cmp(c, pred::NE, x, args[3 + k as usize]);
            changes = Some(match changes {
                Some(any) => self.s.m.bin(c, Bin::Or, any, differs),
                None => differs,
            });
        }
        changes
    }

    /// `dx.op.rawBufferLoad.f32` of `args`, in `block`.
    fn raw_op(&mut self, block: usize, args: &[V]) -> V {
        self.s.op(block, "rawBufferLoad.f32", Attr::ReadOnly, self.s.t.res_ret, 139, args)
    }
}
