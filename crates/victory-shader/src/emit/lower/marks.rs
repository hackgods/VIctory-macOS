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
//!
//! The load sees the host's copy, which holds the title's bytes at
//! submission (every view a shader can store through is brought up to
//! date then) but for one case: bytes shaders wrote that the host kept
//! from title memory because their list's writes were withheld. A store of
//! the same bytes by a later list would find nothing changed, and the
//! title would never get them. The host keeps a bit per such page (the
//! kept pages, after the written pages' buffer:
//! [`crate::emit::VIEW_KEPT_AT`]); a store to a kept page marks whatever
//! it stores, as every store did.

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
    /// the lanes to mark in. `x` and `bytes` are the store's first byte
    /// in the view and its length: a store to a page the host keeps
    /// shaders' bytes of marks whatever it stores. `None` to mark as
    /// before: without precise marks, or through a handle whose writes are
    /// not marked.
    pub(super) fn store_changes(&mut self, args: &[V], count: u32, x: V, bytes: V) -> Option<V> {
        if !PRECISE_MARKS.get() || !self.x.view_marks {
            return None;
        }
        let &index = self.s.view_index.get(&args[0])?;
        let kept = self.kept_page(index, x, bytes);
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
        changes.map(|any| self.s.m.bin(c, Bin::Or, any, kept))
    }

    /// Whether the page of the first or the last byte of a store of `bytes`
    /// bytes at byte `x` of the view at heap index `index` is one whose
    /// copy keeps bytes of shaders that title memory lacks: its bit among
    /// the kept pages, numbered as the written pages are (the view's base
    /// bit and phase from the mark table, as [`Lower::mark_view_write`]
    /// reads them). False for a view without marks.
    fn kept_page(&mut self, index: V, x: V, bytes: V) -> V {
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let table = self.s.view_marks_handle();
        let c = self.cur;
        let eight = k(self, 8);
        let at = self.s.m.bin(c, Bin::Mul, index, eight);
        let (base, phase) = self.raw_pair(table, at);
        let none = k(self, crate::emit::VIEW_MARK_NONE);
        let valid = self.s.m.cmp(c, pred::NE, base, none);
        let first = self.s.m.bin(c, Bin::Add, phase, x);
        let (zero, one) = (k(self, 0), k(self, 1));
        let nonempty = self.s.m.cmp(c, pred::NE, bytes, zero);
        let span = self.s.m.bin(c, Bin::Sub, bytes, one);
        let span = self.s.m.select(c, nonempty, span, zero);
        let last = self.s.m.bin(c, Bin::Add, first, span);
        let (k16, cap) = (k(self, 16), k(self, crate::emit::VIEW_WRITE_BITS - 1));
        let (k5, k2, k31, kept_at) = (k(self, 5), k(self, 2), k(self, 31), k(self, crate::emit::VIEW_KEPT_AT));
        let bitmap = self.s.view_writes_handle();
        let mut any = None;
        for byte in [first, last] {
            let page = self.s.m.bin(c, Bin::LShr, byte, k16);
            let n = self.s.m.bin(c, Bin::Add, base, page);
            let over = self.s.m.cmp(c, pred::IUGT, n, cap);
            let n = self.s.m.select(c, over, cap, n);
            let word = self.s.m.bin(c, Bin::LShr, n, k5);
            let word = self.s.m.bin(c, Bin::Shl, word, k2);
            let word = self.s.m.bin(c, Bin::Add, word, kept_at);
            let bit = self.s.m.bin(c, Bin::And, n, k31);
            let bits = self.raw_dword(bitmap, word);
            let bits = self.s.m.bin(c, Bin::LShr, bits, bit);
            let bits = self.s.m.bin(c, Bin::And, bits, one);
            let set = self.s.m.cmp(c, pred::NE, bits, zero);
            any = Some(match any {
                Some(other) => self.s.m.bin(c, Bin::Or, other, set),
                None => set,
            });
        }
        let any = any.expect("two bytes");
        self.s.m.bin(c, Bin::And, any, valid)
    }

    /// `dx.op.rawBufferLoad.f32` of `args`, in `block`.
    fn raw_op(&mut self, block: usize, args: &[V]) -> V {
        self.s.op(block, "rawBufferLoad.f32", Attr::ReadOnly, self.s.t.res_ret, 139, args)
    }
}
