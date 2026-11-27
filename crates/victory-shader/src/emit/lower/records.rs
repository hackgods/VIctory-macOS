//! Views read from the title's own records (`emit`, "Views from the
//! title's records"): the record at a view's heap index, read through the
//! title page table, and what it names.

use super::*;

thread_local! {
    /// The view kinds translated shaders take from the title's own records
    /// ([`crate::emit::Options::gpu_records`]), for the lowering running on this
    /// thread: bit 0, constant buffer views; bit 1, raw buffer views of
    /// placed memory.
    pub(crate) static GPU_RECORDS: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// The tag of a view record in a title's descriptor heap (victory-gpu's
/// `descriptor::VIEW_TAG`).
pub(crate) const RECORD_TAG: u32 = u32::from_le_bytes(*b"vVIE");

/// A read through a handle made from a title record
/// ([`Lower::record_read`]), for [`Lower::raw_buffer_load`].
#[derive(Clone, Copy)]
pub(super) struct RecordRead {
    /// The offset handed out with the handle.
    given: V,
    /// The offset in the view that it stands for.
    user: V,
    /// The last offset in the view a dword may be read at.
    last: V,
    /// The handle of the memory holding the access's last byte where that
    /// lies on the next 64 KiB page (the null view's elsewhere), the
    /// access's offset in it, whether the access crosses, and the bytes of
    /// it before the page's end.
    tail: V,
    tail_at: V,
    crosses: V,
    before: V,
    /// For an access whose offset is not known to be 4-byte aligned: its
    /// low two bits, whether they are set, and the handles of the first
    /// page's memory and the next page's where they are (the null view's
    /// where the access is aligned): [`Lower::record_load`] then puts each
    /// dword together from the two aligned dwords it lies in.
    odd: Option<(V, V, V, V)>,
}

/// What the title's record at a heap index names, read by the shader
/// (`emit`, "Views from the title's records").
#[derive(Clone, Copy)]
pub(super) struct RecordView {
    /// The record names a view whole that is read from title memory: a
    /// constant buffer view, or a raw buffer view of placed memory.
    constant: V,
    /// Where its bytes start: the low dword, and bits 32-46.
    lo: V,
    hi: V,
    /// Its bytes.
    size: V,
    /// The heap index the host's own view is at, for any other record.
    legacy: V,
}

impl Lower<'_> {
    /// Through a handle made from a title record, the dwords past the
    /// end of the view the record names read 0, as each did through the
    /// host view of that size (D3D12 bounds raw loads per component).
    /// And the dwords of an access that crosses onto the next 64 KiB page
    /// are read through the view of the memory holding that page, which
    /// need not be the first page's (two copies of title memory, or
    /// pages of a heap that do not follow each other).
    pub(super) fn record_load(&mut self, block: usize, args: &[V], read: RecordRead) -> V {
        if let Some(odd) = read.odd {
            return self.record_load_odd(block, args, read, odd);
        }
        let r = self.raw_op(block, args);
        // How far this load is from the access's start.
        let moved = if args[1] == read.given {
            self.s.m.i32(0)
        } else {
            self.s.m.bin(block, Bin::Sub, args[1], read.given)
        };
        let user = self.s.m.bin(block, Bin::Add, read.user, moved);
        let fields = self.s.m.int_value(args[3]).map_or(0xF, |mask| mask as u8);
        let first = self.s.m.cmp(block, pred::IULE, user, read.last);
        let one = self.s.m.i8(1);
        let mut masked = [None; 4];
        for (k, field) in masked.iter_mut().enumerate() {
            if fields >> k & 1 == 0 {
                continue;
            }
            let step = self.s.m.i32(4 * k as u32);
            let held = if k == 0 {
                first
            } else {
                let at = self.s.m.bin(block, Bin::Add, user, step);
                let inside = self.s.m.cmp(block, pred::IULE, at, read.last);
                self.s.m.bin(block, Bin::And, first, inside)
            };
            let from = self.s.m.bin(block, Bin::Add, moved, step);
            let past = self.s.m.cmp(block, pred::IUGE, from, read.before);
            let past = self.s.m.bin(block, Bin::And, read.crosses, past);
            // Each dword in a load of its own: the access's offset in
            // the next page's memory is below 0 (it starts on the page
            // before), and an offset wraps from one load to the next,
            // not within a load.
            let tail_at = self.s.m.bin(block, Bin::Add, read.tail_at, from);
            let tail = self.s.op(
                block,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                139,
                &[read.tail, tail_at, args[2], one, args[4]],
            );
            *field = Some(crate::emit::bitcode::Masked {
                held,
                tail: Some((past, tail, 0)),
                whole: None,
            });
        }
        self.s.m.masked.insert(r, masked);
        r
    }

    /// [`Self::record_load`] at an offset that need not be 4-byte aligned:
    /// raw loads address dwords, so each dword read is put together from
    /// the two aligned dwords it lies in (its low bytes from the first,
    /// the rest from the second), and each of those is read where it is:
    /// in the first page's memory or the next page's, 0 past the view's
    /// size. A dword across two pages so takes its bytes from both. Where
    /// the offset turns out aligned, the second dwords are read through
    /// the null view and not used.
    fn record_load_odd(&mut self, block: usize, args: &[V], read: RecordRead, odd: (V, V, V, V)) -> V {
        let (low, is_odd, main_next, tail_next) = odd;
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let fields = self.s.m.int_value(args[3]).map_or(0xF, |mask| mask as u8);
        // The load, from the aligned dword its first dword starts in.
        let aligned = self.s.m.bin(block, Bin::Sub, args[1], low);
        let r = self.raw_op(block, &[args[0], aligned, args[2], args[3], args[4]]);
        // That dword's place from the access's start (below 0 by up to 3)
        // and in the view.
        let moved = self.s.m.bin(block, Bin::Sub, aligned, read.given);
        let user = self.s.m.bin(block, Bin::Add, read.user, moved);
        let tail_at = self.s.m.bin(block, Bin::Add, read.tail_at, moved);
        let (one, zero) = (self.s.m.i8(1), k(self, 0));
        // Aligned dword `j` from there, as a number.
        let mut words: [Option<V>; 5] = [None; 5];
        for (j, slot) in words.iter_mut().enumerate() {
            let own = j < 4 && fields >> j & 1 != 0;
            let next = j > 0 && fields >> (j - 1) & 1 != 0;
            if !own && !next {
                continue;
            }
            let step = k(self, 4 * j as u32);
            let (from, at, tail_at) = (
                self.s.m.bin(block, Bin::Add, moved, step),
                self.s.m.bin(block, Bin::Add, user, step),
                self.s.m.bin(block, Bin::Add, tail_at, step),
            );
            // In the first page's memory: a field of the load, or for a
            // dword only the last odd dword needs, a load of its own.
            let main = if own {
                self.s.m.extract(block, r, j as u32)
            } else {
                let at = self.s.m.bin(block, Bin::Add, aligned, step);
                let x = self.raw_op(block, &[main_next, at, args[2], one, args[4]]);
                self.s.m.extract(block, x, 0)
            };
            let through = if own { read.tail } else { tail_next };
            let tail = self.raw_op(block, &[through, tail_at, args[2], one, args[4]]);
            let tail = self.s.m.extract(block, tail, 0);
            let past = self.s.m.cmp(block, pred::SGE, from, read.before);
            let past = self.s.m.bin(block, Bin::And, read.crosses, past);
            let word = self.s.m.select(block, past, tail, main);
            let word = self.s.m.cast(block, Cast::Bitcast, word, self.s.t.i32);
            let held = self.s.m.cmp(block, pred::IULE, at, read.last);
            *slot = Some(self.s.m.select(block, held, word, zero));
        }
        let (k3, k31, k32) = (k(self, 3), k(self, 31), k(self, 32));
        let down = self.s.m.bin(block, Bin::Shl, low, k3);
        let up = self.s.m.bin(block, Bin::Sub, k32, down);
        let up = self.s.m.bin(block, Bin::And, up, k31);
        let held = self.s.m.bool(true);
        let mut masked = [None; 4];
        for (j, field) in masked.iter_mut().enumerate() {
            let (Some(first), Some(second)) = (words[j], words[j + 1]) else {
                continue;
            };
            if fields >> j & 1 == 0 {
                continue;
            }
            let lower = self.s.m.bin(block, Bin::LShr, first, down);
            let upper = self.s.m.bin(block, Bin::Shl, second, up);
            let both = self.s.m.bin(block, Bin::Or, lower, upper);
            let whole = self.s.m.select(block, is_odd, both, first);
            let whole = self.s.m.cast(block, Cast::Bitcast, whole, self.s.t.f32);
            *field = Some(crate::emit::bitcode::Masked {
                held,
                tail: None,
                whole: Some(whole),
            });
        }
        self.s.m.masked.insert(r, masked);
        r
    }

    /// The bound view heap's first descriptor in title memory (low dword,
    /// high dword) and its slots: three dwords of the heap-info table, root
    /// SRV `t`[`HEAP_RECORDS`] in [`crate::emit::ROOT_CBV_SPACE`], read
    /// once in the entry block. Zero slots: no record is read. The fourth
    /// dword counts the shapes in the shape table.
    /// The fifth value is the null view's heap index, the page table's
    /// first block-0 entry.
    fn heap_info(&mut self) -> (V, V, V, V, V) {
        if let Some(info) = self.heap_info {
            return info;
        }
        let table = self.s.root_srv_handle(crate::emit::HEAP_RECORDS);
        let b = self.s.entry;
        let (zero, u) = (self.s.m.i32(0), self.s.m.undef(self.s.t.i32));
        let (mask, al) = (self.s.m.i8(15), self.s.m.i32(4));
        let r = self.raw_buffer_load(b, &[table, zero, u, mask, al]);
        let [lo, hi, count, shapes] = [0, 1, 2, 3].map(|j| {
            let x = self.s.m.extract(b, r, j);
            self.s.m.cast(b, Cast::Bitcast, x, self.s.t.i32)
        });
        let pages = self.s.root_srv_handle(crate::emit::HEAP_PAGES);
        let (first, mask) = (self.s.m.i32(4096 * 4), self.s.m.i8(1));
        let r = self.raw_buffer_load(b, &[pages, first, u, mask, al]);
        let x = self.s.m.extract(b, r, 0);
        let null = self.s.m.cast(b, Cast::Bitcast, x, self.s.t.i32);
        self.heap_info = Some((lo, hi, count, shapes, null));
        (lo, hi, count, shapes, null)
    }

    /// The page table's entry for title address `(hi:lo)`, without a handle
    /// or a mark: the heap index of the raw view of the memory holding the
    /// page (the null view's if none does) and the low dword of that
    /// memory's title address.
    fn page_view(&mut self, lo: V, hi: V) -> (V, V) {
        let c = self.cur;
        let map = self.s.root_srv_handle(crate::emit::HEAP_PAGES);
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        // Level 1: (hi:lo) >> 28, 4096 entries for 40-bit addresses; then
        // the page's entry in its block (block 0: no memory).
        let (k4, k28, k256, zero, two) = (k(self, 4), k(self, 28), k(self, 256), k(self, 0), k(self, 2));
        let top = self.s.m.bin(c, Bin::Shl, hi, k4);
        let low = self.s.m.bin(c, Bin::LShr, lo, k28);
        let l1 = self.s.m.bin(c, Bin::Or, top, low);
        let inside = self.s.m.cmp(c, pred::IULT, hi, k256);
        let l1 = self.s.m.select(c, inside, l1, zero);
        let at = self.s.m.bin(c, Bin::Shl, l1, two);
        let block = self.raw_dword(map, at);
        let block = self.s.m.select(c, inside, block, zero);
        let (k16, k12, mask) = (k(self, 16), k(self, 12), k(self, 0xFFF));
        let page = self.s.m.bin(c, Bin::LShr, lo, k16);
        let page = self.s.m.bin(c, Bin::And, page, mask);
        let entry = self.s.m.bin(c, Bin::Shl, block, k12);
        let entry = self.s.m.bin(c, Bin::Or, entry, page);
        let (k3, first) = (self.s.m.i32(3), self.s.m.i32(4096 * 4));
        let at = self.s.m.bin(c, Bin::Shl, entry, k3);
        let at = self.s.m.bin(c, Bin::Add, at, first);
        let (u, mask, al) = (self.s.m.undef(self.s.t.i32), self.s.m.i8(3), self.s.m.i32(4));
        let r = self.raw_buffer_load(c, &[map, at, u, mask, al]);
        let [view, start] = [0, 1].map(|j| {
            let x = self.s.m.extract(c, r, j);
            self.as_int(x)
        });
        (view, start)
    }

    /// Reads, in the current block, the title's record at heap index
    /// `index` (`emit`, "Views from the title's records"): the record's
    /// three dwords through the page table at the heap's address, and what
    /// they name. `legacy` is the index of the host's own view of the slot.
    fn record_view(&mut self, index: V, legacy: V) -> RecordView {
        let (start_lo, start_hi, count, shapes, null) = self.heap_info();
        let c = self.cur;
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let inside = self.s.m.cmp(c, pred::IULT, index, count);
        let k5 = k(self, 5);
        let at = self.s.m.bin(c, Bin::Shl, index, k5);
        let (lo, hi) = self.add64(start_lo, start_hi, at, false);
        let (view, start) = self.page_view(lo, hi);
        let c = self.cur;
        let view = self.s.m.select(c, inside, view, null);
        let h = self.s.heap_handle(c, view, false, true, (11, 0));
        self.s.view_index.remove(&h);
        let off = self.s.m.bin(c, Bin::Sub, lo, start);
        let (u, mask, al) = (self.s.m.undef(self.s.t.i32), self.s.m.i8(7), k(self, 4));
        let r = self.raw_buffer_load(c, &[h, off, u, mask, al]);
        let [tag, low, high] = [0, 1, 2].map(|j| {
            let x = self.s.m.extract(c, r, j);
            self.as_int(x)
        });
        // The tag, then the kind in the top two bits of the third dword: 1,
        // a constant buffer view named whole, its 47-bit address below 15
        // bits of size (bytes below 16 KiB, else units of 256 bytes).
        let tagged = {
            let tag_word = k(self, RECORD_TAG);
            self.s.m.cmp(c, pred::EQ, tag, tag_word)
        };
        let (k30, one) = (k(self, 30), k(self, 1));
        let kind = self.s.m.bin(c, Bin::LShr, high, k30);
        let is_constant = self.s.m.cmp(c, pred::EQ, kind, one);
        let constant = self.s.m.bin(c, Bin::And, inside, tagged);
        let constant = self.s.m.bin(c, Bin::And, constant, is_constant);
        // A view at an address that is not 4-byte aligned keeps its host
        // view (a copy of its own bytes, which raw loads can address).
        let (three, none) = (k(self, 3), k(self, 0));
        let odd = self.s.m.bin(c, Bin::And, low, three);
        let aligned = self.s.m.cmp(c, pred::EQ, odd, none);
        let constant = self.s.m.bin(c, Bin::And, constant, aligned);
        let (k15, m15, m14, k8, big) = (k(self, 15), k(self, 0x7FFF), k(self, 0x3FFF), k(self, 8), k(self, 1 << 14));
        let hi = self.s.m.bin(c, Bin::And, high, m15);
        let code = self.s.m.bin(c, Bin::LShr, high, k15);
        let code = self.s.m.bin(c, Bin::And, code, m15);
        let units = self.s.m.bin(c, Bin::And, code, m14);
        let units = self.s.m.bin(c, Bin::Shl, units, k8);
        let in_units = self.s.m.cmp(c, pred::IUGE, code, big);
        let size = self.s.m.select(c, in_units, units, code);
        if GPU_RECORDS.get() & 2 == 0 {
            return RecordView {
                constant,
                lo: low,
                hi,
                size,
                legacy,
            };
        }
        // Kind 2, a view of placed memory: a 40-bit address below 22 bits
        // of shape number. The shape table says whether the shape is a
        // buffer view read raw, where its bytes start from the address,
        // and how many they are.
        let (two, zero, m30, m8) = (k(self, 2), k(self, 0), k(self, 0x3FFF_FFFF), k(self, 0xFF));
        let is_placed = self.s.m.cmp(c, pred::EQ, kind, two);
        let shape = self.s.m.bin(c, Bin::And, high, m30);
        let shape = self.s.m.bin(c, Bin::LShr, shape, k8);
        let known = self.s.m.cmp(c, pred::IULT, shape, shapes);
        let shape = self.s.m.select(c, known, shape, zero);
        let k4 = k(self, 4);
        let at = self.s.m.bin(c, Bin::Shl, shape, k4);
        let table = self.s.root_srv_handle(crate::emit::HEAP_SHAPES);
        let (u, mask, al) = (self.s.m.undef(self.s.t.i32), self.s.m.i8(7), k(self, 4));
        let r = self.raw_buffer_load(c, &[table, at, u, mask, al]);
        let [raw, first, bytes] = [0, 1, 2].map(|j| {
            let x = self.s.m.extract(c, r, j);
            self.as_int(x)
        });
        let raw = self.s.m.cmp(c, pred::NE, raw, zero);
        let placed = self.s.m.bin(c, Bin::And, inside, tagged);
        let placed = self.s.m.bin(c, Bin::And, placed, is_placed);
        let placed = self.s.m.bin(c, Bin::And, placed, known);
        let placed = self.s.m.bin(c, Bin::And, placed, raw);
        let placed_hi = self.s.m.bin(c, Bin::And, high, m8);
        let (placed_lo, placed_hi) = self.add64(low, placed_hi, first, false);
        let c = self.cur;
        let odd = self.s.m.bin(c, Bin::And, placed_lo, three);
        let aligned = self.s.m.cmp(c, pred::EQ, odd, zero);
        let placed = self.s.m.bin(c, Bin::And, placed, aligned);
        RecordView {
            constant: self.s.m.bin(c, Bin::Or, constant, placed),
            lo: self.s.m.select(c, placed, placed_lo, low),
            hi: self.s.m.select(c, placed, placed_hi, hi),
            size: self.s.m.select(c, placed, bytes, size),
            legacy,
        }
    }

    /// The title's record behind the raw buffer view `src`, read where the
    /// view's handle would be made (the entry block for a descriptor at a
    /// fixed address, the join for one chosen by control flow, else the
    /// current block) and kept as that handle is; `None` for a view that
    /// is no heap descriptor (a root view).
    pub(super) fn record_for(&mut self, src: &Source) -> Result<Option<RecordView>, Why> {
        const PROPS: (u32, u32) = (11, 0);
        match src {
            Source::Loaded { load, first } if let Some(address) = self.entry_address(*load) => {
                let key = (address, *first);
                if let Some(&record) = self.entry_records.get(&key) {
                    return Ok(Some(record));
                }
                let here = std::mem::replace(&mut self.cur, self.s.entry);
                let index = self.entry_heap_index(address, *first, false);
                let legacy = self.buffer_guard(src, index, false, PROPS);
                let record = self.record_view(index, legacy);
                self.cur = here;
                self.entry_records.insert(key, record);
                Ok(Some(record))
            }
            Source::Loaded { load, first } => {
                let key = (self.cur, *load, *first);
                if let Some(&record) = self.block_records.get(&key) {
                    return Ok(Some(record));
                }
                let index = self.heap_index(*load, *first, false)?;
                let legacy = self.buffer_guard(src, index, false, PROPS);
                let record = self.record_view(index, legacy);
                self.block_records.insert(key, record);
                Ok(Some(record))
            }
            Source::Phi { block, .. } => {
                let key = format!("{src:?}");
                if let Some(&record) = self.join_records.get(&key) {
                    return Ok(Some(record));
                }
                let join = self.block[*block].ok_or("a descriptor chosen by control flow")?;
                let here = std::mem::replace(&mut self.cur, join);
                let record = self.uniform_heap_index(src, false).map(|index| {
                    let legacy = self.buffer_guard(src, index, false, PROPS);
                    self.record_view(index, legacy)
                });
                self.cur = here;
                let record = record?;
                self.join_records.insert(key, record);
                Ok(Some(record))
            }
            Source::Select { scc, args } => {
                let [
                    Source::Loaded { load: a, first: fa },
                    Source::Loaded { load: b, first: fb },
                ] = args[..]
                else {
                    return Err("a descriptor chosen by a condition");
                };
                let cond = match self.get(*scc)? {
                    Val::Bit(x) => x,
                    _ => return Err("SCC without a value"),
                };
                let set = self.heap_index(a, fa, false)?;
                let clear = self.heap_index(b, fb, false)?;
                let index = self.s.m.select(self.cur, cond, set, clear);
                let legacy = self.buffer_guard(src, index, false, PROPS);
                Ok(Some(self.record_view(index, legacy)))
            }
            Source::Fetched { .. } | Source::Lanes { .. } | Source::LanePhi { .. } => {
                let index = self.lane_heap_index(src, false)?;
                let legacy = self.buffer_guard(src, index, false, PROPS);
                Ok(Some(self.record_view(index, legacy)))
            }
            _ => Ok(None),
        }
    }

    /// A read at byte `off` of the raw view `record` stands for: a handle
    /// and the offset in it. For a constant buffer view named whole, the
    /// memory the page table names at its address plus `off`; for any other
    /// record, the host's own view of the slot. One handle at the index
    /// chosen (handles cannot be chosen between). Dwords read through it
    /// past the constant buffer's size read 0, and the dwords of an access
    /// of `bytes` bytes that crosses onto the next 64 KiB page are read
    /// through that page's memory, looked up only where the access crosses
    /// ([`Self::raw_buffer_load`]). An access at an offset not known to be
    /// 4-byte aligned is read a dword's two aligned dwords at a time
    /// ([`Self::record_load_odd`]). May end in a new block.
    pub(super) fn record_read(&mut self, src: &Source, record: RecordView, off: V, bytes: u32) -> (V, V) {
        // The host's own views of the heap are still read for other records.
        if self.is_heap_direct(src) {
            self.heap[0] = true;
        }
        let (lo, hi) = self.add64(record.lo, record.hi, off, false);
        let (view, start) = self.page_view(lo, hi);
        let c = self.cur;
        let null = self.heap_info().4;
        let four = self.s.m.i32(4);
        let fits = self.s.m.cmp(c, pred::IUGE, record.size, four);
        let view = self.s.m.select(c, fits, view, null);
        let index = self.s.m.select(c, record.constant, view, record.legacy);
        let in_memory = self.s.m.bin(c, Bin::Sub, lo, start);
        let at = self.s.m.select(c, record.constant, in_memory, off);
        let h = self.s.heap_handle(c, index, false, true, (11, 0));
        // An offset that may be off a dword: its low bits (the view itself
        // is aligned), and the same memory for the dwords' second halves,
        // the null view where it is not.
        let odd = match self.s.m.int_value(off) {
            Some(off) if off % 4 == 0 => None,
            _ => {
                let (three, zero) = (self.s.m.i32(3), self.s.m.i32(0));
                let low = self.s.m.bin(c, Bin::And, at, three);
                let is_odd = self.s.m.cmp(c, pred::NE, low, zero);
                let next = self.s.m.select(c, is_odd, index, null);
                let next = self.s.heap_handle(c, next, false, true, (11, 0));
                self.s.view_index.remove(&next);
                Some((low, is_odd, next))
            }
        };
        let all = self.s.m.i32(u32::MAX);
        let last = self.s.m.bin(c, Bin::Sub, record.size, four);
        let last = self.s.m.select(c, record.constant, last, all);
        // The access's last byte on the next page: that page's memory.
        let (page, low) = (self.s.m.i32(1 << 16), self.s.m.i32(0xFFFF));
        let in_page = self.s.m.bin(c, Bin::And, lo, low);
        let before = self.s.m.bin(c, Bin::Sub, page, in_page);
        let size = self.s.m.i32(bytes.max(1));
        let crosses = self.s.m.cmp(c, pred::IUGT, size, before);
        let crosses = self.s.m.bin(c, Bin::And, crosses, record.constant);
        let crosses = self.s.m.bin(c, Bin::And, crosses, fits);
        let guard = self.begin_guarded(Some(crosses));
        let (tail_view, tail_start) = {
            let end = self.s.m.i32(bytes.max(1) - 1);
            let (end_lo, end_hi) = self.add64(lo, hi, end, false);
            self.page_view(end_lo, end_hi)
        };
        let zero = self.s.m.i32(0);
        let [tail_view, tail_start] = self.end_guarded(guard, &[(tail_view, null), (tail_start, zero)])[..]
        else {
            unreachable!("two values in, two out")
        };
        let c = self.cur;
        let tail = self.s.heap_handle(c, tail_view, false, true, (11, 0));
        self.s.view_index.remove(&tail);
        let tail_at = self.s.m.bin(c, Bin::Sub, lo, tail_start);
        let odd = odd.map(|(low, is_odd, next)| {
            let view = self.s.m.select(c, is_odd, tail_view, null);
            let tail_next = self.s.heap_handle(c, view, false, true, (11, 0));
            self.s.view_index.remove(&tail_next);
            (low, is_odd, next, tail_next)
        });
        self.record_reads.insert(
            h,
            RecordRead {
                given: at,
                user: off,
                last,
                tail,
                tail_at,
                crosses,
                before,
                odd,
            },
        );
        (h, at)
    }
}
