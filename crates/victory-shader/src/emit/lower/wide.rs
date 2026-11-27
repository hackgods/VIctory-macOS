//! Mac (`Mac::wide_loads`, `emit::Options::wide_loads`): the dwords of one
//! console load read four a `rawBufferLoad` call, after 64bitMango 759c4d6b
//! and fxceless/victory-fork c36971b2. In a file of its own so that
//! `lower.rs` stays under the asset policy's file size limit.

use super::Lower;
use crate::emit::bitcode::{Bin, V};

impl Lower<'_> {
    /// Mac (`Mac::wide_loads`, after 64bitMango 759c4d6b and
    /// fxceless/victory-fork c36971b2): `count` dwords from byte `addr` on
    /// of the raw view `h`, as integers, read in the current block by one
    /// call for every four of them. They are the dwords of ONE console
    /// load, which the console reads as one: the same bytes a call a dword
    /// reads. Loads of different console instructions are never joined:
    /// their offsets are computed in 32-bit registers, and `x - 4` and `x`
    /// at `x = 0` are one dword out of the view and its first one, where
    /// one call from `x - 4` on would read both out of it. The console
    /// answers each dword past the view's end with zero; so does the host
    /// for a call that reaches past it, dword by dword ([`Mac::wide_loads`]).
    ///
    /// A call a dword reads dword `k` at `addr + 4k`, wrapping at 32 bits;
    /// what one call for four does past 32 bits is up to the host. So each
    /// group of four is one call only where its offset, `addr + 4 * first`
    /// as the per-dword reads add it, is a constant whose bytes all lie
    /// below 2^32; anywhere else (an offset computed at run time, or one
    /// that wraps) its dwords are read a call each, as without the switch.
    pub(super) fn wide_dwords(&mut self, h: V, addr: V, count: u32) -> Vec<V> {
        let c = self.cur;
        let (u, al) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
        let base = self.s.m.int_value(addr).map(|x| x as u32);
        let mut out = Vec::with_capacity(count as usize);
        let mut read = |l: &mut Self, a: V, n: u32| {
            let mask = l.s.m.i8((1u8 << n) - 1);
            let r = l.raw_buffer_load(c, &[h, a, u, mask, al]);
            for k in 0..n {
                let x = l.s.m.extract(c, r, k);
                out.push(l.as_int(x));
            }
        };
        for first in (0..count).step_by(4) {
            let n = (count - first).min(4);
            let start = base.map(|b| b.wrapping_add(4 * first));
            if start.is_some_and(|s| u64::from(s) + u64::from(4 * n) <= 1 << 32) {
                let a = match first {
                    0 => addr,
                    _ => {
                        let off = self.s.m.i32(4 * first);
                        self.s.m.bin(c, Bin::Add, addr, off)
                    }
                };
                read(self, a, n);
                continue;
            }
            for k in first..first + n {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                read(self, a, 1);
            }
        }
        out
    }
}
