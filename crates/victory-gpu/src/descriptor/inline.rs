//! mac: views named whole in their records, after fxceless's fork
//! (`VICTORY_VIEW_INLINE`, 578d459c there; `crate::mac_switches::view_inline`,
//! off by default).
//!
//! A record is the tag and two words, `i` and `generation`. Without the
//! switch both name an entry of the view table (`super::VIEWS`). The title
//! writes some 14,000 constant buffer views a frame, each at the address its
//! constants have that frame, and nothing releases a view without a
//! resource object, so the table and its index grow for the run (the fork:
//! 2.45 million entries, 1 GB after four minutes; 64bitMango: ~0.28 GB a
//! minute, and ~40 ms a frame of waits for the table's lock, which every
//! new view takes alone). With the switch the top two bits of `generation`
//! say what the other 62 bits are: [`CONSTANT`], a constant buffer view,
//! whole; [`PLACED`] (`=all` only), a view of placed memory: its address and
//! the number of its shape (the view without its address: as many as the
//! title has kinds of textures and buffers, however often it moves them).
//! Table generations stay below those bits ([`GENERATIONS`]). Records of
//! either kind never go stale: they name no resource object.

// The readers are Windows and test code (`super::view_at`, the heap sync).
#![cfg_attr(not(any(windows, test)), allow(dead_code))]

use super::View;

/// Where the kind sits in a record's `generation` word.
pub(super) const KIND_SHIFT: u32 = 30;
/// Table generations stay below the kind bits (and above 0).
pub(super) const GENERATIONS: u32 = (1 << KIND_SHIFT) - 1;
/// A constant buffer view, whole.
pub(super) const CONSTANT: u32 = 1;
/// A view of placed memory: its address and shape number.
pub(super) const PLACED: u32 = 2;
/// Bits of a constant buffer view's address in its record: any address a
/// program can have. The 15 bits above them are its size.
const CONSTANT_ADDRESS_BITS: u32 = 47;
/// Bits of a placed view's address in its record: placed memory is title
/// memory, which lies below 1 TiB. The 22 bits above them number its shape.
const PLACED_ADDRESS_BITS: u32 = 40;
const SHAPES: usize = 1 << (62 - PLACED_ADDRESS_BITS);

/// A view of placed memory without its address: whether it is unordered
/// access, the resource description, and the view's words but the two that
/// hold the address (10 and 11).
type Shape = (bool, [u32; 14], [u32; 14]);

/// The shapes of placed views named whole, and where each is. A lock of its
/// own: nearly every placed view has a shape seen before, so it is taken
/// shared, apart from the table's.
struct Shapes {
    list: Vec<Shape>,
    index: std::collections::BTreeMap<Shape, u32>,
}

static SHAPE_TABLE: std::sync::RwLock<Shapes> =
    std::sync::RwLock::new(Shapes { list: Vec::new(), index: std::collections::BTreeMap::new() });

/// The kind a record's `generation` word says (0: a table entry).
pub(super) fn kind(generation: u32) -> u32 {
    generation >> KIND_SHIFT
}

/// The two record words of the 62 bits `payload` of kind `kind`.
fn kind_words(kind: u32, payload: u64) -> [u32; 2] {
    [payload as u32, kind << KIND_SHIFT | (payload >> 32) as u32]
}

/// The 62 bits two record words carry beside their kind.
fn payload(i: u32, generation: u32) -> u64 {
    u64::from(generation & GENERATIONS) << 32 | u64::from(i)
}

/// The record words that name the constant buffer view of `size` bytes at
/// `address` whole; `None` if they cannot: an address past 47 bits, or a
/// size of 16 KiB or more that is no multiple of 256 or reaches 4 MiB (the
/// size has 15 bits: bytes below 16 KiB, else units of 256 bytes). Such a
/// view is a table entry.
fn constant_words(address: u64, size: u32) -> Option<[u32; 2]> {
    if address >> CONSTANT_ADDRESS_BITS != 0 {
        return None;
    }
    let code = if size < 1 << 14 {
        u64::from(size)
    } else if size.is_multiple_of(256) && size >> 8 < 1 << 14 {
        1 << 14 | u64::from(size >> 8)
    } else {
        return None;
    };
    Some(kind_words(CONSTANT, code << CONSTANT_ADDRESS_BITS | address))
}

/// The constant buffer view [`constant_words`] named.
fn constant_of(i: u32, generation: u32) -> View {
    let payload = payload(i, generation);
    let code = (payload >> CONSTANT_ADDRESS_BITS) as u32;
    let size = if code >> 14 == 0 { code } else { (code & 0x3fff) << 8 };
    View::Constant(payload & ((1 << CONSTANT_ADDRESS_BITS) - 1), size)
}

/// A placed view's shape and its address.
fn shape_of(unordered: bool, resource: &[u32; 14], desc: &[u32; 16]) -> (Shape, u64) {
    let mut rest = [0u32; 14];
    rest[..10].copy_from_slice(&desc[..10]);
    rest[10..].copy_from_slice(&desc[12..]);
    ((unordered, *resource, rest), u64::from(desc[10]) | u64::from(desc[11]) << 32)
}

/// The number of `shape`, a new one if it is not there yet; `None` when the
/// shapes are full (the view is then a table entry).
fn shape_number(shape: Shape) -> Option<u32> {
    if let Some(&at) = SHAPE_TABLE.read().unwrap_or_else(|e| e.into_inner()).index.get(&shape) {
        return Some(at);
    }
    let mut shapes = SHAPE_TABLE.write().unwrap_or_else(|e| e.into_inner());
    if let Some(&at) = shapes.index.get(&shape) {
        return Some(at);
    }
    if shapes.list.len() >= SHAPES {
        return None;
    }
    let at = shapes.list.len() as u32;
    shapes.list.push(shape);
    shapes.index.insert(shape, at);
    Some(at)
}

/// The record words that name `view` whole, if `placed` (`=all`) allows its
/// kind and it fits; `None`: the view is a table entry, as without the
/// switch.
pub(super) fn words(view: &View, placed: bool) -> Option<[u32; 2]> {
    match view {
        View::Constant(address, size) => constant_words(*address, *size),
        View::Placed(unordered, resource, desc) if placed => {
            let (shape, address) = shape_of(*unordered, resource, desc);
            if address >> PLACED_ADDRESS_BITS != 0 {
                return None;
            }
            let shape = shape_number(shape)?;
            Some(kind_words(PLACED, u64::from(shape) << PLACED_ADDRESS_BITS | address))
        }
        _ => None,
    }
}

/// The view the record words `i` and `generation` of kind [`CONSTANT`] or
/// [`PLACED`] name; `None` for another kind, or a shape not made here.
pub(super) fn view_of(i: u32, generation: u32) -> Option<View> {
    match kind(generation) {
        CONSTANT => Some(constant_of(i, generation)),
        PLACED => {
            let payload = payload(i, generation);
            let address = payload & ((1 << PLACED_ADDRESS_BITS) - 1);
            let shapes = SHAPE_TABLE.read().unwrap_or_else(|e| e.into_inner());
            let &(unordered, resource, rest) = shapes.list.get((payload >> PLACED_ADDRESS_BITS) as usize)?;
            let mut desc = [0u32; 16];
            desc[..10].copy_from_slice(&rest[..10]);
            (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
            desc[12..].copy_from_slice(&rest[10..]);
            Some(View::Placed(unordered, resource, desc))
        }
        _ => None,
    }
}

/// Whether the record words of an inline kind name a view (what
/// [`view_of`] would find): constant views always, placed views when their
/// shape is there.
pub(super) fn current(i: u32, generation: u32) -> bool {
    match kind(generation) {
        CONSTANT => true,
        PLACED => {
            let shape = (payload(i, generation) >> PLACED_ADDRESS_BITS) as usize;
            shape < SHAPE_TABLE.read().unwrap_or_else(|e| e.into_inner()).list.len()
        }
        _ => false,
    }
}

/// The shapes of placed views kept (for the views line).
#[cfg(windows)]
pub(super) fn shapes() -> usize {
    SHAPE_TABLE.read().unwrap_or_else(|e| e.into_inner()).list.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Constant buffer views are named whole in their records: any address
    /// a program can have, any size below 16 KiB, multiples of 256 bytes up
    /// to 4 MiB; what does not fit says so (the fork's test).
    #[test]
    fn constant_views_fit_their_records() {
        let top = (1u64 << CONSTANT_ADDRESS_BITS) - 1;
        for address in [0u64, 0x100, 0x2602_0000, 0xd_6b60_0000, 0x7ffd_a34f_0000, top] {
            for size in [0u32, 1, 255, 256, 4000, (1 << 14) - 1, 1 << 14, 65536, (1 << 22) - 256] {
                let [i, generation] = constant_words(address, size).expect("fits");
                assert_eq!(kind(generation), CONSTANT);
                assert!(current(i, generation));
                assert_eq!(view_of(i, generation), Some(View::Constant(address, size)), "{address:#x} {size}");
            }
        }
        assert_eq!(constant_words(top + 1, 256), None, "an address past 47 bits");
        assert_eq!(constant_words(0x1000, (1 << 14) + 1), None, "16 KiB or more, no multiple of 256");
        assert_eq!(constant_words(0x1000, 1 << 22), None, "4 MiB");
        // Different views never share their words.
        assert_ne!(constant_words(0x1000, 256), constant_words(0x1000, 512));
        assert_ne!(constant_words(0x1000, 256), constant_words(0x1100, 256));
        assert_ne!(constant_words(0x1000, 1 << 14), constant_words(0x1000, 64));
    }

    /// Placed views are named whole only with `=all`; one shape stands for
    /// the views alike at any address below 1 TiB, unordered ones are
    /// another; memory past 1 TiB is a table entry.
    #[test]
    fn placed_views_share_a_shape() {
        let resource: [u32; 14] = std::array::from_fn(|k| 0x5ee1_0000 + k as u32);
        let mut desc: [u32; 16] = std::array::from_fn(|k| 0x7a71_0000 + k as u32);
        let view = |unordered, desc: [u32; 16]| View::Placed(unordered, resource, desc);
        assert_eq!(words(&view(false, desc), false), None, "not without =all");
        let mut seen = None;
        for n in 0..1000u64 {
            let address = 0xd_4940_0000 + n * 0x1_0000;
            (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
            let [i, generation] = words(&view(false, desc), true).expect("fits");
            assert_eq!(kind(generation), PLACED);
            assert!(current(i, generation));
            assert_eq!(view_of(i, generation), Some(view(false, desc)));
            let shape = payload(i, generation) >> PLACED_ADDRESS_BITS;
            assert_eq!(*seen.get_or_insert(shape), shape, "one shape");
        }
        let [i, generation] = words(&view(true, desc), true).expect("fits");
        assert_ne!(payload(i, generation) >> PLACED_ADDRESS_BITS, seen.unwrap(), "unordered: another shape");
        assert_eq!(view_of(i, generation), Some(view(true, desc)));
        (desc[10], desc[11]) = (0, 0x100);
        assert_eq!(words(&view(false, desc), true), None, "past 1 TiB");
    }

    /// Other views are never named whole, and a table record's words read
    /// as no inline kind.
    #[test]
    fn other_views_stay_entries() {
        assert_eq!(words(&View::Shader(0x1000, None), true), None);
        assert_eq!(words(&View::Sampler([1; 13]), true), None);
        assert_eq!(kind(GENERATIONS), 0);
        assert!(!current(5, 1));
        assert_eq!(view_of(5, 1), None);
    }
}
