//! macOS port: the title's pools of buildings and objects, watched.
//!
//! Grand Theft Auto VI 1.0.2.1 (b99f876b...) stops on its own `int3` at
//! +0x5149aec about five minutes into the opening sequence: its pool of
//! buildings (22,500 entries) has no free entry when the entities of a piece
//! of map are made (notes/pool-crash.md).
//!
//! `VICTORY_MAPDATA_WATCH=1` logs every five seconds how full the building
//! and object pools are, how many entries of each were ever freed, and how
//! many interiors wait in the list of the interiors' per-frame step. It
//! reads the title's memory and changes nothing.
//!
//! That step (RVA 0x16cd4d0) was first read as the step of the whole map
//! and paced by `VICTORY_MAPDATA_STEPS`; it is the state machine of the
//! interiors only (its pools are made under the names "InteriorProxy" and
//! "InteriorInst"), a run with the pacing stopped at the same place, and the
//! pacing is gone again.
//!
//! Only offsets and short byte patterns of the build are stored here
//! (docs/asset-policy.md).

/// The switch for the log line.
pub const WATCH_SWITCH: &str = "VICTORY_MAPDATA_WATCH";

/// The interiors' per-frame step, `push r15; push r14; push r13; push r12`,
/// here to tell the build.
const STEP: u32 = 0x016c_d4d0;
const STEP_BYTES: [u8; 8] = [0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54];
/// Its one per-frame caller: `sub rsp, 0x28; xor ecx, ecx; mov dl, 1;
/// call <step>; mov al, 1; add rsp, 0x28; ret`.
const THUNK: u32 = 0x022a_23a0;
const THUNK_BYTES: [u8; 20] = [
    0x48, 0x83, 0xec, 0x28, 0x31, 0xc9, 0xb2, 0x01, 0xe8, 0x23, 0xb1, 0x42, 0xff, 0xb0, 0x01, 0x48, 0x83, 0xc4,
    0x28, 0xc3,
];
/// Head of the list of interiors with a change of state waiting: nodes of
/// `{ interior, next }`, new ones put at the head.
const WAITING: u32 = 0x076d_5098;
/// Pointers to the pools of buildings and of objects. A pool's header has
/// its storage at +0x0, its entry size at +0x8, its capacity at +0xc, the
/// entries in use at +0x10, in the upper half of the free list's head at
/// +0x20 a count of the entries ever freed, and at +0x28 a dword per entry
/// whose low 17 bits are all set while the entry is in use.
const BUILDINGS: u32 = 0x0a9e_c938;
const OBJECTS: u32 = 0x0abb_e2e8;
/// A walk of the waiting list stops here: the list is read without the
/// title's lock, so a node freed under us can lead anywhere.
const WALK_LIMIT: u32 = 1 << 16;

/// What the switch's value asks for.
pub fn watch_wanted(value: Option<&str>) -> bool {
    value == Some("1")
}

/// One pool's numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pool {
    pub used: u32,
    pub capacity: u32,
    /// Entries ever returned to the pool.
    pub freed: u32,
}

/// A pool's numbers from the 0x28 bytes at the start of its header.
fn pool_from(header: &[u8; 0x28]) -> Pool {
    let dword = |at: usize| u32::from_le_bytes(header[at..at + 4].try_into().unwrap());
    Pool { capacity: dword(0xc), used: dword(0x10), freed: dword(0x24) }
}

/// The log line.
fn line(buildings: Option<Pool>, objects: Option<Pool>, waiting: Option<(u32, bool)>) -> String {
    let pool = |p: Option<Pool>| match p {
        Some(p) => format!("{} of {} ({} freed so far)", p.used, p.capacity, p.freed),
        None => "not made yet".to_owned(),
    };
    let waiting = match waiting {
        Some((n, true)) => format!("{n} or more"),
        Some((n, false)) => n.to_string(),
        None => "unknown".to_owned(),
    };
    format!(
        "map data: buildings {}, objects {}, interiors waiting for their step {waiting}",
        pool(buildings),
        pool(objects)
    )
}

/// Whether the image mapped at `base` is the build these offsets are for.
#[cfg(windows)]
fn known(base: u64) -> bool {
    let at = |rva: u32| (base + u64::from(rva)) as *const u8;
    // SAFETY: the image is mapped and these RVAs are inside its code.
    unsafe {
        std::slice::from_raw_parts(at(STEP), STEP_BYTES.len()) == STEP_BYTES
            && std::slice::from_raw_parts(at(THUNK), THUNK_BYTES.len()) == THUNK_BYTES
    }
}

/// Reads of the title's memory that cannot fault: a page is asked about
/// before it is read, and the answer for the last page is kept.
#[cfg(windows)]
#[derive(Default)]
struct Reader {
    page: Option<usize>,
}

#[cfg(windows)]
impl Reader {
    fn readable(&mut self, address: usize) -> bool {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
        };
        let page = address & !0xfff;
        if self.page == Some(page) {
            return true;
        }
        // SAFETY: a zeroed MEMORY_BASIC_INFORMATION is valid.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: reads the address space's bookkeeping, not `address` itself.
        let n = unsafe { VirtualQuery(address as *const _, &mut info, size_of_val(&info)) };
        let ok = n != 0
            && info.State == MEM_COMMIT
            && info.Protect & PAGE_GUARD == 0
            && info.Protect & 0xff != PAGE_NOACCESS;
        self.page = ok.then_some(page);
        ok
    }

    fn bytes<const N: usize>(&mut self, address: usize) -> Option<[u8; N]> {
        if address == 0 || !self.readable(address) || !self.readable(address + N - 1) {
            return None;
        }
        // SAFETY: both ends are in committed, readable pages; the bytes are
        // plain data that another thread may change under us.
        Some(unsafe { std::ptr::read_volatile(address as *const [u8; N]) })
    }

    fn pointer(&mut self, address: usize) -> Option<usize> {
        self.bytes::<8>(address).map(usize::from_le_bytes)
    }

    fn pool(&mut self, at: usize) -> Option<Pool> {
        let header = self.pointer(at)?;
        self.bytes::<0x28>(header).map(|h| pool_from(&h))
    }

    /// Nodes in the waiting list, and whether the walk was cut short.
    fn waiting(&mut self, head: usize) -> Option<(u32, bool)> {
        let mut node = self.pointer(head)?;
        let mut n = 0;
        while node != 0 {
            if n == WALK_LIMIT {
                return Some((n, true));
            }
            let Some(pair) = self.bytes::<16>(node) else { return Some((n, true)) };
            node = usize::from_le_bytes(pair[8..].try_into().unwrap());
            n += 1;
        }
        Some((n, false))
    }
}

/// Starts a thread that logs the pools and the interiors' waiting list of
/// the image mapped at `base` every five seconds. Says why if it does not.
#[cfg(windows)]
pub fn watch(base: u64) -> Result<(), &'static str> {
    if !known(base) {
        return Err("not the build it was written for (bytes differ)");
    }
    let base = base as usize;
    std::thread::Builder::new()
        .name("victory pool watch".into())
        .spawn(move || {
            let mut reader = Reader::default();
            loop {
                std::thread::sleep(std::time::Duration::from_secs(5));
                reader.page = None;
                let buildings = reader.pool(base + BUILDINGS as usize);
                let objects = reader.pool(base + OBJECTS as usize);
                let waiting = reader.waiting(base + WAITING as usize);
                tracing::info!("{}", line(buildings, objects, waiting));
            }
        })
        .map(|_| ())
        .map_err(|_| "its thread did not start")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_one_turns_the_watch_on() {
        assert!(watch_wanted(Some("1")));
        assert!(!watch_wanted(Some("0")));
        assert!(!watch_wanted(Some("")));
        assert!(!watch_wanted(None));
    }

    /// The thunk's call goes to the step these offsets name.
    #[test]
    fn the_thunk_calls_the_step() {
        assert_eq!(THUNK_BYTES[8], 0xe8);
        let delta = i32::from_le_bytes(THUNK_BYTES[9..13].try_into().unwrap());
        assert_eq!((i64::from(THUNK) + 13 + i64::from(delta)) as u32, STEP);
    }

    #[test]
    fn a_pool_header_gives_its_numbers() {
        let mut header = [0u8; 0x28];
        header[0x8..0xc].copy_from_slice(&0xe0u32.to_le_bytes());
        header[0xc..0x10].copy_from_slice(&22_500u32.to_le_bytes());
        header[0x10..0x14].copy_from_slice(&22_196u32.to_le_bytes());
        // The free list's head: an empty list (0x1fffe) after 5821 frees.
        header[0x20..0x28].copy_from_slice(&0x16bd_0001_fffeu64.to_le_bytes());
        assert_eq!(pool_from(&header), Pool { used: 22_196, capacity: 22_500, freed: 5821 });
    }

    #[test]
    fn the_line_says_what_is_known() {
        let full = Pool { used: 22_500, capacity: 22_500, freed: 5821 };
        assert_eq!(
            line(Some(full), None, Some((12, false))),
            "map data: buildings 22500 of 22500 (5821 freed so far), objects not made yet, \
             interiors waiting for their step 12"
        );
        assert!(line(None, None, Some((WALK_LIMIT, true))).ends_with("65536 or more"));
        assert!(line(None, None, None).ends_with("unknown"));
    }
}
