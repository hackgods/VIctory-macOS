//! macOS port: the title's map data streaming, watched and paced by time.
//!
//! Grand Theft Auto VI 1.0.2.1 (b99f876b...) keeps the placed parts of its
//! map ("map data": lists of buildings, objects and instance lists) in a
//! state machine that one routine at RVA 0x16cd4d0 steps once a frame. A
//! step walks the slots whose wanted state changed, newest first, and ends
//! after the first heavy change: one slot's entities made, one slot's
//! entities removed, or one slice of at most 250 entities switched on. The
//! opening sequence moves its camera by the clock, so at 2 frames a second
//! the slots left behind wait for their turn while new ones are made, and
//! the building pool (22,500 entries) runs out: the `int3` at +0x5149aec
//! (notes/pool-crash.md).
//!
//! `VICTORY_MAPDATA_WATCH=1` logs every five seconds how full the building
//! and object pools are, how many buildings were ever freed, and how many
//! slots wait in the routine's list. It reads the title's memory and changes
//! nothing.
//!
//! `VICTORY_MAPDATA_STEPS=<n>` (2 to 64) makes the per-frame caller run the
//! step up to `n` times a frame, as many as bring it to the console's 30
//! steps a second (one at 30 frames a second, 15 at 2). The caller's
//! `call <step>` is pointed at a small trampoline that repeats the call with
//! the same arguments and stops early once a step has staged a slice of
//! entities to switch on: later stages of the same frame consume that slice,
//! and a second step before them would stage the next one over it. Each step
//! is the title's own, taken and released under its own lock; what differs
//! from a console is that the frame's other work does not run between them.
//! A stopgap for a slow frame rate, not tried in the game yet.
//!
//! Only offsets and short byte patterns of the build are stored here
//! (docs/asset-policy.md).

/// The switch for the log line.
pub const WATCH_SWITCH: &str = "VICTORY_MAPDATA_WATCH";

/// The per-frame step, `push r15; push r14; push r13; push r12`.
const STEP: u32 = 0x016c_d4d0;
const STEP_BYTES: [u8; 8] = [0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54];
/// Its one per-frame caller: `sub rsp, 0x28; xor ecx, ecx; mov dl, 1;
/// call <step>; mov al, 1; add rsp, 0x28; ret`.
const THUNK: u32 = 0x022a_23a0;
const THUNK_BYTES: [u8; 20] = [
    0x48, 0x83, 0xec, 0x28, 0x31, 0xc9, 0xb2, 0x01, 0xe8, 0x23, 0xb1, 0x42, 0xff, 0xb0, 0x01, 0x48, 0x83, 0xc4,
    0x28, 0xc3,
];
/// Head of the list of slots with a change waiting: nodes of
/// `{ slot, next }`, new ones put at the head.
const WAITING: u32 = 0x076d_5098;
/// Pointers to the pools of buildings and of objects. A pool's header has
/// its entry size at +0x8, its capacity at +0xc, the entries in use at +0x10
/// and, in the upper half of the free list's head at +0x20, a count of the
/// entries ever freed.
const BUILDINGS: u32 = 0x0a9e_c938;
const OBJECTS: u32 = 0x0abb_e2e8;
/// The switch for the pacing, and the most steps a frame it accepts.
pub const STEPS_SWITCH: &str = "VICTORY_MAPDATA_STEPS";
const MOST_STEPS: u32 = 64;
/// Steps a second on a console (one a frame at 30 frames a second).
const CONSOLE_STEPS_A_SECOND: u64 = 30;
/// The thunk's `call <step>`.
const CALL_SITE: u32 = THUNK + 8;
/// How many slices of entities the last step staged for the frame's later
/// stages (a dword the step clears on entry).
const STAGED: u32 = 0x076d_50d4;
/// A walk of the waiting list stops here: the list is read without the
/// title's lock, so a node freed under us can lead anywhere.
const WALK_LIMIT: u32 = 1 << 16;

/// What the switch's value asks for.
pub fn watch_wanted(value: Option<&str>) -> bool {
    value == Some("1")
}

/// The most steps a frame the switch's value asks for; `None` leaves the
/// title as it is (unset, 0, 1 or not a number from 2 to 64).
pub fn steps_wanted(value: Option<&str>) -> Option<u32> {
    value?.trim().parse().ok().filter(|n| (2..=MOST_STEPS).contains(n))
}

/// Steps a frame that make 30 a second when `frames` frames took `elapsed`,
/// from 1 to `most`.
fn steps_for(frames: u32, elapsed: std::time::Duration, most: u32) -> u32 {
    if frames == 0 {
        return most.max(1);
    }
    let wanted = (CONSOLE_STEPS_A_SECOND * elapsed.as_millis() as u64).div_ceil(1000 * u64::from(frames));
    wanted.clamp(1, u64::from(most.max(1))) as u32
}

/// Appends the rel32 that reaches `target` from an instruction placed at
/// `at`, which ends `trailing` bytes after the rel32.
fn rel32(code: &mut Vec<u8>, at: u64, target: u64, trailing: u64) -> Option<()> {
    let next = at.checked_add(code.len() as u64 + 4 + trailing)?;
    let delta = i32::try_from((target as i64).checked_sub(next as i64)?).ok()?;
    code.extend(delta.to_le_bytes());
    Some(())
}

/// The trampoline placed at `at` for the image at `base`: the thunk's call,
/// repeated. `steps` is the dword that says how often, `frames` the dword
/// that counts the thunk's calls. `None` if a target is out of rel32 reach.
fn trampoline(base: u64, at: u64, steps: u64, frames: u64) -> Option<Vec<u8>> {
    let mut c = vec![
        0x53, // push rbx (the step keeps rbx; the stack is 16-aligned again)
        0x48, 0x83, 0xec, 0x20, // sub rsp, 0x20: the callee's shadow space
        0xf0, 0xff, 0x05, // lock inc dword [rip + frames]
    ];
    rel32(&mut c, at, frames, 0)?;
    c.extend([0x8b, 0x1d]); // mov ebx, [rip + steps]
    rel32(&mut c, at, steps, 0)?;
    let again = c.len();
    c.extend([
        0x31, 0xc9, // xor ecx, ecx (the thunk's arguments: the title's own list,
        0xb2, 0x01, // mov dl, 1     one heavy change a step)
        0xe8, // call <step>
    ]);
    rel32(&mut c, at, base + u64::from(STEP), 0)?;
    c.extend([0x83, 0x3d]); // cmp dword [rip + staged], 0
    rel32(&mut c, at, base + u64::from(STAGED), 1)?;
    c.push(0x00);
    c.extend([0x75, 0x04]); // jne <done>: a slice waits for the frame's later stages
    c.extend([0xff, 0xcb]); // dec ebx
    c.push(0x7f); // jg <again>
    c.push((again as i64 - (c.len() as i64 + 1)) as i8 as u8);
    c.extend([
        0x48, 0x83, 0xc4, 0x20, // done: add rsp, 0x20
        0x5b, // pop rbx
        0xc3, // ret
    ]);
    Some(c)
}

/// `call <trampoline>` for the five bytes at the site.
fn call_to(site: u64, at: u64) -> Option<[u8; 5]> {
    let delta = i32::try_from((at as i64).checked_sub(site as i64 + 5)?).ok()?;
    let d = delta.to_le_bytes();
    Some([0xe8, d[0], d[1], d[2], d[3]])
}

/// Where the trampoline's two dwords live once [`pace`] has run (0 before),
/// and the most steps a frame it was asked for.
static CELLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static MOST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

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

/// The log line; `paced` is the steps a frame now and the thunk's calls a
/// second, when the pacing is on.
fn line(
    buildings: Option<Pool>,
    objects: Option<Pool>,
    waiting: Option<(u32, bool)>,
    paced: Option<(u32, f32)>,
) -> String {
    let pool = |p: Option<Pool>| match p {
        Some(p) => format!("{} of {} ({} freed so far)", p.used, p.capacity, p.freed),
        None => "not made yet".to_owned(),
    };
    let waiting = match waiting {
        Some((n, true)) => format!("{n} or more"),
        Some((n, false)) => n.to_string(),
        None => "unknown".to_owned(),
    };
    let paced = match paced {
        Some((steps, calls)) => format!(", up to {steps} steps a frame at {calls:.1} frames/s"),
        None => String::new(),
    };
    format!(
        "map data: buildings {}, objects {}, slots waiting for their step {waiting}{paced}",
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

/// Points the per-frame caller of the image mapped at `base` at a
/// trampoline that runs the step up to `most` times a frame. Before any
/// title thread runs. Says why if it does not.
#[cfg(windows)]
pub fn pace(base: u64, most: u32) -> Result<(), &'static str> {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_READWRITE, VirtualAlloc,
        VirtualProtect,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    if !known(base) {
        return Err("not the build it was written for (bytes differ)");
    }
    // One page of code and one of data within reach of a rel32 call.
    const PAGE: usize = 0x1000;
    let mut pool = std::ptr::null_mut();
    for step in 1..2048u64 {
        let Some(want) = base.checked_sub(step << 20) else { break };
        // SAFETY: a plain reservation at a hint; failure returns null.
        pool = unsafe { VirtualAlloc(want as *const _, 2 * PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        if !pool.is_null() {
            break;
        }
    }
    if pool.is_null() {
        return Err("no memory near the image for the trampoline");
    }
    let code_at = pool as u64;
    let cells = code_at + PAGE as u64;
    let site = base + u64::from(CALL_SITE);
    let (Some(code), Some(call)) = (trampoline(base, code_at, cells, cells + 4), call_to(site, code_at)) else {
        return Err("the trampoline is out of reach");
    };
    let mut old = 0;
    // SAFETY: our pages hold the code and the two dwords (one step a frame
    // to begin with); the site is the title's code, made writable for the
    // write, before any title thread runs.
    unsafe {
        std::ptr::copy_nonoverlapping(code.as_ptr(), code_at as *mut u8, code.len());
        (cells as *mut u32).write(1);
        VirtualProtect(pool, PAGE, PAGE_EXECUTE_READ, &mut old);
        if VirtualProtect(site as *const _, call.len(), PAGE_EXECUTE_READWRITE, &mut old) == 0 {
            return Err("the title's code could not be made writable");
        }
        std::ptr::copy_nonoverlapping(call.as_ptr(), site as *mut u8, call.len());
        VirtualProtect(site as *const _, call.len(), old, &mut old);
        FlushInstructionCache(GetCurrentProcess(), std::ptr::null(), 0);
    }
    MOST.store(most.clamp(1, MOST_STEPS), std::sync::atomic::Ordering::Relaxed);
    CELLS.store(cells as usize, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// Starts the thread that keeps the pacing of [`pace`] at 30 steps a second
/// (if it ran) and, with `log`, writes the pools and the waiting list of the
/// image mapped at `base` to the log every five seconds. Says why if it does
/// not.
#[cfg(windows)]
pub fn watch(base: u64, log: bool) -> Result<(), &'static str> {
    use std::sync::atomic::{AtomicU32, Ordering};
    if !known(base) {
        return Err("not the build it was written for (bytes differ)");
    }
    let base = base as usize;
    let cells = CELLS.load(Ordering::Relaxed);
    let most = MOST.load(Ordering::Relaxed);
    std::thread::Builder::new()
        .name("victory map data watch".into())
        .spawn(move || {
            // SAFETY: our own page, alive for the process's lifetime: the
            // steps a frame, then the thunk's calls so far.
            let paced = (cells != 0)
                .then(|| unsafe { (&*(cells as *const AtomicU32), &*((cells + 4) as *const AtomicU32)) });
            let mut reader = Reader::default();
            let (mut seen, mut since) = (0u32, std::time::Instant::now());
            let (mut rate, mut second) = (0.0, 0u64);
            loop {
                std::thread::sleep(std::time::Duration::from_secs(1));
                second += 1;
                if let Some((steps, frames)) = paced
                    && second % 2 == 0
                {
                    let (now, total) = (std::time::Instant::now(), frames.load(Ordering::Relaxed));
                    let (elapsed, new) = (now - since, total.wrapping_sub(seen));
                    steps.store(steps_for(new, elapsed, most), Ordering::Relaxed);
                    rate = new as f32 / elapsed.as_secs_f32().max(0.001);
                    (seen, since) = (total, now);
                }
                if log && second % 5 == 0 {
                    reader.page = None;
                    let buildings = reader.pool(base + BUILDINGS as usize);
                    let objects = reader.pool(base + OBJECTS as usize);
                    let waiting = reader.waiting(base + WAITING as usize);
                    let paced = paced.map(|(steps, _)| (steps.load(Ordering::Relaxed), rate));
                    tracing::info!("{}", line(buildings, objects, waiting, paced));
                }
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
            line(Some(full), None, Some((1234, false)), None),
            "map data: buildings 22500 of 22500 (5821 freed so far), objects not made yet, \
             slots waiting for their step 1234"
        );
        assert!(line(None, None, Some((WALK_LIMIT, true)), None).ends_with("65536 or more"));
        assert!(line(None, None, None, None).ends_with("unknown"));
        assert!(line(None, None, None, Some((15, 2.04))).ends_with("unknown, up to 15 steps a frame at 2.0 frames/s"));
    }

    #[test]
    fn the_switch_takes_two_to_sixty_four_steps() {
        assert_eq!(steps_wanted(Some("16")), Some(16));
        assert_eq!(steps_wanted(Some(" 2 ")), Some(2));
        assert_eq!(steps_wanted(Some("64")), Some(64));
        for off in [None, Some(""), Some("0"), Some("1"), Some("65"), Some("-3"), Some("auto")] {
            assert_eq!(steps_wanted(off), None, "{off:?}");
        }
    }

    #[test]
    fn steps_make_thirty_a_second() {
        let s = std::time::Duration::from_secs;
        assert_eq!(steps_for(60, s(2), 16), 1, "30 frames a second: the title as it is");
        assert_eq!(steps_for(90, s(2), 16), 1, "faster than a console is still one step");
        assert_eq!(steps_for(4, s(2), 16), 15, "2 frames a second");
        assert_eq!(steps_for(5, s(2), 16), 12, "2.5 frames a second");
        assert_eq!(steps_for(1, s(2), 16), 16, "0.5 frames a second wants 60: the most allowed");
        assert_eq!(steps_for(0, s(2), 16), 16, "no frame at all");
        assert_eq!(steps_for(0, s(2), 0), 1, "never none");
    }

    #[test]
    fn the_trampoline_repeats_the_thunks_call() {
        let (base, at) = (0x6fff_ed55_0000u64, 0x6fff_ed35_0000u64);
        let (steps, frames) = (at + 0x1000, at + 0x1004);
        let code = trampoline(base, at, steps, frames).unwrap();
        assert_eq!(code.len(), 46);
        let target = |end: usize, trailing: u64| {
            let delta = i32::from_le_bytes(code[end - 4..end].try_into().unwrap());
            (at + end as u64 + trailing).wrapping_add(delta as i64 as u64)
        };
        assert_eq!(&code[5..8], &[0xf0, 0xff, 0x05]);
        assert_eq!(target(12, 0), frames);
        assert_eq!(&code[12..14], &[0x8b, 0x1d]);
        assert_eq!(target(18, 0), steps);
        // The loop's body starts with the thunk's own argument bytes and call.
        assert_eq!(&code[18..23], &THUNK_BYTES[4..9]);
        assert_eq!(target(27, 0), base + u64::from(STEP));
        assert_eq!(&code[27..29], &[0x83, 0x3d]);
        assert_eq!(target(33, 1), base + u64::from(STAGED), "the rel32 counts the imm8 after it");
        assert_eq!(code[33], 0);
        // jne <done> skips dec and jg; jg goes back to the argument bytes.
        assert_eq!(&code[34..36], &[0x75, 0x04]);
        assert_eq!(36 + 4, 40);
        assert_eq!(&code[36..38], &[0xff, 0xcb]);
        assert_eq!(code[38], 0x7f);
        assert_eq!(40 + i64::from(code[39] as i8), 18);
        assert_eq!(&code[40..], &[0x48, 0x83, 0xc4, 0x20, 0x5b, 0xc3]);
        // One push and 0x20 of shadow space: 16-aligned at the call, as in the thunk.
        assert_eq!((8 + 8 + 0x20) % 16, 0);
        assert!(trampoline(base, base - (3 << 30), steps, frames).is_none(), "out of reach is refused");
    }

    #[test]
    fn the_site_call_lands_on_the_trampoline() {
        let base = 0x6fff_ed55_0000u64;
        let (site, at) = (base + u64::from(CALL_SITE), 0x6fff_ed35_0000u64);
        assert_eq!(THUNK_BYTES[(CALL_SITE - THUNK) as usize], 0xe8, "the site is the thunk's call");
        let c = call_to(site, at).unwrap();
        assert_eq!(c[0], 0xe8);
        let delta = i32::from_le_bytes(c[1..5].try_into().unwrap());
        assert_eq!((site + 5).wrapping_add(delta as i64 as u64), at);
        assert!(call_to(site, site + (1 << 40)).is_none(), "out of reach is refused");
    }
}
