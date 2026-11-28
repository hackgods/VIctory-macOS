//! mac: when [`crate::shadow::release_idle`] (VIctory-macOS #381) gives up
//! the host copies of title memory that no view or list asked for lately,
//! and a line every 5 s of what went.
//!
//! `VICTORY_RELEASE_COPIES` (upstream it is on unless `0`; on this line it
//! is off unless set, so that a run with it is a clean A/B):
//!
//! - unset or `0`: never.
//! - `1` (or any value other than the two below, as upstream): while less
//!   than `VICTORY_RAM_LOW` (0.10) of the computer's memory is free as Wine
//!   counts it, until `VICTORY_RAM_FREE` (0.15) is, together with the host
//!   textures (`crate::resource::evict_unused`). Wine counts free plus
//!   inactive pages (`host_statistics64`, `free_count + inactive_count`,
//!   `ntdll/unix/system.c`; `GlobalMemoryStatusEx` keeps it for a second).
//!   Inactive pages include anonymous ones, which macOS frees only by
//!   compressing or swapping them, so that count does not fall as the
//!   compressor and swap grow: on the 48 GB M4 Pro in the boat it read about
//!   7 GB (14 %) while macOS had 0.1 GB free, 7.4 GB in the compressor and
//!   2.4 GB swapped out.
//! - `host`, `host:<low GB>` or `host:<low GB>:<free GB>`: while less than
//!   `low` GB ([`HOST_LOW_GB`]) is available as macOS counts it, until
//!   `free` GB (`low` + [`HOST_ABOVE_GB`]) is. The textures keep Wine's
//!   count. macOS's count is `kern.memorystatus_level`, the share that
//!   `memory_pressure` prints as "System-wide memory free percentage": free
//!   pages and pages macOS can drop without compressing them (file-backed,
//!   purgeable), not the anonymous inactive pages Wine adds. `host:0` gives
//!   nothing up but still writes the 5-s line, with both counts.
//!
//! With any of the last two, a line every 5 s gives the copies and MB given
//! up in those 5 s, the MB since the start, the copies made again since the
//! start, what the copies that may go hold now, and the memory as both
//! counts see it, with macOS's compressor and swap.

#![cfg_attr(not(windows), allow(dead_code))]

const GB: f64 = (1u64 << 30) as f64;

/// The `low` mark of `VICTORY_RELEASE_COPIES=host`, in GB. In the boat on
/// the 48 GB M4 Pro, macOS's count fell from 68 % to 28-30 % (13-14 GB)
/// while the title loaded and stayed there, so 15 GB acts once loading is
/// done and on through the run.
pub(crate) const HOST_LOW_GB: f64 = 15.0;
/// How far above `low` the `free` mark is unless given.
pub(crate) const HOST_ABOVE_GB: f64 = 2.0;
/// The largest mark a value may name, in GB.
const HOST_MOST_GB: f64 = 1024.0;

/// When the idle copies go.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Mode {
    Off,
    /// With the textures, by Wine's count (`resource::ram_over`).
    Wine,
    /// By macOS's count: below `low` bytes available, until `free` are.
    Host { low: u64, free: u64 },
}

/// The mode for a value of `VICTORY_RELEASE_COPIES`. A mark that is not a
/// number of GB from 0 to [`HOST_MOST_GB`] is left out; `free` is never
/// under `low`.
pub(crate) fn mode_from(value: Option<&str>) -> Mode {
    let Some(value) = value.map(str::trim) else {
        return Mode::Off;
    };
    if value == "0" {
        return Mode::Off;
    }
    let rest = match value.strip_prefix("host") {
        Some(rest) if rest.is_empty() || rest.starts_with(':') => rest,
        _ => return Mode::Wine,
    };
    let gb = |v: &str| {
        v.trim()
            .parse::<f64>()
            .ok()
            .filter(|g| (0.0..=HOST_MOST_GB).contains(g))
    };
    let mut marks = rest.split(':').skip(1);
    let low = marks.next().and_then(gb).unwrap_or(HOST_LOW_GB);
    let free = marks
        .next()
        .and_then(gb)
        .unwrap_or(low + HOST_ABOVE_GB)
        .max(low);
    Mode::Host {
        low: (low * GB) as u64,
        free: (free * GB) as u64,
    }
}

/// With `available` bytes free by macOS's count: what to give up so that
/// `free` would be, once less than `low` is.
pub(crate) fn host_excess(available: u64, low: u64, free: u64) -> u64 {
    if available < low { free.saturating_sub(available) } else { 0 }
}

/// The counts of the 5-s line.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct Counts {
    /// Copies given up in this window, and their bytes.
    pub(crate) copies: u64,
    pub(crate) bytes: u64,
    /// Frames in this window in which some went.
    pub(crate) frames: u64,
}

impl Counts {
    /// A frame in which `copies` copies of `bytes` went.
    pub(crate) fn add(&mut self, copies: u64, bytes: u64) {
        if copies > 0 {
            self.copies += copies;
            self.bytes += bytes;
            self.frames += 1;
        }
    }
}

#[cfg(windows)]
pub(crate) use imp::*;

#[cfg(windows)]
mod imp {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::sync::{Mutex, OnceLock};
    use std::time::{Duration, Instant};

    /// `VICTORY_RELEASE_COPIES`, read once and logged when set.
    pub(crate) fn mode() -> Mode {
        static MODE: OnceLock<Mode> = OnceLock::new();
        *MODE.get_or_init(|| {
            let mode = mode_from(std::env::var("VICTORY_RELEASE_COPIES").ok().as_deref());
            match mode {
                Mode::Off => {}
                Mode::Wine => tracing::info!(
                    "VICTORY_RELEASE_COPIES: idle host copies of title memory are given up with the host textures, \
                     while Wine counts less than VICTORY_RAM_LOW of the memory free (free + inactive pages)"
                ),
                Mode::Host { low, free } => tracing::info!(
                    low_mb = low >> 20,
                    free_mb = free >> 20,
                    "VICTORY_RELEASE_COPIES=host: idle host copies of title memory are given up while macOS counts \
                     less than `low` available (kern.memorystatus_level), until `free` is; textures keep Wine's count"
                ),
            }
            mode
        })
    }

    /// What [`crate::shadow::release_idle`] gives up this frame: nothing,
    /// `wine` (what the textures give up for Wine's count), or what macOS's
    /// count asks for.
    pub(crate) fn want(wine: u64) -> u64 {
        match mode() {
            Mode::Off => 0,
            Mode::Wine => wine,
            Mode::Host { low, free } => host::memory()
                .map_or(0, |m| host_excess(m.available, low, free)),
        }
    }

    struct Window {
        counts: Counts,
        since: Instant,
    }

    static WINDOW: Mutex<Option<Window>> = Mutex::new(None);
    static TOTAL_BYTES: AtomicU64 = AtomicU64::new(0);

    /// After [`crate::shadow::release_idle`] ran this frame and `copies`
    /// copies of `bytes` went: every 5 s, a line. `made_again` is the
    /// copies made again since the start, and `held` the copies that may go
    /// (not aliased) and their bytes; both are asked for only for the line.
    pub(crate) fn after(
        copies: usize,
        bytes: u64,
        made_again: impl FnOnce() -> u64,
        held: impl FnOnce() -> (usize, u64),
    ) {
        if mode() == Mode::Off {
            return;
        }
        let total = TOTAL_BYTES.fetch_add(bytes, Relaxed) + bytes;
        let mut window = WINDOW.lock().unwrap_or_else(|e| e.into_inner());
        let w = window.get_or_insert_with(|| Window {
            counts: Counts::default(),
            since: Instant::now(),
        });
        w.counts.add(copies as u64, bytes);
        if w.since.elapsed() < Duration::from_secs(5) {
            return;
        }
        let Counts {
            copies,
            bytes,
            frames,
        } = std::mem::take(&mut w.counts);
        w.since = Instant::now();
        drop(window);
        let (held_copies, held_bytes) = held();
        let mac = host::memory();
        let mb = |b: u64| b >> 20;
        tracing::info!(
            copies,
            mb = mb(bytes),
            frames,
            total_mb = mb(total),
            made_again = made_again(),
            held_copies,
            held_mb = mb(held_bytes),
            wine_free_mb = wine_available().map(mb),
            mac_free_mb = mac.map(|m| mb(m.available)),
            compressor_mb = mac.and_then(|m| m.compressor).map(mb),
            swap_mb = mac.and_then(|m| m.swap).map(mb),
            "host copies of title memory given up (5 s): copies and MB in these 5 s, frames with some, MB since \
             the start, copies made again since the start, copies that may go and their MB now; free memory as \
             Wine counts it (free + inactive) and as macOS does (kern.memorystatus_level), macOS's compressor \
             and swap"
        );
    }

    /// The computer's free memory as Wine counts it.
    fn wine_available() -> Option<u64> {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        // SAFETY: plain data, its length set as the call asks.
        let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        m.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
        // SAFETY: a valid out pointer.
        (unsafe { GlobalMemoryStatusEx(&mut m) } != 0).then_some(m.ullAvailPhys)
    }

    /// The memory as macOS counts it, read with `sysctl(2)` straight from
    /// the title's x86-64 code under Rosetta (as `crate::alias` makes its
    /// Mach calls).
    pub(crate) mod host {
        /// macOS's count, in bytes.
        #[derive(Clone, Copy, Debug)]
        pub(crate) struct Memory {
            /// `kern.memorystatus_level` (a whole percent) of `hw.memsize`.
            pub(crate) available: u64,
            /// `vm.compressor_bytes_used`.
            pub(crate) compressor: Option<u64>,
            /// `vm.swapusage`'s `xsu_used`.
            pub(crate) swap: Option<u64>,
        }

        /// macOS's count now; `None` off macOS or where it cannot be read.
        pub(crate) fn memory() -> Option<Memory> {
            #[cfg(target_arch = "x86_64")]
            {
                read()
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                None
            }
        }

        #[cfg(target_arch = "x86_64")]
        fn read() -> Option<Memory> {
            use std::sync::OnceLock;
            struct Mibs {
                level: Vec<i32>,
                compressor: Option<Vec<i32>>,
                total: u64,
            }
            static MIBS: OnceLock<Option<Mibs>> = OnceLock::new();
            let mibs = MIBS
                .get_or_init(|| {
                    if !crate::watch::rosetta() {
                        return None;
                    }
                    let mibs = Mibs {
                        level: mib(c"kern.memorystatus_level")?,
                        compressor: mib(c"vm.compressor_bytes_used"),
                        total: value::<u64>(&[CTL_HW, HW_MEMSIZE])?,
                    };
                    (mibs.total > 0).then_some(mibs)
                })
                .as_ref()?;
            let level = value::<u32>(&mibs.level)?.min(100) as u64;
            Some(Memory {
                available: mibs.total / 100 * level,
                compressor: mibs.compressor.as_deref().and_then(value::<u64>),
                swap: value::<[u64; 4]>(&[CTL_VM, VM_SWAPUSAGE]).map(|s| s[2]),
            })
        }

        const CTL_VM: i32 = 2;
        const VM_SWAPUSAGE: i32 = 5;
        const CTL_HW: i32 = 6;
        const HW_MEMSIZE: i32 = 24;

        /// The MIB of a sysctl name (`sysctlnametomib`: `{0, 3}`).
        #[cfg(target_arch = "x86_64")]
        fn mib(name: &std::ffi::CStr) -> Option<Vec<i32>> {
            let mut mib = [0i32; 12];
            let name = name.to_bytes();
            // SAFETY: `mib` and `name` are valid for the lengths given.
            let len = unsafe {
                sysctl(
                    &[0, 3],
                    mib.as_mut_ptr().cast(),
                    size_of_val(&mib),
                    name.as_ptr(),
                    name.len(),
                )
            }?;
            let n = len / size_of::<i32>();
            (n > 0).then(|| mib[..n].to_vec())
        }

        /// A sysctl value of exactly `T`'s size.
        #[cfg(target_arch = "x86_64")]
        fn value<T: Copy + Default>(mib: &[i32]) -> Option<T> {
            let mut v = T::default();
            // SAFETY: `v` is valid for its size, and plain data.
            let len = unsafe {
                sysctl(
                    mib,
                    (&raw mut v).cast(),
                    size_of::<T>(),
                    std::ptr::null(),
                    0,
                )
            }?;
            (len == size_of::<T>()).then_some(v)
        }

        /// `sysctl(2)` (BSD system call 202), the bytes written to `old`.
        ///
        /// # Safety
        /// `old` is valid for `old_len` bytes and `new` for `new_len`, and
        /// this is x86-64 code on macOS (Rosetta).
        #[cfg(target_arch = "x86_64")]
        unsafe fn sysctl(mib: &[i32], old: *mut u8, old_len: usize, new: *const u8, new_len: usize) -> Option<usize> {
            const SYS_SYSCTL: u64 = 0x0200_0000 | 202;
            let mut len = old_len;
            let failed: u8;
            // SAFETY: the x86-64 XNU system call ABI: arguments in rdi, rsi,
            // rdx, r10, r8, r9; rax, rdx, rcx and r11 are written; the carry
            // flag says it failed (rax is then the errno). The kernel
            // writes only `old` and `len`.
            unsafe {
                core::arch::asm!(
                    "syscall",
                    "setc {failed}",
                    failed = out(reg_byte) failed,
                    inlateout("rax") SYS_SYSCTL => _,
                    in("rdi") mib.as_ptr(),
                    in("rsi") mib.len(),
                    inlateout("rdx") old => _,
                    in("r10") &raw mut len,
                    in("r8") new,
                    in("r9") new_len,
                    lateout("rcx") _,
                    lateout("r11") _,
                    options(nostack),
                );
            }
            (failed == 0).then_some(len)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: u64 = 1 << 30;

    /// Unset or `0` is off on this line; `1` and other values keep
    /// upstream's behaviour; `host` takes macOS's count with its marks.
    #[test]
    fn the_switch_picks_the_mode() {
        assert_eq!(mode_from(None), Mode::Off);
        assert_eq!(mode_from(Some("0")), Mode::Off);
        assert_eq!(mode_from(Some(" 0 ")), Mode::Off);
        assert_eq!(mode_from(Some("1")), Mode::Wine);
        assert_eq!(mode_from(Some("yes")), Mode::Wine);
        assert_eq!(mode_from(Some("hostile")), Mode::Wine);
        assert_eq!(
            mode_from(Some("host")),
            Mode::Host {
                low: 15 * G,
                free: 17 * G
            }
        );
        assert_eq!(
            mode_from(Some("host:4")),
            Mode::Host {
                low: 4 * G,
                free: 6 * G
            }
        );
        assert_eq!(
            mode_from(Some("host:4:8")),
            Mode::Host {
                low: 4 * G,
                free: 8 * G
            }
        );
        assert_eq!(
            mode_from(Some("host:0")),
            Mode::Host { low: 0, free: 2 * G }
        );
    }

    /// A mark that is no number of GB is left out, and `free` is never
    /// under `low`.
    #[test]
    fn host_marks_ignore_bad_values_and_keep_their_order() {
        let default = mode_from(Some("host"));
        for bad in ["host:", "host:x", "host:-1", "host:2000", "host:nan"] {
            assert_eq!(mode_from(Some(bad)), default, "{bad:?}");
        }
        assert_eq!(
            mode_from(Some("host:8:4")),
            Mode::Host {
                low: 8 * G,
                free: 8 * G
            }
        );
        assert_eq!(
            mode_from(Some("host:4:x")),
            Mode::Host {
                low: 4 * G,
                free: 6 * G
            }
        );
    }

    /// Below `low` it asks for what brings the memory up to `free`; at or
    /// above `low`, nothing. `low` 0 never asks.
    #[test]
    fn host_excess_works_to_the_free_mark() {
        assert_eq!(host_excess(14 * G, 15 * G, 17 * G), 3 * G);
        assert_eq!(host_excess(15 * G, 15 * G, 17 * G), 0);
        assert_eq!(host_excess(0, 0, 2 * G), 0);
        assert_eq!(host_excess(G, 2 * G, 2 * G), G);
    }

    /// The window counts copies, bytes and the frames in which some went.
    #[test]
    fn the_window_counts_frames_with_copies() {
        let mut c = Counts::default();
        c.add(0, 0);
        c.add(3, 30);
        c.add(1, 5);
        assert_eq!(
            c,
            Counts {
                copies: 4,
                bytes: 35,
                frames: 2
            }
        );
    }

    /// Under Wine on macOS, macOS's count reads: a share of the memory
    /// available, and the compressor and swap.
    #[cfg(windows)]
    #[test]
    fn macos_memory_reads_under_rosetta() {
        if !crate::watch::rosetta() {
            assert!(host::memory().is_none());
            return;
        }
        let m = host::memory().expect("macOS's count");
        assert!(m.available > 0 && m.available < 1024 * G, "{m:?}");
        assert!(m.compressor.is_some() && m.swap.is_some(), "{m:?}");
    }
}
