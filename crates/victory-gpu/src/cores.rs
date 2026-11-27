//! On which cores the threads that submit prepare their submissions, and
//! how much of the preparation they were running (the frame report).
//!
//! Runs on an M5 Pro go through long stretches where preparing takes about
//! twice as long for the same work: as many refreshes, watches and guard
//! faults, but each slower. Two things can do that from outside: the thread
//! runs on a slower core (Apple silicon has two or three kinds, `hw.perflevelN`,
//! the fastest numbered last), or it is not running for part of the time
//! (other threads have the cores). The line tells them apart: the share of
//! the preparation spent on the fastest cores, and the thread's own CPU time
//! in it against the time it took.
//!
//! Under Rosetta both come straight from the kernel: the CPU number through
//! Wine's `GetCurrentProcessorNumber` (`pthread_cpu_number_np`), the
//! thread's CPU time through the `thread_selfusage` syscall, as
//! `crate::mac_pressure` reads the kernel's memory pressure.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// A thread's place and CPU time at one moment.
#[derive(Clone, Copy)]
pub(crate) struct Sample {
    cpu: u32,
    used_ns: Option<u64>,
    at: std::time::Instant,
}

/// Now, on the calling thread.
pub(crate) fn sample() -> Sample {
    Sample {
        cpu: current_cpu(),
        used_ns: thread_used_ns(),
        at: std::time::Instant::now(),
    }
}

/// Since the last frame report: `[0]` preparations, `[1]` microseconds
/// they took, `[2]` microseconds of it begun and ended on the fastest cores,
/// `[3]` preparations that ended on another core than they began, `[4]`
/// microseconds the thread ran during the preparations it could tell
/// (`[5]` microseconds those took).
static PREP: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// One submission's preparation, from `start` to now, on this thread.
pub(crate) fn note_preparation(start: Sample) {
    let end = sample();
    let wall = end.at.saturating_duration_since(start.at).as_micros() as u64;
    let fastest = fastest_from();
    PREP[0].fetch_add(1, Ordering::Relaxed);
    PREP[1].fetch_add(wall, Ordering::Relaxed);
    if fastest.is_some_and(|f| start.cpu >= f && end.cpu >= f) {
        PREP[2].fetch_add(wall, Ordering::Relaxed);
    }
    if start.cpu != end.cpu {
        PREP[3].fetch_add(1, Ordering::Relaxed);
    }
    if let (Some(a), Some(b)) = (start.used_ns, end.used_ns) {
        PREP[4].fetch_add(b.saturating_sub(a) / 1000, Ordering::Relaxed);
        PREP[5].fetch_add(wall, Ordering::Relaxed);
    }
}

/// The frame report's line, per frame over `frames` frames.
pub(crate) fn log(frames: f64) {
    let [n, wall, fast, moved, ran, timed] = PREP.each_ref().map(|a| a.swap(0, Ordering::Relaxed));
    if n == 0 {
        return;
    }
    let ms = |us: u64| us as f64 / 1e3 / frames;
    let share = |a: u64, b: u64| {
        if b > 0 {
            100.0 * a as f64 / b as f64
        } else {
            0.0
        }
    };
    let fastest = match (fastest_from(), cpu_count()) {
        (Some(f), Some(all)) => format!("CPUs {f}-{}", all - 1),
        _ => "unknown".into(),
    };
    tracing::info!(
        "preparing on the submitting threads: {:.1} ms/frame in {:.1} submissions; {:.0}% of it begun and ended \
         on the fastest cores ({fastest}), {:.1} submissions/frame moved between cores; the thread ran {:.0}% \
         of the time it took ({})",
        ms(wall),
        n as f64 / frames,
        share(fast, wall),
        moved as f64 / frames,
        share(ran, timed),
        if timed > 0 {
            "its own CPU time"
        } else {
            "not known here"
        },
    );
}

/// The first CPU number of the fastest cores: Apple silicon numbers its
/// kinds of cores from the slowest, so the fastest (`hw.perflevel0`) are
/// the last `hw.perflevel0.logicalcpu`. None where that is not known.
fn fastest_from() -> Option<u32> {
    static FROM: OnceLock<Option<u32>> = OnceLock::new();
    *FROM.get_or_init(|| {
        let all = cpu_count()?;
        let fastest = sysctl_u32(b"hw.perflevel0.logicalcpu")?;
        (fastest > 0 && fastest < all).then(|| all - fastest)
    })
}

fn cpu_count() -> Option<u32> {
    static ALL: OnceLock<Option<u32>> = OnceLock::new();
    *ALL.get_or_init(|| sysctl_u32(b"hw.logicalcpu"))
}

fn sysctl_u32(name: &[u8]) -> Option<u32> {
    if !crate::watch::rosetta() {
        return None;
    }
    let mut value = [0u8; 4];
    (crate::mac_pressure::sysctl_by_name(name, &mut value)? == 4).then(|| u32::from_le_bytes(value))
}

fn current_cpu() -> u32 {
    // SAFETY: no arguments.
    unsafe { windows_sys::Win32::System::Threading::GetCurrentProcessorNumber() }
}

/// The calling thread's CPU time in nanoseconds (`thread_selfusage`, BSD
/// syscall 482), under Rosetta only.
#[cfg(target_arch = "x86_64")]
fn thread_used_ns() -> Option<u64> {
    static UNDER_ROSETTA: OnceLock<bool> = OnceLock::new();
    if !*UNDER_ROSETTA.get_or_init(crate::watch::rosetta) {
        return None;
    }
    let ret: u64;
    let failed: u8;
    // SAFETY: XNU's x86-64 BSD syscall ABI under Rosetta 2:
    // `thread_selfusage` takes nothing and returns the calling thread's
    // user and system time in nanoseconds; an error sets carry.
    unsafe {
        core::arch::asm!(
            "syscall",
            "setc {failed}",
            failed = out(reg_byte) failed,
            inlateout("rax") 0x0200_0000u64 | 482 => ret,
            lateout("rdx") _,
            lateout("rcx") _,
            lateout("r11") _,
        );
    }
    (failed == 0).then_some(ret)
}

#[cfg(not(target_arch = "x86_64"))]
fn thread_used_ns() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A thread that spins for 50 ms ran about all of it, one that sleeps
    /// barely any (where the kernel tells).
    #[test]
    fn a_threads_cpu_time_counts_its_running_only() {
        let Some(a) = thread_used_ns() else {
            return;
        };
        let start = std::time::Instant::now();
        let mut x = 1u64;
        while start.elapsed() < std::time::Duration::from_millis(50) {
            x = std::hint::black_box(x.wrapping_mul(6364136223846793005).wrapping_add(1));
        }
        let b = thread_used_ns().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        let c = thread_used_ns().unwrap();
        assert!(b - a >= 25_000_000, "spinning 50 ms ran {} ns", b - a);
        assert!(c - b < 25_000_000, "sleeping 50 ms ran {} ns", c - b);
    }
}
