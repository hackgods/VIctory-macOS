//! Locks taken by trying a while before sleeping (`VICTORY_SPIN_LOCKS=1`;
//! off by default): the locks title threads recording lists share with
//! each other and the heap sync for a few microseconds at a time (a host
//! heap's slot locks, its held resources, the bindless census, the retired
//! resources). A thread that finds such a lock taken sleeps at once with a
//! plain `Mutex::lock`, and under Wine a sleep and its wake cost far more
//! than the hold: in the five-thread list bench a ~6 us hold turned into
//! ~69 us waits, the lock handed from thread to thread. Trying again up to
//! [`SPINS`] times first lets the holder finish; the lock is then taken as
//! it would be, with nothing else changed. Based on halfwave50's spin-first
//! heap locks (VIctory-macOS-halfwave50 main-mac, descriptor/slots.rs
//! `locked_as`, SPINS = 128): 16,640 sleeps on the census lock per 49,152
//! slots without spinning, 331 with, in their Wine test.

use std::sync::{Mutex, MutexGuard, TryLockError};

/// Tries before a lock sleeps.
pub(crate) const SPINS: u32 = 128;

#[cfg(test)]
pub(crate) static TEST_ON: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Whether locks spin before they sleep (`VICTORY_SPIN_LOCKS=1`).
pub(crate) fn on() -> bool {
    #[cfg(test)]
    match TEST_ON.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_SPIN_LOCKS"))
}

/// `mutex` locked: with the switch, tried up to [`SPINS`] times before the
/// thread sleeps for it.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    if on() {
        for _ in 0..SPINS {
            match mutex.try_lock() {
                Ok(guard) => return guard,
                Err(TryLockError::Poisoned(e)) => return e.into_inner(),
                Err(TryLockError::WouldBlock) => std::hint::spin_loop(),
            }
        }
    }
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    /// A lock another thread holds briefly is taken by spinning, one held
    /// long by sleeping, and either way only once the holder lets go.
    #[test]
    fn a_spinning_lock_waits_for_its_holder() {
        super::TEST_ON.store(2, std::sync::atomic::Ordering::Relaxed);
        let mutex = std::sync::Mutex::new(0u32);
        for hold_ms in [0u64, 20] {
            let (held_tx, held_rx) = std::sync::mpsc::channel();
            std::thread::scope(|s| {
                s.spawn(|| {
                    let mut guard = super::lock(&mutex);
                    held_tx.send(()).unwrap();
                    std::thread::sleep(std::time::Duration::from_millis(hold_ms));
                    *guard += 1;
                });
                held_rx.recv().unwrap();
                let before = *super::lock(&mutex);
                assert!(before >= 1, "taken only after the holder let go");
            });
        }
        assert_eq!(*mutex.lock().unwrap(), 2);
        super::TEST_ON.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}
