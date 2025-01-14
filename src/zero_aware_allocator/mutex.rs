//! Provides a mutex container type similar to `std::sync::Mutex<T>` but
//! parameterized over any type `L` that implements the `Lock` trait, for better
//! no-std support.

use core::{
    cell::{Cell, UnsafeCell},
    ops::{Deref, DerefMut},
};

/// A trait for providing mutual exclusion.
///
/// If you do not need to use the allocator, and collections using it, in a
/// multi-threaded environment, you may use [`SingleThreadedLock`], which is the
/// moral equivalent of a `RefCell`.
///
/// # Safety
///
/// * If the implementation type is `Sync`, then an allocator using this locking
///   mechanism will be `Sync`, and therefore this method must provide actual
///   mutual exclusion and prevent against unsynchronized accesses.
///
/// * Even in single-threaded contexts, where real synchronization is not
///   required, this type must prevent recursive locking and re-entering the
///   lock when it is already held. The prevention may be a panic, abort,
///   infinite loop, or etc...
pub unsafe trait Lock {
    /// Lock this mutex.
    ///
    /// If it is already locked, this must result in a panic, abort, infinite
    /// loop, or etc... and locking must not succeed.
    fn lock(&self);

    /// Unlock this mutex.
    fn unlock(&self);
}

/// A single-threaded implementation of [`Lock`].
///
/// This is effectively a `RefCell`. It allows using the `ZeroAwareAllocator` in
/// single-threaded scenarios.
#[derive(Debug)]
pub struct SingleThreadedLock {
    locked: Cell<bool>,
}

unsafe impl Lock for SingleThreadedLock {
    #[inline]
    fn lock(&self) {
        assert!(!self.locked.get());
        self.locked.set(true);
    }

    #[inline]
    fn unlock(&self) {
        assert!(self.locked.get());
        self.locked.set(false);
    }
}

impl Default for SingleThreadedLock {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl SingleThreadedLock {
    /// Construct a new `SingleThreadedLock`.
    #[inline]
    pub fn new() -> Self {
        SingleThreadedLock {
            locked: Cell::new(false),
        }
    }
}

#[derive(Default)]
pub(super) struct Mutex<T, L> {
    lock: L,
    value: UnsafeCell<T>,
}

// Safety: if `T` and `L` can be sent between threads, then the mutex can as
// well. The API, implementation, and borrow checker do not allow for
// unsynchronized accesses in the face of sending these across threads.
unsafe impl<T, L> Send for Mutex<T, L>
where
    T: Send,
    L: Send,
{
}

// Safety: upheld by the `Lock` trait's implementation contract.
//
// Additionally, `T` must be `Send` because locking a mutex from another thread
// and getting a mutex guard allows getting `&mut T`, which can be used to
// `mem::replace()` the `T`, effectively sending it between threads.
unsafe impl<T, L> Sync for Mutex<T, L>
where
    T: Send,
    L: Sync + Lock,
{
}

impl<T, L> Mutex<T, L>
where
    L: Lock,
{
    pub(super) const fn new(value: T, lock: L) -> Self {
        let value = UnsafeCell::new(value);
        Mutex { lock, value }
    }

    pub(super) fn lock(&self) -> MutexGuard<'_, T, L> {
        self.lock.lock();
        MutexGuard { mutex: self }
    }
}

pub(super) struct MutexGuard<'a, T, L>
where
    L: Lock,
{
    mutex: &'a Mutex<T, L>,
}

impl<'a, T, L> Drop for MutexGuard<'a, T, L>
where
    L: Lock,
{
    fn drop(&mut self) {
        self.mutex.lock.unlock();
    }
}

impl<T, L> Deref for MutexGuard<'_, T, L>
where
    L: Lock,
{
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.mutex.value.get() }
    }
}

impl<T, L> DerefMut for MutexGuard<'_, T, L>
where
    L: Lock,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.mutex.value.get() }
    }
}
