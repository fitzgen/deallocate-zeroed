#![doc = include_str!("../README.md")]
#![no_std]
#![deny(missing_docs)]
#![cfg_attr(feature = "allocator_api", feature(allocator_api))]

use cfg_if::cfg_if;
use core::{alloc::Layout, ptr::NonNull};

mod default;
pub use default::DefaultDeallocateZeroed;

cfg_if! {
    if #[cfg(feature = "zero_aware_allocator")] {
        mod zero_aware_allocator;
        pub use zero_aware_allocator::{Lock, SingleThreadedLock, ZeroAwareAllocator};
    }
}

cfg_if! {
    if #[cfg(feature = "allocator_api")] {
        pub use core::alloc::{AllocError, Allocator};
    } else if #[cfg(feature = "allocator_api2")] {
        pub use allocator_api2::alloc::{AllocError, Allocator};
    } else {
        compile_error!("Must enable one of the `allocator_api` or `allocator_api2` cargo features");
    }
}

/// A trait for allocators that support deallocating already-zeroed memory.
pub trait DeallocateZeroed: Allocator {
    /// Deallocate already-zeroed memory.
    ///
    /// # Safety
    ///
    /// The memory block pointed to by `pointer` must be zeroed.
    ///
    /// Additionally, this method inherits all of
    /// [`core::alloc::Allocator::deallocate`]'s safety requirements.
    #[inline]
    unsafe fn deallocate_zeroed(&self, pointer: NonNull<u8>, layout: Layout) {
        self.deallocate(pointer, layout);
    }
}
