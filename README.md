A extension trait for allocators to provide a method for deallocating
already-zeroed memory.

## Why?

By providing a method for deallocating already-zeroed memory, we allow
applications to tell the allocator which free memory is already zeroed. This
means that the allocator can often avoid zeroing memory on the allocation
critical path, and the zeroing can instead be performed in a background task
just before deallocation.

## `ZeroAwareAllocator`

When the `zero_aware_allocator` cargo feature is enabled, this crate provides a
`ZeroAwareAllocator` type. This is a layered allocator, wrapping an innner
allocator and adding the tracking of already-zeroed memory on top of it.

## Using Nightly Rust's Unstable `feature(allocator_api)`

By default, this crate uses the `allocator_api2` crate to polyfill nightly
Rust's `feature(allocator_api)`. You can instead use the nightly
`feature(allocator_api)` by disabling the `allocator_api2` cargo feature and
enabling the `allocator_api` feature. Note that the nightly feature is unstable
and may break semver.
