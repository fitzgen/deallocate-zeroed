use super::*;

/// A wrapper around an allocator `A` that provides a default implementation of
/// `DeallocateZeroed` that simply forwards the pointer to `A::deallocate`.
pub struct DefaultDeallocateZeroed<A>(pub A);

unsafe impl<A> Allocator for DefaultDeallocateZeroed<A>
where
    A: Allocator,
{
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.0.allocate(layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.0.deallocate(ptr, layout);
    }
}

impl<T> DeallocateZeroed for DefaultDeallocateZeroed<T> where T: Allocator {}
