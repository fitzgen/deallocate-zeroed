//! The zero-aware memory allocator.
//!
//! Zeroed memory is organized into a 2-layer freelist to facillitate fast and
//! easy allocation:
//!
//! 1. By alignment, in a fixed-size array.
//! 2. By size, in a dynamically-sized splay tree.
//!
//! When allocating a block, we use a hybrid best- and first-fit search:
//!
//! * First, we prioritize memory blocks with the minimum-satisfying alignment
//!   before considering blocks with alignment greater than requested. This
//!   helps us avoid "wasting" a rare, very-aligned block on an allocation that
//!   doesn't have any particular alignment requirements.
//!
//!   This prioritization is implemented via a loop over each entry in the
//!   fixed-size array (or "align-class") starting with the align-class that
//!   exactly matches the requested allocation's alignment and moving towards
//!   larger and larger alignments until we find a block that satisfies the
//!   allocation constraints or we fail to allocate.
//!
//! * Second, we search for memory blocks that will fit the requested allocation
//!   size, without introducing too much internal fragmentation. We care about
//!   fragmentation because we do not split blocks. Splitting blocks would
//!   require either that the underlying allocator support block splitting
//!   (which it might not and the `Allocator` trait doesn't have methods for
//!   deallocating a split-off chunk of previously allocated block anyways) or
//!   else additional bookkeeping on our part to track when a block is split or
//!   not and whether it can be merged again and then returned to the underlying
//!   allocator.

use core::ptr;

use super::*;
use metadata::{BlockInfo, FreeList};

mod mutex;
use mutex::*;
pub use mutex::{Lock, SingleThreadedLock};

/// The maximum alignment for which we have an align-class.
const MAX_ALIGN_WITH_CLASS: usize = 4096;

/// The number of align-classes we have.
const NUM_ALIGN_CLASSES: usize = MAX_ALIGN_WITH_CLASS.ilog2() as usize + 1;

/// We allow satisfying an allocation with a block that is only up to
///
/// ```ignore
/// 1 / ACCEPTABLE_WASTE_DIVISOR
/// ```
///
/// bytes larger than the requested allocation's size.
///
/// This balances fragmentation (because we do not split blocks) with allocation
/// efficiency and how deep in a freelist we will keep searching despite having
/// seen a block that technically could satisfy the allocation (i.e. best- vs
/// first-fit). Making this value larger will decrease fragmentation but
/// increase time spent searching freelists and the probability we will ask the
/// underlying allocator for a new zeroed block, rather than reusing an
/// already-zeroed block; making it smaller has the opposite effects.
///
/// NB: keep this a power of two so that the compiler can strength-reduce the
/// division into a shift.
const ACCEPTABLE_WASTE_DIVISOR: usize = 8;

/// A memory allocator that keeps track of already-zeroed memory blocks.
///
/// This lets applications move zeroing off of the `allocate_zeroed` critical
/// path. Instead they can, for example, bulk-zero memory blocks in the
/// background before returning them to the allocator.
///
/// This allocator wraps an underinglying, inner allocator of type `A`, layering
/// the bookkeeping of already-zeroed blocks on top of it.
///
/// Because this crate is `no_std` and does not assume the presence of an
/// operating system, you must provide your own locking mechanism via the `L`
/// type parameter. See the [`Lock`] trait for details.
#[derive(Default)]
pub struct ZeroAwareAllocator<A, L>
where
    A: Allocator,
    L: Lock,
{
    /// The underlying allocator.
    inner: A,

    /// Metadata keeping track of already-zeroed memory.
    zeroed: Mutex<Zeroed, L>,
}

#[derive(Default)]
struct Zeroed {
    /// Already-zeroed memory blocks, organized into align-classes.
    align_classes: [FreeList; NUM_ALIGN_CLASSES],

    /// A fallback freelist for already-zeroed memory blocks that have alignment
    /// larger than we have an align-class for.
    very_large_aligns: FreeList,
}

impl<A, L> ZeroAwareAllocator<A, L>
where
    A: Allocator,
    L: Lock,
{
    /// Create a new `ZeroAwareAllocator` that wraps the given `inner`
    /// allocator.
    #[inline]
    pub const fn new(inner: A, lock: L) -> Self {
        let zeroed = Zeroed {
            align_classes: [const { FreeList::new() }; NUM_ALIGN_CLASSES],
            very_large_aligns: FreeList::new(),
        };
        let zeroed = Mutex::new(zeroed, lock);
        ZeroAwareAllocator { inner, zeroed }
    }

    /// Get a shared reference to the inner allocator.
    #[inline]
    pub fn inner(&self) -> &A {
        &self.inner
    }

    /// Get an exclusive reference to the inner allocator.
    #[inline]
    pub fn inner_mut(&mut self) -> &mut A {
        &mut self.inner
    }

    /// Return all of the known-zeroed memory blocks to the underlying
    /// allocator.
    pub fn return_zeroed_memory_to_inner(&mut self) {
        let mut zeroed = self.zeroed.lock();
        let zeroed = &mut *zeroed;

        for freelist in zeroed
            .align_classes
            .iter_mut()
            .chain(Some(&mut zeroed.very_large_aligns))
        {
            while let Some(node) = freelist.pop_root() {
                let block_ptr = node.ptr();
                let block_layout = node.layout();

                // Safety: the node has been removed from the freelist, was
                // allocated with `self.inner`, and is currently allocated.
                unsafe {
                    self.deallocate_block_info(node);
                }

                // Safety: the block is currently allocated (from the inner
                // allocator's point of view), has the associated layout, and
                // nothing else is referencing this block anymore now that its
                // node has been deallocated.
                unsafe { self.inner.deallocate(block_ptr, block_layout) };
            }
        }
    }

    /// Allocate a new `BlockInfo` for our internal bookkeeping for the
    /// given already-zeroed block described by `ptr` and `layout`.
    fn allocate_block_info(
        &self,
        ptr: NonNull<u8>,
        layout: Layout,
    ) -> Result<&'static BlockInfo<'static>, AllocError> {
        let node_ptr = self.inner.allocate(Layout::new::<BlockInfo<'_>>())?;
        let node_ptr = node_ptr.cast::<BlockInfo<'_>>();
        // Safety: `node_ptr` is valid for writes, is properly aligned, and is
        // valid for conversion to a reference.
        unsafe {
            node_ptr.write(BlockInfo::new(ptr, layout));
            Ok(node_ptr.as_ref())
        }
    }

    /// Deallocate a `BlockInfo` freelist node.
    ///
    /// ### Safety
    ///
    /// * The node must not be in a freelist or otherwise referenced by anything
    ///   else.
    ///
    /// * The node must have been allocated with `self.inner`.
    ///
    /// * The node must be currently allocated.
    unsafe fn deallocate_block_info(&self, node: &'static BlockInfo<'static>) {
        self.inner
            .deallocate(NonNull::from(node).cast(), Layout::new::<BlockInfo<'_>>());
    }

    /// Attempt to allocate an already-zeroed block of the given layout.
    ///
    /// Does not fallback to the inner allocator upon allocation failure.
    #[inline]
    fn allocate_already_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(NonNull::from(&[]));
        }

        // The index of this layout's align-class, if we have one.
        let align_class = layout.align().ilog2() as usize;

        {
            let mut zeroed = self.zeroed.lock();
            let zeroed = &mut *zeroed;

            // Start at the layout's align-class and try to find a suitably-sized
            // block for the layout, moving to larger and larger align-classes as
            // necessary.
            let freelists = zeroed.align_classes[align_class..NUM_ALIGN_CLASSES]
                .iter_mut()
                .chain(Some(&mut zeroed.very_large_aligns));

            for freelist in freelists {
                // Look for a block that satisfies this allocation layout. If we
                // find one, then deallocate our metadata with the underlying
                // allocator and return the block.
                if let Some(node) = freelist.remove(&layout) {
                    let ret = node.non_null_slice_ptr();

                    // Safety: the node was allocated from `self.inner`, is
                    // currently allocated, and was removed from its freelist.
                    unsafe {
                        self.deallocate_block_info(node);
                    }

                    return Ok(ret);
                }
            }
        }

        // Failed to find an already-zeroed block that satisfied the requested
        // layout.
        Err(AllocError)
    }

    #[inline]
    unsafe fn deallocate_already_zeroed(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            return;
        }

        match self.allocate_block_info(ptr, layout) {
            // If the inner allocator fails to allocate the node that we need
            // for bookkeeping, then we can't keep track of this already-zeroed
            // block, so simply eagerly return it to the inner allocator.
            Err(_) => self.inner.deallocate(ptr, layout),

            Ok(node) => {
                let mut zeroed = self.zeroed.lock();
                let zeroed = &mut *zeroed;

                // Get the appropriate freelist for this block's align-class.
                let align_class = layout.align().ilog2() as usize;
                let freelist = zeroed
                    .align_classes
                    .get_mut(align_class)
                    .unwrap_or_else(|| &mut zeroed.very_large_aligns);

                // Insert the block into its freelist.
                freelist.insert(node);
            }
        }
    }
}

impl<A, L> Drop for ZeroAwareAllocator<A, L>
where
    A: Allocator,
    L: Lock,
{
    fn drop(&mut self) {
        self.return_zeroed_memory_to_inner();
    }
}

unsafe impl<A, L> Allocator for ZeroAwareAllocator<A, L>
where
    A: Allocator,
    L: Lock,
{
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.inner
            .allocate(layout)
            .or_else(|_| self.allocate_already_zeroed(layout))
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.inner.deallocate(ptr, layout);
    }

    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.allocate_already_zeroed(layout)
            .or_else(|_| self.inner.allocate_zeroed(layout))
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        self.inner.grow(ptr, old_layout, new_layout).or_else(|_| {
            let new = self.allocate_already_zeroed(new_layout)?;
            ptr::copy_nonoverlapping(
                ptr.as_ptr().cast_const(),
                new.cast().as_ptr(),
                old_layout.size(),
            );
            Ok(new)
        })
    }

    #[inline]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let bytes_to_zero = new_layout.size() - old_layout.size();
        let bytes_to_copy = new_layout.size() - bytes_to_zero;

        // We can't split or grow blocks in place so if we don't want to use the
        // underlying allocator, we have to do a new zeroed allocation and copy
        // over the old data. As a heuristic, if that will end up copying way
        // more bytes than the new allocation would need to zero (assuming the
        // old allocation could be grown in place) then just defer to the
        // underlying allocator.
        if bytes_to_copy > 2usize.saturating_mul(bytes_to_zero) {
            if let Ok(p) = self.inner.grow_zeroed(ptr, old_layout, new_layout) {
                return Ok(p);
            }
        }

        let new = self.allocate_zeroed(new_layout)?;
        ptr::copy_nonoverlapping(
            ptr.as_ptr().cast_const(),
            new.cast().as_ptr(),
            old_layout.size(),
        );
        Ok(new)
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // We can't split allocations ourselves, so just defer to the inner
        // allocator.
        self.inner.shrink(ptr, old_layout, new_layout)
    }
}

impl<A, L> DeallocateZeroed for ZeroAwareAllocator<A, L>
where
    A: Allocator,
    L: Lock,
{
    unsafe fn deallocate_zeroed(&self, pointer: NonNull<u8>, layout: Layout) {
        self.deallocate_already_zeroed(pointer, layout);
    }
}

mod metadata {
    use super::*;
    use core::cmp::Ordering;
    use intrusive_splay_tree::{Node, SplayTree, TreeOrd};

    /// Metadata about an already-zeroed block of memory, allocated from our
    /// underlying allocator.
    ///
    /// Note: the `'a` lifetime is used internally to this module as much as
    /// possible to keep things free of `unsafe` when possible, but outside this
    /// module is always erased to `'static` and the lifetimes are manually
    /// managed.
    #[derive(Debug)]
    pub(super) struct BlockInfo<'a> {
        ptr: NonNull<u8>,
        layout: Layout,
        node: Node<'a>,
    }

    impl<'a> BlockInfo<'a> {
        pub(super) unsafe fn new(ptr: NonNull<u8>, layout: Layout) -> Self {
            debug_assert!(
                core::slice::from_raw_parts(ptr.as_ptr(), layout.size())
                    .iter()
                    .all(|b| *b == 0),
                "supposedly already-zeroed block contains non-zero memory"
            );
            BlockInfo {
                ptr,
                layout,
                node: Node::default(),
            }
        }

        fn size(&self) -> usize {
            self.layout.size()
        }

        fn align(&self) -> usize {
            1 << (self.ptr.as_ptr() as usize).trailing_zeros()
        }

        pub(super) fn ptr(&self) -> NonNull<u8> {
            self.ptr
        }

        pub(super) fn non_null_slice_ptr(&self) -> NonNull<[u8]> {
            NonNull::slice_from_raw_parts(self.ptr, self.size())
        }

        pub(super) fn layout(&self) -> Layout {
            self.layout
        }
    }

    /// Comparison between two `BlockInfo`s.
    ///
    /// This is used for ordering blocks within a tree.
    impl<'a> TreeOrd<'a, BlockInfo<'a>> for BlockInfo<'a> {
        fn tree_cmp(&self, other: &'a BlockInfo<'a>) -> Ordering {
            // Compare first by size, since that is the first hard constraint we
            // must satisfy and property we query for.
            self.size()
                .cmp(&other.size())
                // Then by alignment, since that is the other hard constraint.
                .then_with(|| {
                    let self_align = (self.ptr.as_ptr() as usize).trailing_zeros();
                    let other_align = (other.ptr.as_ptr() as usize).trailing_zeros();
                    self_align.cmp(&other_align)
                })
                // And finally by address. This final tie breaker should
                // generally improve spatial locality between a sequence of
                // allocations.
                .then_with(|| {
                    let self_addr = self.ptr.as_ptr() as usize;
                    let other_addr = other.ptr.as_ptr() as usize;
                    self_addr.cmp(&other_addr)
                })
        }
    }

    /// Comparison between a `Layout` and a `BlockInfo`.
    ///
    /// This is used when searching for a block to satisfy a requested
    /// allocation.
    impl<'a> TreeOrd<'a, BlockInfo<'a>> for Layout {
        fn tree_cmp(&self, block: &'a BlockInfo<'a>) -> Ordering {
            // Compare sizes. Allow for some fragmentation, but not too much,
            // because we can't split blocks ourselves. See the doc comment for
            // `ACCEPTABLE_WASTE_DIVISOR` for more info.
            let by_size = match self.size().cmp(&block.size()) {
                Ordering::Greater => Ordering::Greater,
                Ordering::Equal => Ordering::Equal,
                Ordering::Less => {
                    let acceptable_waste = self.size() / ACCEPTABLE_WASTE_DIVISOR;
                    let potential_waste = block.size() - self.size();
                    if potential_waste <= acceptable_waste {
                        Ordering::Equal
                    } else {
                        Ordering::Less
                    }
                }
            };
            by_size
                // Compare alignments. If the block matches the requested
                // alignment at all, even if it is actually over-aligned, then
                // return `Ordering::Equal` to signal that it can satisfy the
                // allocation.
                .then_with(|| match self.align().cmp(&block.align()) {
                    Ordering::Less | Ordering::Equal => Ordering::Equal,
                    Ordering::Greater => Ordering::Greater,
                })
        }
    }

    intrusive_splay_tree::impl_intrusive_node! {
        impl<'a> IntrusiveNode<'a> for BlockInfo<'a>
        where
            type Elem = BlockInfo<'a>,
            node = node;
    }

    pub(super) type FreeList = SplayTree<'static, BlockInfo<'static>>;
}
