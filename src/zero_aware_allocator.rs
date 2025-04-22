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
use metadata::{BlockInfo, FreeList, LiveSet};

mod mutex;
pub use mutex::{LockingMechanism, Mutex, MutexGuard, SingleThreadedLockingMechanism};

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
/// type parameter. See the [`LockingMechanism`] trait for details.
#[derive(Default)]
pub struct ZeroAwareAllocator<A, L>
where
    A: Allocator,
    L: LockingMechanism,
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

    /// The set of live allocations we've handed out to the user from one of our
    /// free-lists.
    ///
    /// We maintain this set because the `Layout` that the block was originally
    /// allocated from `inner` with is not necessarily the same `Layout` that
    /// the user gave us when they asked for an already-zereod block of
    /// memory. Consider the following sequence of events, for example:
    ///
    /// * `deallocate_zeroed(0x12345678, size = 64KiB)`
    ///
    ///   Now we have an already-zeroed entry for this allocation in one of our
    ///   free-lists.
    ///
    /// * `allocate_zeroed(size = 62KiB) -> 0x12345678`
    ///
    ///   This new allocation's requested size isn't an exact match for our
    ///   free-list entry, but is close enough that it is worth accepting a
    ///   little bit of fragmentation slop rather than spending time zeroing
    ///   62KiB of memory.
    ///
    /// * `grow(0x12345678, old_size=62KiB, new_size=...)`
    ///
    ///   When the user grows their allocation of "62KiB", we need to realize
    ///   that it is actually 64KiB in size and act accordingly.
    ///
    /// It is critical that whenever we call into the `inner` allocator, we pass
    /// the allocation's actual, original `Layout`. Passing the wrong layout is
    /// a violation of the `Allocator` API's safety contract, so we could
    /// trigger UB if we aren't careful here.
    live_set: LiveSet,
}

impl<A, L> ZeroAwareAllocator<A, L>
where
    A: Allocator,
    L: LockingMechanism,
{
    /// Create a new `ZeroAwareAllocator` that wraps the given `inner`
    /// allocator.
    #[inline]
    pub const fn new(inner: A, lock: L) -> Self {
        let zeroed = Zeroed {
            align_classes: [const { FreeList::new() }; NUM_ALIGN_CLASSES],
            very_large_aligns: FreeList::new(),
            live_set: LiveSet::new(),
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
        debug_assert_ne!(layout.size(), 0);

        // The index of this layout's align-class, if we have one.
        let align_class = layout.align().ilog2() as usize;

        {
            let mut zeroed = self.zeroed.lock();
            let zeroed = &mut *zeroed;

            // Start at the layout's align-class and try to find a suitably-sized
            // block for the layout, moving to larger and larger align-classes as
            // necessary.
            let freelists = zeroed
                .align_classes
                .get_mut(align_class..NUM_ALIGN_CLASSES)
                .into_iter()
                .flat_map(|classes| classes.iter_mut())
                .chain(Some(&mut zeroed.very_large_aligns));

            for freelist in freelists {
                // Look for a block that satisfies this allocation layout. If we
                // find one, then deallocate our metadata with the underlying
                // allocator and return the block.
                if let Some(node) = freelist.remove(&layout) {
                    let ret = node.non_null_slice_ptr();

                    // Correctly sized.
                    debug_assert!(
                        ret.len() >= layout.size(),
                        "{ret:#p}'s size should be greater than or equal to user layout's size\n\
                         actual size        = {}\n\
                         user layout's size = {}",
                        ret.len(),
                        layout.size()
                    );
                    debug_assert!(
                        ret.len() >= node.layout().size(),
                        "{ret:#p}'s size should be greater than or equal to its original layout's size\n\
                         actual size            = {}\n\
                         original layout's size = {}",
                        ret.len(),
                        layout.size()
                    );

                    // Correctly aligned.
                    debug_assert_eq!(
                        ret.cast::<u8>().as_ptr() as usize % layout.align(),
                        0,
                        "{ret:#p} should be aligned to user layout's alignment of {:#x}",
                        layout.align()
                    );
                    debug_assert_eq!(
                        ret.cast::<u8>().as_ptr() as usize % node.layout().align(),
                        0,
                        "{ret:#p} should be aligned to user layout's alignment of {:#x}",
                        node.layout().align()
                    );

                    // Correctly zeroed.
                    debug_assert!({
                        let slice = unsafe {
                            core::slice::from_raw_parts(
                                node.ptr().as_ptr().cast_const(),
                                node.layout().size(),
                            )
                        };
                        slice.iter().all(|b| *b == 0)
                    });

                    zeroed.live_set.insert(node);
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
        // Correctly sized.
        debug_assert_ne!(layout.size(), 0);

        // Correctly aligned.
        debug_assert_eq!(ptr.as_ptr() as usize % layout.align(), 0);

        // Correctly zeroed.
        debug_assert!({
            let slice = core::slice::from_raw_parts(ptr.as_ptr().cast_const(), layout.size());
            slice.iter().all(|b| *b == 0)
        });

        let mut zeroed = self.zeroed.lock();
        let zeroed = &mut *zeroed;
        let node = match zeroed.live_set.remove(&ptr) {
            Some(node) => {
                debug_assert!(
                    node.layout().size() >= layout.size(),
                    "actual size should be greater than or equal to user's size\n\
                     actual size = {}\n\
                     user's size = {}",
                    node.layout().size(),
                    layout.size(),
                );

                // NB: the allocation might just happen to be aligned to the
                // user layout, in which case it may *not* have been the case
                // that the actual layout's alignment is greater than or equal
                // to the user layout's alignment.
                //
                // The pointer itself must be suitably aligned to the layout
                // either way, however.
                debug_assert_eq!(ptr.as_ptr() as usize % node.layout().align(), 0);

                // And any fragmentation we had accepted still be zeroed.
                debug_assert!({
                    let slice = core::slice::from_raw_parts(
                        ptr.add(layout.size()).as_ptr().cast_const(),
                        node.layout().size() - layout.size(),
                    );
                    slice.iter().all(|b| *b == 0)
                });

                node
            }
            None => match self.allocate_block_info(ptr, layout) {
                Ok(node) => node,
                // If the inner allocator fails to allocate the block info that
                // we need for bookkeeping, then we can't keep track of this
                // already-zeroed block, so simply eagerly return it to the
                // inner allocator.
                Err(_) => {
                    self.inner.deallocate(ptr, layout);
                    return;
                }
            },
        };

        // Get the appropriate freelist for this block's align-class.
        let align_class = node.layout().align().ilog2() as usize;
        let freelist = zeroed
            .align_classes
            .get_mut(align_class)
            .unwrap_or_else(|| &mut zeroed.very_large_aligns);

        // Insert the block into its freelist.
        freelist.insert(node);
    }

    /// Before a `grow` or `grow_zeroed`, check whether we have the pointer in
    /// our live-set. If it already satisfies the new size requirements, reuse
    /// it. Otherwise, remove it from our live-set, as we will need to pass it
    /// to the underlying allocator or re-allocate it.
    unsafe fn pre_grow(
        &self,
        ptr: NonNull<u8>,
        user_old_layout: Layout,
        new_layout: Layout,
    ) -> PreGrow {
        // Correctly sized.
        debug_assert_ne!(user_old_layout.size(), 0);

        // Correctly aligned.
        debug_assert_eq!(
            ptr.as_ptr() as usize % user_old_layout.align(),
            0,
            "{ptr:#p} should be aligned to user's layout's alignment of {:#x}",
            user_old_layout.align()
        );

        let mut zeroed = self.zeroed.lock();
        let zeroed = &mut *zeroed;

        let actual_old_layout = if let Some(node) = zeroed.live_set.find(&ptr) {
            debug_assert_eq!(node.ptr(), ptr);

            // Correctly sized.
            debug_assert!(node.layout().size() >= user_old_layout.size());

            // Correctly aligned.
            //
            // NB: it is not necessarily the case that the original layout's
            // align is greater than or equal to the user layout's align in the
            // case where we reused an allocation that happened to be aligned
            // greater than its original layout requested.
            debug_assert_eq!(
                ptr.as_ptr() as usize % node.layout().align(),
                0,
                "{ptr:#p} should be aligned to original layout's alignment of {:#x}",
                node.layout().align()
            );

            // If this is an allocation that was originally already zeroed,
            // and its original inner layout can already satisfy this new
            // layout, then we don't need to do anything further.
            if node.layout().size() >= new_layout.size()
                && node.layout().align() >= new_layout.align()
            {
                return PreGrow::Reuse {
                    ptr: NonNull::slice_from_raw_parts(ptr, node.layout().size()),
                    actual_layout: node.layout(),
                };
            }

            let actual_old_layout = node.layout();

            // We no longer need the live-set node. It is only required for
            // when we are handing out already-zeroed blocks that we
            // bookkeep internally and whose original layout might not match
            // what the user thinks its layout is. From this point on, this
            // allocation will either be fully managed by the inner
            // allocator (in the case of a successful `grow`) or will be
            // deallocated and replaced by a new already-zeroed allocation
            // (in the case of growth failure). In either case, we no longer
            // need a block-info node for this allocation.
            zeroed
                .live_set
                .remove(node)
                .expect("just found the node in the live-set, should still be there");
            self.deallocate_block_info(node);

            actual_old_layout
        } else {
            // This is not an allocation we are managing: use the user's
            // given old layout.
            user_old_layout
        };

        if new_layout.size() >= actual_old_layout.size() {
            PreGrow::DoGrow { actual_old_layout }
        } else {
            PreGrow::DoShrink { actual_old_layout }
        }
    }
}

enum PreGrow {
    /// The underlying inner allocation can be reused and does not need to be
    /// resized.
    Reuse {
        ptr: NonNull<[u8]>,
        actual_layout: Layout,
    },

    /// The underlying allocation cannot satisfy the growth request, we need to
    /// use the inner allocator to grow.
    DoGrow { actual_old_layout: Layout },

    /// The underlying allocation cannot satisfy the growth request due to
    /// less-than-requested alignment, but its actual size is *larger* than the
    /// new layout's size. Therefore, we need to use the inner allocator to
    /// shrink instead of grow.
    DoShrink { actual_old_layout: Layout },
}

impl<A, L> Drop for ZeroAwareAllocator<A, L>
where
    A: Allocator,
    L: LockingMechanism,
{
    fn drop(&mut self) {
        self.return_zeroed_memory_to_inner();
    }
}

unsafe impl<A, L> Allocator for ZeroAwareAllocator<A, L>
where
    A: Allocator,
    L: LockingMechanism,
{
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.inner
            .allocate(layout)
            .or_else(|_| self.allocate_already_zeroed(layout))
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, user_layout: Layout) {
        if user_layout.size() == 0 {
            self.inner.deallocate(ptr, user_layout);
            return;
        }

        let mut zeroed = self.zeroed.lock();
        let zeroed = &mut *zeroed;
        let actual_layout = if let Some(node) = zeroed.live_set.remove(&ptr) {
            let layout = node.layout();
            self.deallocate_block_info(node);
            layout
        } else {
            user_layout
        };
        self.inner.deallocate(ptr, actual_layout);
    }

    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return self.inner.allocate_zeroed(layout);
        }

        self.allocate_already_zeroed(layout)
            .or_else(|_| self.inner.allocate_zeroed(layout))
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        user_old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if user_old_layout.size() == 0 {
            return self.inner.grow(ptr, user_old_layout, new_layout);
        }

        let actual_old_layout = match self.pre_grow(ptr, user_old_layout, new_layout) {
            PreGrow::Reuse {
                ptr,
                actual_layout: _,
            } => {
                // No need to update our metadata here: the caller is
                // responsible for keeping track of the correct `Layout` and the
                // size of the actual underlying block didn't grow as we have
                // nothing to change.
                return Ok(ptr);
            }
            PreGrow::DoShrink { actual_old_layout } => {
                // No need to update any metadata here, we are no longer
                // managing this allocation, the inner allocator is.
                return self.inner.shrink(ptr, actual_old_layout, new_layout);
            }
            PreGrow::DoGrow { actual_old_layout } => actual_old_layout,
        };

        match self.inner.grow(ptr, actual_old_layout, new_layout) {
            Ok(ptr) => Ok(ptr),

            // If the inner allocator cannot grow this allocation, see if we
            // have an already-zeroed block that will work.
            Err(_) => {
                let new_ptr = self.allocate_already_zeroed(new_layout)?;
                debug_assert_ne!(new_ptr.cast::<u8>(), ptr);

                // Copy over the bytes from the old allocation to the new one.
                ptr::copy_nonoverlapping(
                    ptr.as_ptr().cast_const(),
                    new_ptr.cast::<u8>().as_ptr(),
                    // NB: we only need to copy the user's bytes, not the actual
                    // inner allocation's bytes.
                    user_old_layout.size(),
                );

                // The contract is that when `grow` succeeds, it takes ownership
                // of the old pointer, so we must deallocate the old allocation.
                self.inner.deallocate(ptr, actual_old_layout);

                Ok(new_ptr)
            }
        }
    }

    #[inline]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        user_old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if user_old_layout.size() == 0 {
            return self.inner.allocate_zeroed(new_layout);
        }

        let actual_old_layout = match self.pre_grow(ptr, user_old_layout, new_layout) {
            PreGrow::Reuse { ptr, actual_layout } => {
                // NB: we cannot assume that the fragmentation slop (if any) is
                // still zeroed, because we give out the whole block as a
                // `NonNull<[u8]>` slice to the user. Therefore, we have to
                // manually zero here.
                let slop = ptr.cast::<u8>().add(user_old_layout.size());
                let slop_len = actual_layout.size() - user_old_layout.size();
                slop.write_bytes(0, slop_len);

                // No need to update our metadata here: the caller is
                // responsible for keeping track of the correct `Layout` and the
                // size of the actual underlying block didn't grow os we have
                // nothing to change.
                return Ok(ptr);
            }
            PreGrow::DoShrink { actual_old_layout } => {
                let new_ptr = self.inner.shrink(ptr, actual_old_layout, new_layout)?;

                // We need to zero the range of bytes between the user's old
                // size and the new size, which should be within the shrunken
                // area.
                debug_assert!(new_layout.size() < actual_old_layout.size());
                debug_assert!(new_layout.size() >= user_old_layout.size());
                let to_zero = new_ptr.cast::<u8>().add(user_old_layout.size());
                let to_zero_len = new_layout.size() - user_old_layout.size();
                to_zero.write_bytes(0, to_zero_len);

                // No need to update our metadata here: we are no longer
                // managing this allocation, the inner allocator is.
                return Ok(new_ptr);
            }
            PreGrow::DoGrow { actual_old_layout } => actual_old_layout,
        };

        // NB: we could have enough capacity in the existing allocation, but
        // cannot reuse the allocation because it has the wrong alignment.
        // Therefore, it is not necessarily the case that the old layout is
        // smaller than the new layout and so we do a saturating subtraction
        // here.
        let bytes_to_zero = new_layout.size().saturating_sub(actual_old_layout.size());
        let bytes_to_copy = new_layout.size() - bytes_to_zero;

        // We can't split or grow blocks in place so if we don't want to use the
        // underlying allocator, we have to do a new zeroed allocation and copy
        // over the old data. As a heuristic, if that will end up copying way
        // more bytes than the new allocation would need to zero (assuming the
        // old allocation could be grown in place) then just defer to the
        // underlying allocator.
        if bytes_to_copy > 2usize.saturating_mul(bytes_to_zero) {
            if let Ok(p) = self.inner.grow_zeroed(ptr, user_old_layout, new_layout) {
                return Ok(p);
            }
        }

        let new = self.allocate_zeroed(new_layout)?;
        ptr::copy_nonoverlapping(
            ptr.as_ptr().cast_const(),
            new.cast().as_ptr(),
            user_old_layout.size(),
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
        if old_layout.size() == 0 {
            return self.inner.shrink(ptr, old_layout, new_layout);
        }

        // We can't split allocations ourselves, so just defer to the inner
        // allocator.

        let mut zeroed = self.zeroed.lock();
        let zeroed = &mut *zeroed;

        // Make sure that we are passing the original, underlying `Layout` to
        // the inner allocator. After shrinking, this pointer will no longer be
        // managed by us, so remove and deallocate its block-info node from the
        // live-set (if any).
        let old_layout = if let Some(node) = zeroed.live_set.remove(&ptr) {
            let actual_old_layout = node.layout();
            self.deallocate_block_info(node);
            actual_old_layout
        } else {
            old_layout
        };

        self.inner.shrink(ptr, old_layout, new_layout)
    }
}

impl<A, L> DeallocateZeroed for ZeroAwareAllocator<A, L>
where
    A: Allocator,
    L: LockingMechanism,
{
    unsafe fn deallocate_zeroed(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            self.inner.deallocate(ptr, layout);
            return;
        }

        self.deallocate_already_zeroed(ptr, layout);
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

    pub(super) struct ByAlignClass;
    pub(super) type FreeList = SplayTree<'static, ByAlignClass>;

    /// Comparison between two `BlockInfo`s.
    ///
    /// This is used for ordering blocks within a tree.
    impl<'a> TreeOrd<'a, ByAlignClass> for BlockInfo<'a> {
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
    impl<'a> TreeOrd<'a, ByAlignClass> for Layout {
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
        impl<'a> IntrusiveNode<'a> for ByAlignClass
        where
            type Elem = BlockInfo<'a>,
            node = node;
    }

    pub(super) struct ByPointer;
    pub(super) type LiveSet = SplayTree<'static, ByPointer>;

    /// Comparison between two `BlockInfo`s by pointer.
    ///
    /// This is used for ordering blocks within a tree.
    impl<'a> TreeOrd<'a, ByPointer> for BlockInfo<'a> {
        fn tree_cmp(&self, other: &'a BlockInfo<'a>) -> Ordering {
            Ord::cmp(&self.ptr, &other.ptr)
        }
    }

    /// Comparison between a pointer and a `BlockInfo`.
    ///
    /// Used when searching the live-set for a particular allocation.
    impl<'a> TreeOrd<'a, ByPointer> for NonNull<u8> {
        fn tree_cmp(&self, other: &'a BlockInfo<'a>) -> Ordering {
            Ord::cmp(self, &other.ptr)
        }
    }

    intrusive_splay_tree::impl_intrusive_node! {
        impl<'a> IntrusiveNode<'a> for ByPointer
        where
            type Elem = BlockInfo<'a>,
            node = node;
    }
}
