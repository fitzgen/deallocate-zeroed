//! Shared fuzzing and testing infrastructure for `deallocate_zeroed`.

#![feature(allocator_api)]

use deallocate_zeroed::{
    Allocator, DeallocateZeroed, LockingMechanism, SingleThreadedLockingMechanism,
    ZeroAwareAllocator,
};
use mutatis::{mutators as m, DefaultMutate, Generate, Mutate};
use std::{collections::BTreeMap, mem, ptr::NonNull};

/// The layout of a test allocation.
//
// Note: it is easier to define our own layout type here than to reuse
// `std::alloc::Layout` because we want to define a default mutator for `Layout`
// but trait orphan rules make that impossible.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    size: usize,
    align: usize,
}

impl Default for Layout {
    fn default() -> Self {
        Self { size: 0, align: 1 }
    }
}

impl Layout {
    /// Create a new `Layout` from the given size and alignment.
    pub fn new(size: usize, align: usize) -> Option<Self> {
        let layout = std::alloc::Layout::from_size_align(size, align).ok()?;
        Some(Layout {
            size: layout.size(),
            align: layout.align(),
        })
    }

    /// Like `Layout::new(...).unwrap()`.
    pub fn unwrap_new(size: usize, align: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(size, align)
            .expect("Layout::unwrap_new on bad size/align");
        Layout {
            size: layout.size(),
            align: layout.align(),
        }
    }

    fn alloc_layout(&self) -> std::alloc::Layout {
        std::alloc::Layout::from_size_align(self.size, self.align)
            .expect("should have a valid size and align")
    }
}

impl DefaultMutate for Layout {
    type DefaultMutate = LayoutMutator;
}

/// A mutator for `Layout`s with configurable maximums for size and alignment.
#[derive(Debug)]
pub struct LayoutMutator {
    pub max_size: usize,
    pub max_align: usize,
}

impl Default for LayoutMutator {
    fn default() -> Self {
        Self {
            max_size: 4096,
            max_align: 4096,
        }
    }
}

fn round_down_to_pow2(x: usize) -> usize {
    if x == 0 {
        1
    } else {
        1 << (mem::size_of::<usize>() * 8 - (x.leading_zeros() as usize))
    }
}

fn max_size_for_align(align: usize) -> usize {
    isize::MAX as usize + 1 - align
}

fn max_align_for_size(size: usize) -> usize {
    if size >= (isize::MAX as usize) {
        1
    } else {
        round_down_to_pow2((isize::MAX as usize) - size)
    }
}

impl Mutate<Layout> for LayoutMutator {
    fn mutate(
        &mut self,
        c: &mut mutatis::Candidates<'_>,
        layout: &mut Layout,
    ) -> mutatis::Result<()> {
        // Mutate size.
        c.mutation(|ctx| {
            let max_size = if ctx.shrink() {
                layout.size
            } else {
                std::cmp::min(self.max_size, max_size_for_align(layout.align))
            };
            layout.size = ctx.rng().gen_index(max_size + 1).unwrap();
            Ok(())
        })?;

        // Mutate alignment.
        c.mutation(|ctx| {
            let max_align_log2 = if ctx.shrink() {
                layout.align.trailing_zeros() as usize
            } else {
                std::cmp::min(
                    self.max_align.trailing_zeros() as usize,
                    max_align_for_size(layout.size).trailing_zeros() as usize,
                )
            };
            let align_log2 = ctx.rng().gen_index(max_align_log2 + 1).unwrap();
            layout.align = 1 << align_log2;
            Ok(())
        })?;

        Ok(())
    }
}

impl Generate<Layout> for LayoutMutator {
    fn generate(&mut self, context: &mut mutatis::Context) -> mutatis::Result<Layout> {
        let size = m::range(0..=self.max_size).generate(context)?;
        let align = m::range(1..=self.max_align).generate(context)?;
        let align = round_down_to_pow2(align);
        Ok(Layout { size, align })
    }
}

/// A test operation.
#[derive(Clone, Debug, Mutate)]
pub enum Op {
    Alloc { id: u32, layout: Layout },
    Dealloc { id: u32 },
    AllocZeroed { id: u32, layout: Layout },
    Grow { id: u32, layout: Layout },
    GrowZeroed { id: u32, layout: Layout },
    Shrink { id: u32, layout: Layout },
    DeallocZeroed { id: u32 },
}

impl Generate<Op> for OpMutator {
    fn generate(&mut self, ctx: &mut mutatis::Context) -> mutatis::Result<Op> {
        let choices: &[fn(&mut mutatis::Context) -> mutatis::Result<Op>] = &[
            |ctx| {
                Ok(Op::Alloc {
                    id: ctx.rng().gen_u32(),
                    layout: m::default::<Layout>().generate(ctx)?,
                })
            },
            |ctx| {
                Ok(Op::Dealloc {
                    id: ctx.rng().gen_u32(),
                })
            },
            |ctx| {
                Ok(Op::AllocZeroed {
                    id: ctx.rng().gen_u32(),
                    layout: m::default::<Layout>().generate(ctx)?,
                })
            },
            |ctx| {
                Ok(Op::Grow {
                    id: ctx.rng().gen_u32(),
                    layout: m::default::<Layout>().generate(ctx)?,
                })
            },
            |ctx| {
                Ok(Op::GrowZeroed {
                    id: ctx.rng().gen_u32(),
                    layout: m::default::<Layout>().generate(ctx)?,
                })
            },
            |ctx| {
                Ok(Op::Shrink {
                    id: ctx.rng().gen_u32(),
                    layout: m::default::<Layout>().generate(ctx)?,
                })
            },
            |ctx| {
                Ok(Op::DeallocZeroed {
                    id: ctx.rng().gen_u32(),
                })
            },
        ];

        let f = ctx.rng().choose(choices).unwrap();
        f(ctx)
    }
}

/// A sequence of test operations to perform.
#[derive(Clone, Debug, Default)]
pub struct Ops {
    ops: Vec<Op>,
}

impl DefaultMutate for Ops {
    type DefaultMutate = OpsMutator;
}

#[derive(Default)]
pub struct OpsMutator;

impl Mutate<Ops> for OpsMutator {
    fn mutate(&mut self, c: &mut mutatis::Candidates<'_>, ops: &mut Ops) -> mutatis::Result<()> {
        // Completely random mutations on a single-element basis.
        m::default::<Vec<Op>>().mutate(c, &mut ops.ops)?;

        fn alloc_positions_and_ids(ops: &Ops) -> impl Iterator<Item = (usize, u32)> + '_ {
            ops.ops.iter().enumerate().filter_map(|(i, op)| match op {
                Op::Alloc { id, .. } | Op::AllocZeroed { id, .. } => Some((i, *id)),
                _ => None,
            })
        }

        // Retarget an operation to an existing `id`.
        c.mutation(|ctx| {
            let num_allocs = alloc_positions_and_ids(ops).count();
            if let Some(alloc_index) = ctx.rng().gen_index(num_allocs) {
                let (_, new_id) = alloc_positions_and_ids(ops).nth(alloc_index).unwrap();
                let op_index = ctx.rng().gen_index(ops.ops.len()).unwrap();
                match &mut ops.ops[op_index] {
                    Op::Alloc { id, .. }
                    | Op::Dealloc { id }
                    | Op::AllocZeroed { id, .. }
                    | Op::Grow { id, .. }
                    | Op::GrowZeroed { id, .. }
                    | Op::Shrink { id, .. }
                    | Op::DeallocZeroed { id } => {
                        *id = new_id;
                    }
                }
            }
            Ok(())
        })?;

        // Deallocate an existing allocation.
        if !c.shrink() {
            c.mutation(|ctx| {
                let num_allocs = alloc_positions_and_ids(ops).count();
                if let Some(alloc_index) = ctx.rng().gen_index(num_allocs) {
                    let (op_index, id) = alloc_positions_and_ids(ops).nth(alloc_index).unwrap();
                    let dealloc_op = if ctx.rng().gen_bool() {
                        Op::Dealloc { id }
                    } else {
                        Op::DeallocZeroed { id }
                    };
                    let dealloc_index =
                        op_index + 1 + ctx.rng().gen_index(ops.ops.len() - op_index).unwrap();
                    ops.ops.insert(dealloc_index, dealloc_op);
                }
                Ok(())
            })?;
        }

        // Resize an existing allocation.
        if !c.shrink() {
            c.mutation(|ctx| {
                let num_allocs = alloc_positions_and_ids(ops).count();
                if let Some(alloc_index) = ctx.rng().gen_index(num_allocs) {
                    let (op_index, id) = alloc_positions_and_ids(ops).nth(alloc_index).unwrap();
                    let layout = m::default::<Layout>().generate(ctx)?;
                    let f = ctx
                        .rng()
                        .choose([
                            |id, layout| Op::Shrink { id, layout },
                            |id, layout| Op::Grow { id, layout },
                            |id, layout| Op::GrowZeroed { id, layout },
                        ])
                        .unwrap();
                    let resize_op = f(id, layout);
                    let resize_index =
                        op_index + 1 + ctx.rng().gen_index(ops.ops.len() - op_index).unwrap();
                    ops.ops.insert(resize_index, resize_op);
                }
                Ok(())
            })?;
        }

        Ok(())
    }
}

macro_rules! ensure {
    ( $cond:expr , $msg:expr $( , $args:expr )* $(,)? ) => {{
        let cond = $cond;
        if !cond {
            let msg = format!($msg $( , $args )* );
            let str_cond = stringify!($cond);
            return Err(format!("check failed: `{str_cond}`: {msg}"));
        }
    }};
}

impl Ops {
    /// Create a new `Ops` from the given test operations.
    pub fn new(ops: impl IntoIterator<Item = Op>) -> Self {
        let ops = ops.into_iter().collect();
        Ops { ops }
    }

    /// Run these test operations with the given allocation limit.
    pub fn run(&self, allocation_limit: usize) -> Result<(), String> {
        let allocator =
            ZeroAwareAllocator::new(std::alloc::System, SingleThreadedLockingMechanism::new());
        self.run_with_allocator(allocator, allocation_limit)
    }

    /// Run these test operations with the given allocator and allocation limit.
    pub fn run_with_allocator<A, L>(
        &self,
        allocator: ZeroAwareAllocator<A, L>,
        allocation_limit: usize,
    ) -> Result<(), String>
    where
        A: Allocator,
        L: LockingMechanism,
    {
        log::debug!("========== Running test operations ==========");

        let mut live = LiveMap::new(allocation_limit);

        // Fill an allocation with the given byte pattern.
        let fill = |ptr: NonNull<[u8]>, byte: u8| unsafe {
            ptr.cast::<u8>().write_bytes(byte, ptr.len());
        };

        // Deallocate the allocation with the given id.
        let dealloc = |id: u32, alloc: LiveAlloc| {
            log::debug!("deallocating id{id} -> {alloc:?}");
            unsafe {
                if alloc.zeroed {
                    allocator.deallocate_zeroed(alloc.ptr.cast(), alloc.layout);
                } else {
                    fill(alloc.ptr, FREE_POISON_PATTERN);
                    allocator.deallocate(alloc.ptr.cast(), alloc.layout);
                }
            }
        };

        // Assert that the given allocation is zeroed.
        let assert_zeroed = |ptr: NonNull<[u8]>| -> Result<(), String> {
            let slice = unsafe { ptr.as_ref() };
            ensure!(
                slice.iter().all(|b| *b == 0),
                "supposedly zeroed block of memory contains non-zero byte",
            );
            Ok(())
        };

        // Assert that the given allocation satisfies its requested layout.
        let assert_fits_layout =
            |ptr: NonNull<[u8]>, layout: std::alloc::Layout| -> Result<(), String> {
                ensure!(
                    layout.size() <= ptr.len(),
                    "actual allocated size is less than expected layout size",
                );
                ensure!(
                    layout.align().trailing_zeros()
                        <= (ptr.cast::<u8>().as_ptr() as usize).trailing_zeros(),
                    "actual allocated alignment is less than expected layout alignment",
                );
                Ok(())
            };

        // Assert that the given allocation is not overlapping with any other
        // live allocations.
        let assert_not_overlapping = |live: &LiveMap, ptr: NonNull<[u8]>| -> Result<(), String> {
            let ptr_start = ptr.cast::<u8>().as_ptr() as usize;
            let ptr_end = ptr_start + ptr.len();
            for other in live.map.values() {
                let other_start = other.ptr.cast::<u8>().as_ptr() as usize;
                let other_end = other_start + other.ptr.len();
                ensure!(
                    ptr_end <= other_start || other_end <= ptr_start,
                    "two distinct live allocations should never overlap",
                );
            }
            Ok(())
        };

        // Process a new non-zeroed allocation, checking properties and
        // inserting it into the live set.
        let new_alloc = |live: &mut LiveMap,
                         id: u32,
                         ptr: NonNull<[u8]>,
                         layout: std::alloc::Layout|
         -> Result<(), String> {
            if let Some(old_alloc) = live.remove(id) {
                dealloc(id, old_alloc);
            }

            log::debug!(
                "new non-zeroed allocation: id{id} -> {{ address: {ptr:p}, size: {}, layout: {layout:?} }}",
                ptr.len(),
            );

            assert_fits_layout(ptr, layout)?;
            assert_not_overlapping(&live, ptr)?;
            fill(ptr, LIVE_POISON_PATTERN);

            live.insert(
                id,
                LiveAlloc {
                    ptr,
                    layout,
                    zeroed: false,
                },
            );

            Ok(())
        };

        // Process a new zeroed allocation, checking properties and inserting it
        // into the live set.
        let new_alloc_zeroed = |live: &mut LiveMap,
                                id: u32,
                                ptr: NonNull<[u8]>,
                                layout: std::alloc::Layout|
         -> Result<(), String> {
            if let Some(old_alloc) = live.remove(id) {
                dealloc(id, old_alloc);
            }

            log::debug!(
                    "new zeroed allocation: id{id} -> {{ address: {ptr:p}, size: {}, layout: {layout:?} }}",
                    ptr.len(),
                );

            assert_fits_layout(ptr, layout)?;
            assert_not_overlapping(&live, ptr)?;
            assert_zeroed(ptr)?;

            live.insert(
                id,
                LiveAlloc {
                    ptr,
                    layout,
                    zeroed: true,
                },
            );

            Ok(())
        };

        // Check that a resized allocation's contents are what we expect them to
        // be.
        let check_resized_bytes = |ptr: NonNull<[u8]>,
                                   size: usize,
                                   zeroed: bool|
         -> Result<(), String> {
            let slice = unsafe { ptr.as_ref() };
            let slice = &slice[..size];

            let expected = if zeroed { 0 } else { LIVE_POISON_PATTERN };
            ensure!(
                slice.iter().all(|b| *b == expected),
                "original allocation's bytes not copied to new allocation during a resizing operation",
            );
            Ok(())
        };

        // Interpret each op and check that our invariants are upheld and
        // properties are maintained as we go!
        for op in &self.ops {
            log::debug!("Running {op:?}");

            match op {
                Op::Alloc { id, layout } => {
                    if live.beyond_allocation_limit(layout.size) {
                        continue;
                    }

                    let layout = layout.alloc_layout();
                    if let Ok(ptr) = allocator.allocate(layout) {
                        new_alloc(&mut live, *id, ptr, layout)?;
                    }
                }

                Op::Dealloc { id } => {
                    if let Some(mut alloc) = live.remove(*id) {
                        // Force this to be a non-zero deallocation, since we
                        // already have `Op::DeallocZeroed` for zeroed
                        // deallocations.
                        if alloc.zeroed {
                            alloc.zeroed = false;
                            fill(alloc.ptr, LIVE_POISON_PATTERN);
                        }
                        dealloc(*id, alloc);
                    }
                }

                Op::AllocZeroed { id, layout } => {
                    if live.beyond_allocation_limit(layout.size) {
                        continue;
                    }

                    let layout = layout.alloc_layout();
                    if let Ok(ptr) = allocator.allocate_zeroed(layout) {
                        new_alloc_zeroed(&mut live, *id, ptr, layout)?;
                    }
                }

                Op::Grow { id, layout } => {
                    let new_layout = layout.alloc_layout();
                    if let Some(old_alloc) = live.remove(*id) {
                        if old_alloc.layout.size() <= new_layout.size()
                            && !live.beyond_allocation_limit(new_layout.size())
                        {
                            match unsafe {
                                allocator.grow(old_alloc.ptr.cast(), old_alloc.layout, new_layout)
                            } {
                                Ok(new_ptr) => {
                                    check_resized_bytes(
                                        new_ptr,
                                        old_alloc.layout.size(),
                                        old_alloc.zeroed,
                                    )?;
                                    new_alloc(&mut live, *id, new_ptr, new_layout)?;
                                }
                                Err(_) => {
                                    // Growing failed; just put it back
                                    // unmodified.
                                    live.insert(*id, old_alloc);
                                }
                            }
                        } else {
                            // Cannot grow an allocation to a smaller size; just
                            // put it back unmodified.
                            live.insert(*id, old_alloc);
                        }
                    }
                }

                Op::GrowZeroed { id, layout } => {
                    let new_layout = layout.alloc_layout();
                    if let Some(old_alloc) = live.remove(*id) {
                        if old_alloc.layout.size() <= new_layout.size()
                            && !live.beyond_allocation_limit(new_layout.size())
                        {
                            match unsafe {
                                allocator.grow_zeroed(
                                    old_alloc.ptr.cast(),
                                    old_alloc.layout,
                                    new_layout,
                                )
                            } {
                                Ok(new_ptr) => {
                                    check_resized_bytes(
                                        new_ptr,
                                        old_alloc.layout.size(),
                                        old_alloc.zeroed,
                                    )?;
                                    if !old_alloc.zeroed {
                                        fill(
                                            NonNull::slice_from_raw_parts(
                                                new_ptr.cast(),
                                                old_alloc.layout.size(),
                                            ),
                                            0,
                                        );
                                    }
                                    new_alloc_zeroed(&mut live, *id, new_ptr, new_layout)?;
                                }
                                Err(_) => {
                                    // Growing failed; just put it back
                                    // unmodified.
                                    live.insert(*id, old_alloc);
                                }
                            }
                        } else {
                            // Cannot grow an allocation to a smaller size; just
                            // put it back unmodified.
                            live.insert(*id, old_alloc);
                        }
                    }
                }

                Op::Shrink { id, layout } => {
                    let new_layout = layout.alloc_layout();
                    if let Some(old_alloc) = live.remove(*id) {
                        if old_alloc.layout.size() >= new_layout.size() {
                            match unsafe {
                                allocator.shrink(old_alloc.ptr.cast(), old_alloc.layout, new_layout)
                            } {
                                Ok(new_ptr) => {
                                    check_resized_bytes(
                                        new_ptr,
                                        new_layout.size(),
                                        old_alloc.zeroed,
                                    )?;
                                    if !old_alloc.zeroed {
                                        fill(new_ptr, 0);
                                    }
                                    new_alloc_zeroed(&mut live, *id, new_ptr, new_layout)?;
                                }
                                Err(_) => {
                                    // Shrinking failed; just put it back
                                    // unmodified.
                                    live.insert(*id, old_alloc);
                                }
                            }
                        } else {
                            // Cannot shrink an allocation to a larger size; just
                            // put it back unmodified.
                            live.insert(*id, old_alloc);
                        }
                    }
                }

                Op::DeallocZeroed { id } => {
                    if let Some(mut alloc) = live.remove(*id) {
                        // Force this to be a zeroed deallocation, since we
                        // already have `Op::Dealloc` for non-zeroed
                        // deallocations.
                        if !alloc.zeroed {
                            alloc.zeroed = true;
                            fill(alloc.ptr, 0);
                        }
                        dealloc(*id, alloc);
                    }
                }
            }
        }

        // Finally, deallocate any remaining live allocations.
        for (id, alloc) in live.map {
            dealloc(id, alloc);
        }

        Ok(())
    }
}

// We fill our non-zeroed memory with a poison pattern, just to try an catch
// more bugs in case the allocator is giving us zeroed memory by default and so
// our assertions just happen to accidentally work.
const LIVE_POISON_PATTERN: u8 = 0xAA;
const FREE_POISON_PATTERN: u8 = 0xFF;

/// A currently-live allocation.
struct LiveAlloc {
    /// Pointer and actual allocated length.
    ptr: NonNull<[u8]>,
    /// Requested layout at allocation time.
    layout: std::alloc::Layout,
    /// Is this allocation zeroed or not? If not, then it is filled with
    /// `LIVE_POISON_PATTERN`.
    zeroed: bool,
}

impl std::fmt::Debug for LiveAlloc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let LiveAlloc {
            ptr,
            layout,
            zeroed,
        } = self;
        f.debug_struct("LiveAlloc")
            .field("ptr", &format!("{ptr:p}"))
            .field("size", &ptr.len())
            .field("layout", &layout)
            .field("zeroed", &zeroed)
            .finish()
    }
}

/// The set of currently-live allocations, keyed by ID.
struct LiveMap {
    /// The live allocations themselves.
    map: BTreeMap<u32, LiveAlloc>,

    /// The total number of bytes that are currently allocated.
    ///
    /// Note: this is a sum of the requested allocation sizes, and does not
    /// include the size of any extra bytes that the allocator may have
    /// included.
    total_allocated_bytes: usize,

    /// The total allocated bytes should never surpass this limit.
    allocation_limit: usize,
}

impl LiveMap {
    fn new(allocation_limit: usize) -> Self {
        LiveMap {
            map: BTreeMap::default(),
            total_allocated_bytes: 0,
            allocation_limit,
        }
    }

    /// Would an allocation of the given size push us past our allocation limit?
    fn beyond_allocation_limit(&self, size: usize) -> bool {
        self.total_allocated_bytes + size > self.allocation_limit
    }

    /// Insert a new live allocation.
    ///
    /// It is the caller's responsibility to check that the given allocation
    /// fits within our configured limit.
    fn insert(&mut self, id: u32, alloc: LiveAlloc) {
        self.total_allocated_bytes += alloc.layout.size();
        assert!(self.total_allocated_bytes <= self.allocation_limit);

        let old = self.map.insert(id, alloc);
        assert!(
            old.is_none(),
            "should remove and deallocate old entries before adding new ones"
        );
    }

    /// Remove a live allocation for deallocation.
    fn remove(&mut self, id: u32) -> Option<LiveAlloc> {
        let alloc = self.map.remove(&id)?;
        self.total_allocated_bytes -= alloc.layout.size();
        Some(alloc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mutatis::check::{Check, CheckError, CheckFailure};

    #[test]
    fn run_ops() {
        let _ = env_logger::try_init();

        let seed_corpus = [
            // Empty.
            Ops::default(),
            // Simple alloc/dealloc pair.
            Ops::new([
                Op::Alloc {
                    id: 0,
                    layout: Layout::unwrap_new(8, 8),
                },
                Op::Dealloc { id: 0 },
            ]),
            // Simple alloc-/dealloc-zeroed pair.
            Ops::new([
                Op::AllocZeroed {
                    id: 0,
                    layout: Layout::unwrap_new(8, 8),
                },
                Op::DeallocZeroed { id: 0 },
            ]),
            // Alloc non-zeroed, dealloc zeroed.
            Ops::new([
                Op::Alloc {
                    id: 0,
                    layout: Layout::unwrap_new(8, 8),
                },
                Op::DeallocZeroed { id: 0 },
            ]),
            // Alloc zeroed, dealloc non-zeroed.
            Ops::new([
                Op::AllocZeroed {
                    id: 0,
                    layout: Layout::unwrap_new(8, 8),
                },
                Op::Dealloc { id: 0 },
            ]),
            // Alloc then dealloc at size 1, the alloc-zeroed at size 2.
            Ops::new([
                Op::AllocZeroed {
                    id: 0,
                    layout: Layout::unwrap_new(1, 1),
                },
                Op::DeallocZeroed { id: 0 },
                Op::AllocZeroed {
                    id: 1,
                    layout: Layout::unwrap_new(2, 1),
                },
            ]),
        ];

        match Check::new().iters(100_000).shrink_iters(1).run_with(
            m::default::<Ops>(),
            seed_corpus,
            |ops| {
                let megabyte = 1 << 20;
                ops.run(megabyte)
            },
        ) {
            Ok(()) => {}
            Err(CheckError::Failed(CheckFailure { value, message, .. })) => {
                panic!("test failure: {message}: {value:#?}")
            }
            Err(e) => panic!("check error: {e}"),
        }
    }
}
