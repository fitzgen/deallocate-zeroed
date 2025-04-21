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
#[derive(Clone, Copy, Debug, bincode::Encode, bincode::Decode)]
pub struct Layout {
    size: usize,
    align: usize,
}

const MAX_ALIGN: usize = 1 << 16;

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

    fn align(&self) -> usize {
        // Alignment must be a power of two and less than or equal to
        // `isize::MAX`.
        //
        // NB: We cannot assume that `self.align` is a valid alignment because
        // `libfuzzer` could have arbitrarily mutated its bytes.
        let align = self.align.checked_next_power_of_two().unwrap_or(1);
        align.min(MAX_ALIGN)
    }

    fn size(&self) -> usize {
        let align = self.align();

        // The size must not overflow `isize::MAX` when rounded up to our
        // alignment.
        //
        // NB: We cannot assume that `self.size` is a valid alignment because
        // `libfuzzer` could have arbitrarily mutated its bytes.
        let size = if self
            .size
            .checked_next_multiple_of(align)
            .is_some_and(|s| s < (isize::MAX as usize))
        {
            self.size
        } else {
            ((isize::MAX as usize) - align)
                .checked_next_multiple_of(align)
                .unwrap()
        };

        debug_assert!(size
            .checked_next_multiple_of(align)
            .is_some_and(|s| s < (isize::MAX as usize)));

        size
    }

    fn alloc_layout(&self) -> std::alloc::Layout {
        let align = self.align();
        let size = self.size();
        match std::alloc::Layout::from_size_align(size, align) {
            Ok(l) => l,
            Err(e) => panic!("should be a valid layout: size={size}, align={align}; got {e}"),
        }
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
                layout.size()
            } else {
                std::cmp::min(self.max_size, max_size_for_align(layout.align()))
            };
            layout.size = ctx.rng().gen_index(max_size + 1).unwrap();
            Ok(())
        })?;

        // Mutate alignment.
        c.mutation(|ctx| {
            let max_align_log2 = if ctx.shrink() {
                layout.align().trailing_zeros() as usize
            } else {
                std::cmp::min(
                    self.max_align.trailing_zeros() as usize,
                    max_align_for_size(layout.size()).trailing_zeros() as usize,
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
#[derive(Clone, Debug, Mutate, bincode::Encode, bincode::Decode)]
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

// Note: we use custom implementations of `Encode` and `Decode` so that we can
// limit the size of `ops`; otherwise, `bincode` will attempt to allocate an
// array of arbitrary length from the fuzzer-supplied data, which leads to OOMs.
const MAX_OPS_TO_DECODE: usize = 1_000;
impl<C> bincode::Decode<C> for Ops {
    fn decode<D: bincode::de::Decoder<Context = C>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let len = u64::decode(decoder)?;
        let len = usize::try_from(len)
            .map_err(|e| bincode::error::DecodeError::OtherString(e.to_string()))?;
        if len > MAX_OPS_TO_DECODE {
            return Err(bincode::error::DecodeError::OtherString(format!(
                "cannot decode an `Ops` of length {len}; max supported length is \
                 {MAX_OPS_TO_DECODE}"
            )));
        }
        let mut ops = Vec::with_capacity(len);
        for _ in 0..len {
            let op = Op::decode(decoder)?;
            ops.push(op);
        }
        Ok(Ops { ops })
    }
}

impl bincode::Encode for Ops {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        let len = self.ops.len();
        let len = u64::try_from(len).unwrap();
        u64::encode(&len, encoder)?;
        for op in &self.ops {
            Op::encode(op, encoder)?;
        }
        Ok(())
    }
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
            let file = file!();
            let line = line!();
            let msg = format!($msg $( , $args )* );
            let cond = stringify!($cond);
            return Err(format!("{file}:{line}: check failed: `{cond}`: {msg}"));
        }
    }};
}

impl Ops {
    /// Create a new `Ops` from the given test operations.
    pub fn new(ops: impl IntoIterator<Item = Op>) -> Self {
        let ops = ops.into_iter().collect();
        Ops { ops }
    }

    /// Pop an operation off the end of this sequence. Returns whether an
    /// operation was actually popped or not (i.e. whether this sequence was
    /// non-empty before calling `pop`).
    pub fn pop(&mut self) -> bool {
        self.ops.pop().is_some()
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
            log::trace!(
                "fill {:#p}..{:#p} with {byte:#0x}",
                ptr.cast::<u8>(),
                ptr.cast::<u8>().add(ptr.len())
            );
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
            log::trace!(
                "assert_zeroed(ptr = {ptr:#p}..{:#p} (len = {}))",
                unsafe { ptr.cast::<u8>().add(ptr.len()) },
                ptr.len()
            );
            let slice = unsafe { ptr.as_ref() };
            for b in slice {
                ensure!(
                    *b == 0,
                    "supposedly zeroed block of memory contains non-zero byte {:#0x} at {:#p}",
                    *b,
                    b as *const _,
                );
            }
            Ok(())
        };

        // Assert that the given allocation satisfies its requested layout.
        let assert_fits_layout =
            |ptr: NonNull<[u8]>, layout: std::alloc::Layout| -> Result<(), String> {
                log::trace!(
                    "assert_fits_layout(ptr = {ptr:#p}..{:#p} (len = {}), layout = {layout:?})",
                    unsafe { ptr.cast::<u8>().add(ptr.len()) },
                    ptr.len()
                );
                let actual_size = ptr.len();
                let expected_size = layout.size();
                ensure!(
                    actual_size >= expected_size,
                    "actual allocated size ({actual_size}) is less than expected layout size \
                     ({expected_size})",
                );
                let actual_align = 1 << (ptr.cast::<u8>().as_ptr() as usize).trailing_zeros();
                let expected_align = layout.align();
                ensure!(
                    actual_align >= expected_align,
                    "actual allocated alignment ({actual_align}) is less than expected layout \
                     alignment ({expected_align})",
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
                    if live.beyond_allocation_limit(layout.size()) {
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
                    if live.beyond_allocation_limit(layout.size()) {
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
        self.total_allocated_bytes
            .checked_add(size)
            .is_none_or(|n| n > self.allocation_limit)
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
