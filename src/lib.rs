//! This allocator be straight up *slabbin'*.
//!
//! A slab allocator is in theory the perfect one, if it's applicable. It's blazingly fast and
//! avoids the issue of external fragmentation; but unlike the bump allocator it can free
//! individual allocations, and unlike the stack allocator it can free them in an arbitrary order.
//! The tradeoff here is that all allocations must have the same, fixed layout.
//!
//! The allocator in this crate is totally unsafe, and meant specifically for use cases where
//! stable addresses are required: when you allocate a slot, you get a pointer that stays valid
//! until you deallocate it (or drop the allocator). Example use cases include linked structures or
//! self-referential structures. If you don't have this requirement you may consider using [`slab`]
//! or [`typed-arena`] for example as a safe alternative.
//!
//! # Slabs
//!
//! A slab is a pre-allocated contiguous chunk of memory containing *slots*. Each slot can either
//! be free or occupied. A slab always starts out with all slots free, and new slots are given out
//! on each allocation, until they run out, at which point a new slab is allocated. Slots that are
//! deallocated are chained together in a linked list. Due to this, allocation amounts to 3
//! operations in the best case and ~8 in the worse case. Deallocation is always 3 operations.
//!
//! [`slab`]: https://crates.io/crates/slab
//! [`typed-arena`]: https://crates.io/crates/typed-arena

#![forbid(unsafe_op_in_unsafe_fn)]
#![no_std]

extern crate alloc;

use alloc::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use core::{
    cell::Cell,
    fmt,
    mem::ManuallyDrop,
    ptr::{self, NonNull},
};

/// An efficient slab allocator with stable addresses.
///
/// See also [the crate-level documentation] for more information about slab allocation.
///
/// # Examples
///
/// A doubly linked list that's backed by slabs:
///
/// ```
/// use slabbin::SlabAllocator;
/// use std::ptr::{self, NonNull};
///
/// struct LinkedList<T> {
///     head: Option<NonNull<Node<T>>>,
///     tail: Option<NonNull<Node<T>>>,
///     allocator: SlabAllocator<Node<T>>,
/// }
///
/// impl<T> LinkedList<T> {
///     fn new(slab_capacity: usize) -> Self {
///         LinkedList {
///             head: None,
///             tail: None,
///             allocator: SlabAllocator::new(slab_capacity),
///         }
///     }
///
///     fn push_back(&mut self, value: T) {
///         let node = self.allocator.allocate();
///
///         // SAFETY: `SlabAllocator::allocate` gives out pointers that are valid for writes (but
///         // **not** for reads).
///         unsafe { *node.as_ptr() = Node::new(value) };
///
///         if let Some(tail) = self.tail {
///             unsafe { (*tail.as_ptr()).next = Some(node) };
///             unsafe { (*node.as_ptr()).prev = Some(tail) };
///         } else {
///             self.head = Some(node);
///         }
///
///         self.tail = Some(node);
///     }
///
///     fn pop_back(&mut self) -> Option<T> {
///         if let Some(tail) = self.tail {
///             if let Some(prev) = unsafe { (*tail.as_ptr()).prev } {
///                 unsafe { (*prev.as_ptr()).next = None };
///                 self.tail = Some(prev);
///             } else {
///                 self.head = None;
///                 self.tail = None;
///             }
///
///             // SAFETY: We can move out of the value because the node will be deallocated.
///             let value = unsafe { ptr::read(ptr::addr_of_mut!((*tail.as_ptr()).value)) };
///
///             // SAFETY: We allocated this node, and have just removed all linkage to it so that
///             // it can't be accessed again.
///             unsafe { self.allocator.deallocate(tail) };
///
///             Some(value)
///         } else {
///             None
///         }
///     }
/// }
///
/// struct Node<T> {
///     prev: Option<NonNull<Self>>,
///     next: Option<NonNull<Self>>,
///     value: T,
/// }
///
/// impl<T> Node<T> {
///     fn new(value: T) -> Self {
///         Node {
///             prev: None,
///             next: None,
///             value,
///         }
///     }
/// }
///
/// let mut list = LinkedList::new(64);
/// list.push_back(42);
/// list.push_back(12);
/// list.push_back(69);
///
/// assert_eq!(list.pop_back(), Some(69));
/// assert_eq!(list.pop_back(), Some(12));
/// assert_eq!(list.pop_back(), Some(42));
/// assert_eq!(list.pop_back(), None);
/// ```
///
/// [the crate-level documentation]: self
pub struct SlabAllocator<T> {
    free_list_head: Cell<Option<NonNull<Slot<T>>>>,
    slab_list_head: Cell<Option<NonNull<Slab<T>>>>,
    /// Points to the slot of the slab of the head of the slab list where the free slots start.
    /// If the slab list is empty then this is dangling.
    free_start: Cell<NonNull<Slot<T>>>,
    /// Points to the end of the slab of the head of the slab list. If the slab list is empty then
    /// this is dangling.
    free_end: Cell<NonNull<Slot<T>>>,
    slab_capacity: usize,
}

// SAFETY: The pointers we hold are not referencing the stack or TLS or anything like that, they
// are all heap allocation, and therefore sending them to another thread is safe. Note that it is
// safe to do this regardless of whether `T` is `Send`: the allocator itself doesn't own any `T`s,
// the user does and manages their lifetime explicitly by allocating, initializing, deinitializing
// and then deallocating them. It is therefore their responsibility that a type that is `!Send`
// doesn't escape the thread it was created on (just like with any other allocator, it just so
// happens that this one is parametrized, but it doesn't have to be). Pointers are always `!Send`
// so the user would have to use unsafe code to achieve something like that in the first place.
unsafe impl<T> Send for SlabAllocator<T> {}

impl<T> SlabAllocator<T> {
    /// Creates a new `SlabAllocator`.
    ///
    /// `slab_capacity` is the number of slots in a [slab].
    ///
    /// No memory is allocated until you call one of the `allocate` methods.
    ///
    /// # Panics
    ///
    /// Panics if `slab_capacity` is zero.
    ///
    /// [slab]: self#slabs
    #[inline]
    #[must_use]
    pub const fn new(slab_capacity: usize) -> Self {
        assert!(slab_capacity != 0);

        let dangling = NonNull::dangling();

        SlabAllocator {
            free_list_head: Cell::new(None),
            slab_list_head: Cell::new(None),
            free_start: Cell::new(dangling),
            free_end: Cell::new(dangling),
            slab_capacity,
        }
    }

    /// Allocates a new slot for `T`. The memory referred to by the returned pointer needs to be
    /// initialized before creating a reference to it.
    ///
    /// This operation is *O*(1).
    ///
    /// # Panics
    ///
    /// Panics if the size of a slab exceeds `isize::MAX` bytes.
    #[inline(always)]
    #[must_use]
    pub fn allocate(&self) -> NonNull<T> {
        let ptr = if let Some(ptr) = self.allocate_fast() {
            ptr
        } else if let Some(ptr) = self.allocate_fast2() {
            ptr
        } else {
            self.allocate_slow()
                .unwrap_or_else(|_| handle_alloc_error(self.slab_layout()))
        };

        // We can safety hand the user a pointer to `T`, which is valid for writes of `T`, seeing a
        // `Slot<T>` is a union with `T` as one of its fields. That means that the slot must have a
        // layout that fits `T` as we used `Layout` for the layout calculation of the slots array.
        ptr.cast::<T>()
    }

    /// Allocates a new slot for `T`. The memory referred to by the returned pointer needs to be
    /// initialized before creating a reference to it.
    ///
    /// This operation is *O*(1).
    ///
    /// # Errors
    ///
    /// Returns an error if the global allocator returns an error.
    ///
    /// # Panics
    ///
    /// Panics if the size of a slab exceeds `isize::MAX` bytes.
    #[inline(always)]
    pub fn try_allocate(&self) -> Result<NonNull<T>, AllocError> {
        let ptr = if let Some(ptr) = self.allocate_fast() {
            ptr
        } else if let Some(ptr) = self.allocate_fast2() {
            ptr
        } else {
            self.allocate_slow()?
        };

        Ok(ptr.cast::<T>())
    }

    #[inline(always)]
    fn allocate_fast(&self) -> Option<NonNull<Slot<T>>> {
        let head = self.free_list_head.get()?;

        // SAFETY: Each node in the free-list is, by definition, free and therefore must have been
        // initialized with the `next_free` union field when linking it into the list.
        let next = unsafe { (*head.as_ptr()).next_free };

        self.free_list_head.set(next);

        // Make Miri comprehend that a slot must be initialized before reading it, even if
        // `size_of::<T>()` <= `size_of::<usize>()` in which case we happened to have initialized
        // the bytes.
        #[cfg(miri)]
        {
            use core::mem::MaybeUninit;

            let ptr = head.as_ptr().cast::<MaybeUninit<T>>();

            unsafe { ptr.write(MaybeUninit::uninit()) };
        }

        // We can safety hand the user a pointer to the head of the free-list, seeing as we removed
        // it from the list so that it cannot be handed out again.
        Some(head)
    }

    #[inline(always)]
    fn allocate_fast2(&self) -> Option<NonNull<Slot<T>>> {
        let ptr = self.free_start.get();

        if ptr < self.free_end.get() {
            // SAFETY:
            // * We know the offset must be in bounds of the allocated object because we just
            //   checked that `free_start` doesn't refer to the end of the allocated object yet:
            //   * `free_start` and `free_end` are initialized such that they refer to the start
            //     and end of the slots array respectively.
            //   * If the pointers haven't been initialized yet, then they are both dangling and
            //     equal, which means the the above condition trivially wouldn't hold.
            //   * This is the only place where `free_start` is incremented, always by 1.
            //     `free_end` is unchanging until a new slab is allocated.
            // * The computed offset cannot overflow an `isize` because we used `Layout` for the
            //   layout calculation.
            // * The computed offset cannot wrap around the address space for the same reason as
            //   the previous.
            let free_start = unsafe { NonNull::new_unchecked(ptr.as_ptr().add(1)) };

            self.free_start.set(free_start);

            // We can safety hand the user a pointer to the previous free-start, as we incremented
            // it such that the same slot cannot be handed out again.
            Some(ptr)
        } else {
            None
        }
    }

    #[cold]
    fn allocate_slow(&self) -> Result<NonNull<Slot<T>>, AllocError> {
        let slab = self.add_slab()?;

        // SAFETY: The allocation succeeded, which means we've been given at least the slab header,
        // so the offset must be in range.
        let slots = unsafe { NonNull::new_unchecked(ptr::addr_of_mut!((*slab.as_ptr()).slots)) };

        // SAFETY:
        // * We know that the offset must be in bounds of the allocated object because we allocated
        //   `self.slab_capacity` slots, and by our own invariant, `self.slab_capacity` must be
        //   non-zero.
        // * The computed offset cannot overflow an `isize` because we used `Layout` for the layout
        //   calculation.
        // * The computed offset cannot wrap around the address space for the same reason as the
        //   previous.
        let free_start = unsafe { NonNull::new_unchecked(slots.as_ptr().add(1)) };

        // SAFETY: Same as the previous.
        let free_end = unsafe { NonNull::new_unchecked(slots.as_ptr().add(self.slab_capacity)) };

        self.free_start.set(free_start);
        self.free_end.set(free_end);

        // We can safely hand the user a pointer to the first slot, seeing as we set the free-start
        // to the next slot, so that the same slot cannot be handed out again.
        Ok(slots)
    }

    fn add_slab(&self) -> Result<NonNull<Slab<T>>, AllocError> {
        // SAFETY: Slabs always have a non-zero-sized layout.
        let bytes = unsafe { alloc(self.slab_layout()) };

        let slab = NonNull::new(bytes.cast::<Slab<T>>()).ok_or(AllocError)?;

        // SAFETY: We checked that the pointer is non-null, which means allocation succeeded, and
        // we've been given at least the slab header.
        unsafe { (*slab.as_ptr()).next = self.slab_list_head.get() };

        self.slab_list_head.set(Some(slab));

        Ok(slab)
    }

    fn slab_layout(&self) -> Layout {
        Layout::new::<Option<NonNull<Slab<T>>>>()
            .extend(Layout::array::<Slot<T>>(self.slab_capacity).unwrap())
            .unwrap()
            .0
            .pad_to_align()
    }

    /// Deallocates the slot at the given `ptr`. The `T` is not dropped before deallocating, you
    /// must do so yourself before calling this function if `T` has drop glue (unless you want to
    /// leak).
    ///
    /// This operation is *O*(1).
    ///
    /// # Safety
    ///
    /// `ptr` must refer to a slot that's **currently allocated** by `self`.
    #[inline(always)]
    pub unsafe fn deallocate(&self, ptr: NonNull<T>) {
        let ptr = ptr.cast::<Slot<T>>();

        // SAFETY: The caller must ensure that `ptr` refers to a currently allocated slot, meaning
        // that `ptr` was derived from one of our slabs using `allocate`, making it a valid ponter.
        // We can overwrite whatever was in the slot before, because nothing must access a pointer
        // after its memory block has been deallocated (as that would constitute a Use-After-Free).
        // In our case we reuse the memory for the free-list linkage.
        unsafe { (*ptr.as_ptr()).next_free = self.free_list_head.get() };

        self.free_list_head.set(Some(ptr));
    }

    fn slab_count(&self) -> usize {
        let mut head = self.slab_list_head.get();
        let mut count = 0;

        while let Some(slab) = head {
            // SAFETY: `slab` being in the slab list means it refers to a currently allocated slab
            // and that its header is properly initialized.
            unsafe { head = (*slab.as_ptr()).next };

            count += 1;
        }

        count
    }
}

impl<T> fmt::Debug for SlabAllocator<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SlabAllocator")
            .field("slab_count", &self.slab_count())
            .field("slab_capacity", &self.slab_capacity)
            .finish()
    }
}

impl<T> Drop for SlabAllocator<T> {
    fn drop(&mut self) {
        let slab_layout = self.slab_layout();

        while let Some(slab) = self.slab_list_head.get() {
            // SAFETY: `slab` being in the slab list means it refers to a currently allocated slab
            // and that its header is properly initialized.
            *self.slab_list_head.get_mut() = unsafe { (*slab.as_ptr()).next };

            // SAFETY:
            // * `slab` being in the slab list means it refers to a currently allocated slab.
            // * `self.slab_layout()` returns the same layout that was used to allocate the slab.
            unsafe { dealloc(slab.as_ptr().cast(), slab_layout) };
        }
    }
}

#[repr(C)]
struct Slab<T> {
    next: Option<NonNull<Self>>,
    /// The actual field type is `[Slot<T>]` except that we want a thin pointer.
    slots: Slot<T>,
}

#[repr(C)]
union Slot<T> {
    next_free: Option<NonNull<Self>>,
    value: ManuallyDrop<T>,
}

/// Indicates that allocating memory using the global allocator failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocError;

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("memory allocation failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem;

    #[test]
    fn basic_usage1() {
        let allocator = SlabAllocator::<i32>::new(2);

        let mut x = allocator.allocate();
        unsafe { x.as_ptr().write(69) };

        let mut y = allocator.allocate();
        unsafe { y.as_ptr().write(42) };

        assert_eq!(allocator.slab_count(), 1);

        mem::swap(unsafe { x.as_mut() }, unsafe { y.as_mut() });

        unsafe { allocator.deallocate(x) };

        let mut x2 = allocator.allocate();
        unsafe { x2.as_ptr().write(12) };

        assert_eq!(allocator.slab_count(), 1);

        mem::swap(unsafe { y.as_mut() }, unsafe { x2.as_mut() });

        unsafe { allocator.deallocate(y) };
        unsafe { allocator.deallocate(x2) };
    }

    #[test]
    fn basic_usage2() {
        let allocator = SlabAllocator::<i32>::new(1);

        let mut x = allocator.allocate();
        unsafe { x.as_ptr().write(1) };

        let mut y = allocator.allocate();
        unsafe { y.as_ptr().write(2) };

        let mut z = allocator.allocate();
        unsafe { z.as_ptr().write(3) };

        assert_eq!(allocator.slab_count(), 3);

        mem::swap(unsafe { x.as_mut() }, unsafe { y.as_mut() });
        mem::swap(unsafe { y.as_mut() }, unsafe { z.as_mut() });
        mem::swap(unsafe { z.as_mut() }, unsafe { x.as_mut() });

        unsafe { allocator.deallocate(y) };

        let mut y2 = allocator.allocate();
        unsafe { y2.as_ptr().write(20) };

        assert_eq!(allocator.slab_count(), 3);

        mem::swap(unsafe { x.as_mut() }, unsafe { y2.as_mut() });

        unsafe { allocator.deallocate(x) };
        unsafe { allocator.deallocate(z) };

        let mut x2 = allocator.allocate();
        unsafe { x2.as_ptr().write(10) };

        mem::swap(unsafe { y2.as_mut() }, unsafe { x2.as_mut() });

        let mut z2 = allocator.allocate();
        unsafe { z2.as_ptr().write(30) };

        assert_eq!(allocator.slab_count(), 3);

        mem::swap(unsafe { x2.as_mut() }, unsafe { z2.as_mut() });

        unsafe { allocator.deallocate(x2) };

        mem::swap(unsafe { z2.as_mut() }, unsafe { y2.as_mut() });

        unsafe { allocator.deallocate(y2) };
        unsafe { allocator.deallocate(z2) };
    }

    #[test]
    fn basic_usage3() {
        let allocator = SlabAllocator::<i32>::new(2);

        let mut x = allocator.allocate();
        unsafe { x.as_ptr().write(1) };

        let mut y = allocator.allocate();
        unsafe { y.as_ptr().write(2) };

        assert_eq!(allocator.slab_count(), 1);

        mem::swap(unsafe { x.as_mut() }, unsafe { y.as_mut() });

        let z = allocator.allocate();
        unsafe { z.as_ptr().write(3) };

        assert_eq!(allocator.slab_count(), 2);

        unsafe { allocator.deallocate(x) };
        unsafe { allocator.deallocate(z) };

        let mut z2 = allocator.allocate();
        unsafe { z2.as_ptr().write(30) };

        let mut x2 = allocator.allocate();
        unsafe { x2.as_ptr().write(10) };

        assert_eq!(allocator.slab_count(), 2);

        mem::swap(unsafe { x2.as_mut() }, unsafe { z2.as_mut() });

        unsafe { allocator.deallocate(x2) };
        unsafe { allocator.deallocate(y) };
        unsafe { allocator.deallocate(z2) };
    }

    #[test]
    fn reusing_slots1() {
        let allocator = SlabAllocator::<i32>::new(2);

        let x = allocator.allocate();
        let y = allocator.allocate();

        unsafe { allocator.deallocate(y) };

        let y2 = allocator.allocate();
        assert_eq!(y2, y);

        unsafe { allocator.deallocate(x) };

        let x2 = allocator.allocate();
        assert_eq!(x2, x);

        unsafe { allocator.deallocate(y2) };
        unsafe { allocator.deallocate(x2) };
    }

    #[test]
    fn reusing_slots2() {
        let allocator = SlabAllocator::<i32>::new(1);

        let x = allocator.allocate();

        unsafe { allocator.deallocate(x) };

        let x2 = allocator.allocate();
        assert_eq!(x, x2);

        let y = allocator.allocate();
        let z = allocator.allocate();

        unsafe { allocator.deallocate(y) };
        unsafe { allocator.deallocate(x2) };

        let x3 = allocator.allocate();
        let y2 = allocator.allocate();
        assert_eq!(x3, x2);
        assert_eq!(y2, y);

        unsafe { allocator.deallocate(x3) };
        unsafe { allocator.deallocate(y2) };
        unsafe { allocator.deallocate(z) };
    }

    #[test]
    fn reusing_slots3() {
        let allocator = SlabAllocator::<i32>::new(2);

        let x = allocator.allocate();
        let y = allocator.allocate();

        unsafe { allocator.deallocate(x) };
        unsafe { allocator.deallocate(y) };

        let y2 = allocator.allocate();
        let x2 = allocator.allocate();
        let z = allocator.allocate();
        assert_eq!(x2, x);
        assert_eq!(y2, y);

        unsafe { allocator.deallocate(x2) };
        unsafe { allocator.deallocate(z) };
        unsafe { allocator.deallocate(y2) };

        let y3 = allocator.allocate();
        let z2 = allocator.allocate();
        let x3 = allocator.allocate();
        assert_eq!(y3, y2);
        assert_eq!(z2, z);
        assert_eq!(x3, x2);

        unsafe { allocator.deallocate(x3) };
        unsafe { allocator.deallocate(y3) };
        unsafe { allocator.deallocate(z2) };
    }

    #[test]
    fn same_slab() {
        const MAX_DIFF: usize = 2 * mem::size_of::<Slot<i32>>();

        let allocator = SlabAllocator::<i32>::new(3);

        let x = allocator.allocate();
        let y = allocator.allocate();
        let z = allocator.allocate();

        assert!((x.as_ptr() as usize).abs_diff(y.as_ptr() as usize) <= MAX_DIFF);
        assert!((y.as_ptr() as usize).abs_diff(z.as_ptr() as usize) <= MAX_DIFF);
        assert!((z.as_ptr() as usize).abs_diff(x.as_ptr() as usize) <= MAX_DIFF);
    }

    #[test]
    fn different_slabs() {
        const MIN_DIFF: usize = mem::size_of::<Slab<i32>>();

        let allocator = SlabAllocator::<i32>::new(1);

        let x = allocator.allocate();
        let y = allocator.allocate();
        let z = allocator.allocate();

        assert!((x.as_ptr() as usize).abs_diff(y.as_ptr() as usize) >= MIN_DIFF);
        assert!((y.as_ptr() as usize).abs_diff(z.as_ptr() as usize) >= MIN_DIFF);
        assert!((z.as_ptr() as usize).abs_diff(x.as_ptr() as usize) >= MIN_DIFF);
    }
}
