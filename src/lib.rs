//! This allocator be straight up *slabbin'*.
//!
//! A slab allocator is in theory the perfect one, if it's applicable. It's blazingly fast and
//! avoids the issue of external fragmentation; but unlike the bump allocator it can free
//! individual allocations, and unlike the stack allocator it can free them in an arbitrary order.
//! The tradeoff here is that all allocations must have the same, fixed layout.
//!
//! The allocator in this crate is totally unsafe, and meant specifically for use cases where
//! stable addresses are required: when you allocate a slot, you get a pointer that stays valid
//! until you deallocate it. Example use cases include linked structures or self-referential
//! structures. If you don't have this requirement you may consider using [`slab`] or
//! [`typed-arena`] for example as a safe alternative.
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

#[derive(Debug, Default)]
pub struct SlabAllocator<T> {
    free_list_head: Cell<Option<NonNull<Slot<T>>>>,
    slab_list_head: Cell<Option<NonNull<Slab<T>>>>,
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
    /// `slab_capacity` is the number of slots in a slab.
    ///
    /// No memory is allocated until you call one of the `allocate` methods.
    ///
    /// # Panics
    ///
    /// Panics if `slab_capacity` is zero.
    #[inline]
    #[must_use]
    pub const fn new(slab_capacity: usize) -> Self {
        assert!(slab_capacity != 0);

        SlabAllocator {
            free_list_head: Cell::new(None),
            slab_list_head: Cell::new(None),
            slab_capacity,
        }
    }

    /// Creates a new `SlabAllocator` with the given `slab_count`.
    ///
    /// `slab_capacity` is the number of slots in a slab.
    ///
    /// `slab_count` slabs will be allocated upfront, unless its zero, in which case no memory will
    /// be allocated.
    ///
    /// # Panics
    ///
    /// Panics if `slab_capacity` is zero.
    #[must_use]
    pub fn with_slab_count(slab_capacity: usize, slab_count: usize) -> Self {
        let allocator = SlabAllocator::new(slab_capacity);

        for _ in 0..slab_count {
            allocator
                .add_slab()
                .unwrap_or_else(|_| handle_alloc_error(allocator.slab_layout()));
        }

        allocator
    }

    /// Allocates a new slot for `T`. The memory referred to by the returned pointer needs to be
    /// initialized before creating a reference to it.
    ///
    /// This operation is *O*(1) (amortized).
    #[inline(always)]
    #[must_use]
    pub fn allocate(&self) -> NonNull<T> {
        if let Some(ptr) = self.allocate_fast() {
            ptr
        } else {
            self.allocate_slow()
                .unwrap_or_else(|_| handle_alloc_error(self.slab_layout()))
        }
    }

    /// Allocates a new slot for `T`. The memory referred to by the returned pointer needs to be
    /// initialized before creating a reference to it.
    ///
    /// # Errors
    ///
    /// Returns an error if the global allocator returns an error.
    ///
    /// This operation is *O*(1) (amortized).
    #[inline(always)]
    pub fn try_allocate(&self) -> Result<NonNull<T>, AllocError> {
        if let Some(ptr) = self.allocate_fast() {
            Ok(ptr)
        } else {
            self.allocate_slow()
        }
    }

    #[inline(always)]
    fn allocate_fast(&self) -> Option<NonNull<T>> {
        let head = self.free_list_head.get()?;

        // SAFETY: Each node in the free-list is, by definition, free and therefore must have been
        // initialized with the `next_free` union field when linking it into the list.
        let next = unsafe { (*head.as_ptr()).next_free };

        self.free_list_head.set(next);

        // We can safely hand the user a pointer to `T`, which is valid for writes of `T`, because
        // `Slot<T>` is a `#[repr(C)]` union with `T` as one of its fields. Therefore, the slot
        // must be aligned for `T`, and the field must start at the same point as the slot.
        Some(head.cast::<T>())
    }

    #[cold]
    fn allocate_slow(&self) -> Result<NonNull<T>, AllocError> {
        self.add_slab()?;

        Ok(self.allocate_fast().unwrap())
    }

    fn add_slab(&self) -> Result<(), AllocError> {
        // SAFETY: Slabs always have a non-zero-sized layout.
        let bytes = unsafe { alloc(self.slab_layout()) };

        let slab = NonNull::new(bytes.cast::<Slab<T>>()).ok_or(AllocError)?;

        // SAFETY: We checked that the pointer is non-null, which means allocation succeeded, and
        // we've been given at least the slab header.
        unsafe { (*slab.as_ptr()).next = self.slab_list_head.get() };

        self.slab_list_head.set(Some(slab));

        // SAFETY: The offset must be in bounds for the same reason as the previous.
        let first_slot =
            unsafe { NonNull::new_unchecked(ptr::addr_of_mut!((*slab.as_ptr()).slots)) };

        // SAFETY:
        // * By our own invariant, `self.slab_capacity` must be non-zero, so the subtraction can't
        //   overflow. We know that the offset must be in bounds of the allocated object because we
        //   allocated `capacity` slots.
        // * The computed offset cannot overflow an `isize` because we used `Layout` for the layout
        //   calculation.
        // * The computed offset cannot wrap around the address space for the same reason as the
        //   previous.
        let last_slot = unsafe { first_slot.as_ptr().add(self.slab_capacity - 1) };

        let mut slot = first_slot;

        loop {
            if slot.as_ptr() == last_slot {
                // SAFETY: We know that this pointer is valid for writes because of the same
                // reasons as the previous safety comment as well as the fact that the memory was
                // just now allocated and no pointers to it have been given out yet.
                unsafe { (*slot.as_ptr()).next_free = self.free_list_head.get() };

                break;
            }

            // SAFETY:
            // * `slot` starts out referring to the first slot and is incremented here in each loop
            //   iteration. We checked above that it doesn't refer to the last slot. Therefore the
            //   offset must be in bounds of the allocated object, lest the loop would have ended.
            // * The computed offset cannot overflow an `isize` because we used `Layout` for the
            //   layout calculation.
            // * The computed offset cannot wrap around the address space for the same reason as
            //   the previous.
            let next_free = unsafe { NonNull::new_unchecked(slot.as_ptr().add(1)) };

            // SAFETY: We know that this pointer is valid for writes because of the same reasons as
            // the previous safety comment, as well as the fact that the memory was just now
            // allocated and no pointers to it have been given out yet.
            unsafe { (*slot.as_ptr()).next_free = Some(next_free) };

            slot = next_free;
        }

        self.free_list_head.set(Some(first_slot));

        Ok(())
    }

    fn slab_layout(&self) -> Layout {
        Layout::new::<Option<NonNull<Self>>>()
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
}

impl<T> Drop for SlabAllocator<T> {
    fn drop(&mut self) {
        let slab_layout = self.slab_layout();

        while let Some(slab) = self.slab_list_head.get() {
            // SAFETY: `slab` being in the slab list means it refers to a currently allocated slab,
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
    fn basic_usage() {
        let allocator = SlabAllocator::<i32>::new(2);

        let mut x_ptr = allocator.allocate();
        unsafe { x_ptr.as_ptr().write(69) };
        let x = unsafe { x_ptr.as_mut() };

        let mut y_ptr = allocator.allocate();
        unsafe { y_ptr.as_ptr().write(42) };
        let y = unsafe { y_ptr.as_mut() };

        mem::swap(x, y);

        assert_eq!(&*x, &42);
        assert_eq!(&*y, &69);

        unsafe { allocator.deallocate(x_ptr) };

        let mut z_ptr = allocator.allocate();
        unsafe { z_ptr.as_ptr().write(12) };
        let z = unsafe { z_ptr.as_mut() };

        mem::swap(y, z);

        assert_eq!(&*y, &12);
        assert_eq!(&*z, &69);

        unsafe { allocator.deallocate(z_ptr) };
    }
}
