#![feature(test)]

extern crate test;

use common::ALLOCATION_COUNT;
use slabbin::SlabAllocator;
use std::{
    alloc::{alloc, dealloc, Layout},
    ptr::{self, NonNull},
};
use test::Bencher;

mod common;

#[bench]
fn slab_capacity_000_100(b: &mut Bencher) {
    slab_capacity::<100>(b);
}

#[bench]
fn slab_capacity_001_000(b: &mut Bencher) {
    slab_capacity::<1_000>(b);
}

#[bench]
fn slab_capacity_010_000(b: &mut Bencher) {
    slab_capacity::<10_000>(b);
}

#[bench]
fn slab_capacity_100_000(b: &mut Bencher) {
    slab_capacity::<100_000>(b);
}

fn slab_capacity<const CAPACITY: usize>(b: &mut Bencher) {
    let allocator = SlabAllocator::<i32>::new(CAPACITY);
    let mut ptrs = Box::new([NonNull::dangling(); ALLOCATION_COUNT]);

    b.iter(|| {
        for ptr in ptrs.as_mut_slice() {
            *ptr = allocator.allocate();
        }

        for ptr in ptrs.as_slice() {
            unsafe { allocator.deallocate(*ptr) };
        }
    });
}

#[bench]
fn system(b: &mut Bencher) {
    let mut ptrs = Box::new([ptr::null_mut::<u8>(); ALLOCATION_COUNT]);

    b.iter(|| {
        for ptr in ptrs.as_mut_slice() {
            *ptr = unsafe { alloc(Layout::new::<i32>()) };
        }

        for ptr in ptrs.as_slice() {
            unsafe { dealloc(*ptr, Layout::new::<i32>()) };
        }
    });
}
