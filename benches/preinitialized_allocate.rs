#![feature(test)]

extern crate test;

use common::ALLOCATION_COUNT;
use slabbin::SlabAllocator;
use std::alloc::{alloc, Layout};
use test::{black_box, Bencher};

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
    let allocator = SlabAllocator::<i32>::with_slab_count(CAPACITY, ALLOCATION_COUNT / CAPACITY);

    b.iter(|| {
        for _ in 0..ALLOCATION_COUNT {
            black_box(allocator.allocate());
        }
    });
}

#[bench]
fn system(b: &mut Bencher) {
    b.iter(|| {
        for _ in 0..ALLOCATION_COUNT {
            black_box(unsafe { alloc(Layout::new::<i32>()) });
        }
    });
}
