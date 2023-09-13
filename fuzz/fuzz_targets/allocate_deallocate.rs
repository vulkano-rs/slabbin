#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use slabbin::SlabAllocator;

#[derive(Arbitrary, Debug)]
struct Input {
    slab_capacity: u8,
    methods: Vec<AllocatorMethod>,
}

#[derive(Arbitrary, Debug)]
enum AllocatorMethod {
    Allocate,
    Deallocate { index: usize },
}

fuzz_target!(|input: Input| {
    if input.slab_capacity == 0 {
        return;
    }

    let allocator = SlabAllocator::<i32>::new(usize::from(input.slab_capacity));
    let mut ptrs = Vec::new();

    for method in input.methods {
        match method {
            AllocatorMethod::Allocate => {
                ptrs.push(Some(allocator.allocate()));
            }
            AllocatorMethod::Deallocate { index } => {
                if let Some(&Some(ptr)) = ptrs.get(index) {
                    unsafe { allocator.deallocate(ptr) };
                    ptrs[index] = None;
                }
            }
        }
    }

    for ptr in ptrs.into_iter().flatten() {
        unsafe { allocator.deallocate(ptr) };
    }
});
