/* The calls between victory-metal's Windows side and its Wine unixlib, and
 * their arguments. src/abi.rs is the Rust side of the same layout; the
 * sizes below are checked on both sides, and every argument block starts
 * with its own size, which the unixlib checks before it reads further. */
#ifndef VICTORY_METAL_ABI_H
#define VICTORY_METAL_ABI_H

#include <stdint.h>

enum vm_call {
    /* Nothing: what a call through Wine's dispatcher costs. No arguments. */
    VM_NOOP = 0,
    /* Makes the system's default Metal device, a queue and the fill
     * pipeline, once, and describes the device. struct vm_device. */
    VM_OPEN = 1,
    /* Runs compute work over memory the caller owns, in place.
     * struct vm_fill. */
    VM_FILL = 2,
    /* Empty command buffers, each committed and waited for alone.
     * struct vm_submit. */
    VM_SUBMIT_EMPTY = 3,
    VM_CALLS
};

#define VM_ERROR_LEN 192

struct vm_device {
    uint32_t size;
    uint32_t unified_memory;
    uint64_t registry_id;
    uint64_t max_buffer_length;
    uint64_t recommended_working_set;
    /* The CPU's page size in this process (4 KiB under Rosetta). */
    uint64_t page_size;
    char name[64];
    char error[VM_ERROR_LEN];
};

/* Wraps [address, address + length) in one buffer with no copy, then
 * records `dispatches` compute dispatches into one command buffer, each
 * writing its own equal share of the words, word k = value + k, and waits
 * for it. The times are the unixlib's own clock. */
struct vm_fill {
    uint32_t size;
    uint32_t dispatches;
    uint64_t address;
    uint64_t length;
    uint32_t value;
    uint32_t threads_per_group;
    uint64_t wrap_ns;
    uint64_t encode_ns;
    /* From the commit to the end of the wait for completion. */
    uint64_t wait_ns;
    /* What the GPU reports between the command buffer's start and end. */
    uint64_t gpu_ns;
    char error[VM_ERROR_LEN];
};

struct vm_submit {
    uint32_t size;
    uint32_t count;
    uint64_t total_ns;
    char error[VM_ERROR_LEN];
};

_Static_assert(sizeof(struct vm_device) == 296, "vm_device layout");
_Static_assert(sizeof(struct vm_fill) == 256, "vm_fill layout");
_Static_assert(sizeof(struct vm_submit) == 208, "vm_submit layout");

#endif
