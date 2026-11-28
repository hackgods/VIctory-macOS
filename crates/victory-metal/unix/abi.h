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
    /* Compute and render work taking turns, or batched, in one command
     * buffer. struct vm_turns. */
    VM_TURNS = 4,
    /* Wraps memory the caller owns in a no-copy buffer and keeps it under a
     * handle. struct vm_wrap. */
    VM_WRAP = 5,
    /* Lets a handle (buffer, event or run) go. struct vm_release. */
    VM_RELEASE = 6,
    /* Makes, signals, waits for or reads a shared event from the CPU.
     * struct vm_event. */
    VM_EVENT = 7,
    /* Records a list of ops into one command buffer and commits it, waited
     * for or not. struct vm_run, struct vm_op. */
    VM_RUN = 8,
    /* Waits for a run committed without waiting, and writes its results
     * back into its ops. struct vm_wait. */
    VM_WAIT = 9,
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

/* `pairs` times: one compute dispatch of one thread writes a draw's
 * arguments (20 bytes, MTLDrawIndexedPrimitivesIndirectArguments), then an
 * indirect indexed draw reads them into a 64 x 64 render target. With
 * `batched` 0 each dispatch has a compute encoder of its own and each draw
 * a render encoder of its own (the target loaded and stored each time);
 * with 1, one compute encoder holds all dispatches and one render encoder
 * all draws. The unixlib checks every record and the target. */
struct vm_turns {
    uint32_t size;
    uint32_t pairs;
    uint32_t batched;
    uint32_t reserved;
    uint64_t encode_ns;
    uint64_t wait_ns;
    uint64_t gpu_ns;
    char error[VM_ERROR_LEN];
};

/* VM_WRAP: [address, address + length) as one no-copy buffer
 * (MTLResourceStorageModeShared), kept until VM_RELEASE. `flags` bit 0
 * leaves its hazards untracked; by default Metal tracks them, so ops in
 * different encoders of one run are ordered on the buffer. */
#define VM_WRAP_UNTRACKED 1u
struct vm_wrap {
    uint32_t size;
    uint32_t flags;
    uint64_t address;
    uint64_t length;
    /* Out: the buffer's handle, its GPU address and the time the wrap took. */
    uint64_t handle;
    uint64_t gpu_address;
    uint64_t wrap_ns;
    char error[VM_ERROR_LEN];
};

struct vm_release {
    uint32_t size;
    uint32_t reserved;
    uint64_t handle;
    char error[VM_ERROR_LEN];
};

enum vm_event_op {
    /* A new MTLSharedEvent at value 0; its handle goes to `handle`. */
    VM_EVENT_NEW = 0,
    /* Sets the event's value to `value` from the CPU. */
    VM_EVENT_SIGNAL = 1,
    /* Waits on the CPU until the value is at least `value`, at most
     * `timeout_ns`; `waited_ns` says how long it took. Fails on time out. */
    VM_EVENT_WAIT = 2,
    /* Reads the value into `value`. */
    VM_EVENT_VALUE = 3,
};

struct vm_event {
    uint32_t size;
    uint32_t op;
    uint64_t handle;
    uint64_t value;
    uint64_t timeout_ns;
    uint64_t waited_ns;
    char error[VM_ERROR_LEN];
};

enum vm_op_kind {
    /* Word k of [offset, offset + 4 * words) of `buffer` = value + k. */
    VM_OP_WRITE = 0,
    /* Counts the words of that range that are not value + k into
     * `mismatches`, the lowest such k into `first_bad` (else ~0). */
    VM_OP_CHECK = 1,
    /* The GPU waits until event `other` reaches `other_value`. */
    VM_OP_WAIT_EVENT = 2,
    /* The GPU sets event `other` to `other_value` once all ops before it
     * are done. */
    VM_OP_SIGNAL_EVENT = 3,
    /* Ends the compute encoder: the next op opens a new one. */
    VM_OP_SPLIT = 4,
    /* Copies `words` words from buffer `other` at byte `other_value` to
     * `buffer` at `offset`. */
    VM_OP_COPY = 5,
};

/* One op of a run. Offsets are 4-byte aligned; `words` is at most 2^31. */
struct vm_op {
    uint32_t kind;
    uint32_t words;
    uint64_t buffer;
    uint64_t offset;
    uint32_t value;
    uint32_t mismatches;
    uint64_t other;
    uint64_t other_value;
    uint32_t first_bad;
    uint32_t reserved;
};

/* VM_RUN: `count` ops at `ops` in one command buffer, committed. With
 * `wait` the call waits and writes the results back; else `handle` names
 * the run for VM_WAIT, and the ops must stay where they are until then. */
struct vm_run {
    uint32_t size;
    uint32_t count;
    uint64_t ops;
    uint32_t wait;
    uint32_t reserved;
    uint64_t handle;
    /* Recording and committing; then, when waited, from the commit to the
     * end of the wait, and what the GPU reports between its start and end. */
    uint64_t encode_ns;
    uint64_t wait_ns;
    uint64_t gpu_ns;
    char error[VM_ERROR_LEN];
};

/* VM_WAIT: waits for run `handle`, writes its results into its ops and
 * lets the handle go. */
struct vm_wait {
    uint32_t size;
    uint32_t reserved;
    uint64_t handle;
    uint64_t wait_ns;
    uint64_t gpu_ns;
    char error[VM_ERROR_LEN];
};

_Static_assert(sizeof(struct vm_wrap) == 240, "vm_wrap layout");
_Static_assert(sizeof(struct vm_release) == 208, "vm_release layout");
_Static_assert(sizeof(struct vm_event) == 232, "vm_event layout");
_Static_assert(sizeof(struct vm_op) == 56, "vm_op layout");
_Static_assert(sizeof(struct vm_run) == 248, "vm_run layout");
_Static_assert(sizeof(struct vm_wait) == 224, "vm_wait layout");
_Static_assert(sizeof(struct vm_device) == 296, "vm_device layout");
_Static_assert(sizeof(struct vm_turns) == 232, "vm_turns layout");
_Static_assert(sizeof(struct vm_fill) == 256, "vm_fill layout");
_Static_assert(sizeof(struct vm_submit) == 208, "vm_submit layout");

#endif
