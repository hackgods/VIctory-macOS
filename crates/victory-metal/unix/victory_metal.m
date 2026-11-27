/* victory-metal's Wine unixlib: the macOS side of the bridge from
 * VIctory's Windows code to Metal (VIctory-macOS #221).
 *
 * Wine pairs a builtin PE module with a Mach-O library of the same name in
 * the x86_64-unix folder beside it, and hands the PE side the table below
 * (NtQueryVirtualMemory with MemoryWineUnixFuncs). A call through
 * __wine_unix_call_dispatcher runs one entry on the calling thread, on its
 * Unix stack, in this process: the memory the Windows side owns is at the
 * same addresses here, which is what lets Metal use it in place. D3DMetal
 * is reached the same way (its d3d12.dll is a builtin with a unixlib).
 *
 * Built by tools/macos/build-metal-bridge.sh for x86_64, as the Wine
 * process runs under Rosetta. Everything here is ours; nothing comes from
 * the Game Porting Toolkit. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include "abi.h"

typedef int32_t NTSTATUS;
typedef NTSTATUS (*unixlib_entry_t)(void *args);

#define STATUS_SUCCESS 0
#define STATUS_UNSUCCESSFUL ((NTSTATUS)0xC0000001)
#define STATUS_INVALID_PARAMETER ((NTSTATUS)0xC000000D)

/* One device, queue and pipeline for the process, made by the first
 * VM_OPEN and kept. */
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLComputePipelineState> fill_pipeline;

static NSString *const fill_source =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "struct fill_args { uint value; uint first; uint words; };\n"
     "kernel void fill(device uint *out [[buffer(0)]],\n"
     "                 constant fill_args &a [[buffer(1)]],\n"
     "                 uint i [[thread_position_in_grid]]) {\n"
     "    if (i < a.words) out[i] = a.value + a.first + i;\n"
     "}\n";

static uint64_t now_ns(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

static void set_error(char *error, NSString *what) {
    snprintf(error, VM_ERROR_LEN, "%s", what.UTF8String ? what.UTF8String : "(no message)");
}

/* Makes the device, queue and pipeline if they are not made yet. Holds
 * `lock`. */
static BOOL ensure_open(char *error) {
    if (fill_pipeline) return YES;
    if (!device) device = MTLCreateSystemDefaultDevice();
    if (!device) {
        set_error(error, @"MTLCreateSystemDefaultDevice returned nil");
        return NO;
    }
    if (!queue) queue = [device newCommandQueue];
    if (!queue) {
        set_error(error, @"newCommandQueue returned nil");
        return NO;
    }
    NSError *failure = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:fill_source options:nil error:&failure];
    id<MTLFunction> function = [library newFunctionWithName:@"fill"];
    if (!function) {
        set_error(error, [NSString stringWithFormat:@"fill kernel: %@", failure]);
        return NO;
    }
    fill_pipeline = [device newComputePipelineStateWithFunction:function error:&failure];
    if (!fill_pipeline) {
        set_error(error, [NSString stringWithFormat:@"fill pipeline: %@", failure]);
        return NO;
    }
    return YES;
}

static NTSTATUS vm_noop(void *args) {
    (void)args;
    return STATUS_SUCCESS;
}

static NTSTATUS vm_open(void *args) {
    struct vm_device *d = args;
    if (!d || d->size != sizeof(*d)) return STATUS_INVALID_PARAMETER;
    @autoreleasepool {
        pthread_mutex_lock(&lock);
        BOOL ok = ensure_open(d->error);
        if (ok) {
            d->unified_memory = device.hasUnifiedMemory;
            d->registry_id = device.registryID;
            d->max_buffer_length = device.maxBufferLength;
            d->recommended_working_set = device.recommendedMaxWorkingSetSize;
            d->page_size = (uint64_t)getpagesize();
            snprintf(d->name, sizeof(d->name), "%s", device.name.UTF8String);
        }
        pthread_mutex_unlock(&lock);
        return ok ? STATUS_SUCCESS : STATUS_UNSUCCESSFUL;
    }
}

static NTSTATUS vm_fill(void *args) {
    struct vm_fill *f = args;
    if (!f || f->size != sizeof(*f)) return STATUS_INVALID_PARAMETER;
    uint64_t page = (uint64_t)getpagesize();
    if (!f->dispatches || !f->length || f->address % page || f->length % page ||
        (f->length / 4) % f->dispatches) {
        set_error(f->error, @"the range must be whole pages and split evenly by the dispatches");
        return STATUS_INVALID_PARAMETER;
    }
    @autoreleasepool {
        pthread_mutex_lock(&lock);
        if (!ensure_open(f->error)) {
            pthread_mutex_unlock(&lock);
            return STATUS_UNSUCCESSFUL;
        }
        uint64_t t0 = now_ns();
        /* Shared storage is the only kind a no-copy buffer can have; hazards
         * are left untracked, as a backend that records D3D12's barriers as
         * Metal barriers would have them. */
        id<MTLBuffer> buffer =
            [device newBufferWithBytesNoCopy:(void *)(uintptr_t)f->address
                                      length:f->length
                                     options:MTLResourceStorageModeShared |
                                             MTLResourceHazardTrackingModeUntracked
                                 deallocator:nil];
        uint64_t t1 = now_ns();
        if (!buffer) {
            pthread_mutex_unlock(&lock);
            set_error(f->error, @"newBufferWithBytesNoCopy returned nil");
            return STATUS_UNSUCCESSFUL;
        }
        uint32_t words = (uint32_t)(f->length / 4 / f->dispatches);
        NSUInteger group = f->threads_per_group ? f->threads_per_group : 64;
        if (group > fill_pipeline.maxTotalThreadsPerThreadgroup)
            group = fill_pipeline.maxTotalThreadsPerThreadgroup;
        id<MTLCommandBuffer> commands = [queue commandBufferWithUnretainedReferences];
        /* The dispatches write disjoint words, so they may overlap. */
        id<MTLComputeCommandEncoder> encoder =
            [commands computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        [encoder setComputePipelineState:fill_pipeline];
        [encoder useResource:buffer usage:MTLResourceUsageWrite];
        for (uint32_t i = 0; i < f->dispatches; i++) {
            uint32_t a[3] = {f->value, i * words, words};
            [encoder setBuffer:buffer offset:(NSUInteger)i * words * 4 atIndex:0];
            [encoder setBytes:a length:sizeof(a) atIndex:1];
            [encoder dispatchThreadgroups:MTLSizeMake((words + group - 1) / group, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
        }
        [encoder endEncoding];
        uint64_t t2 = now_ns();
        [commands commit];
        [commands waitUntilCompleted];
        uint64_t t3 = now_ns();
        f->wrap_ns = t1 - t0;
        f->encode_ns = t2 - t1;
        f->wait_ns = t3 - t2;
        f->gpu_ns = (uint64_t)((commands.GPUEndTime - commands.GPUStartTime) * 1e9);
        NSError *failure = commands.error;
        pthread_mutex_unlock(&lock);
        if (commands.status != MTLCommandBufferStatusCompleted) {
            set_error(f->error, [NSString stringWithFormat:@"command buffer: %@", failure]);
            return STATUS_UNSUCCESSFUL;
        }
        return STATUS_SUCCESS;
    }
}

static NTSTATUS vm_submit_empty(void *args) {
    struct vm_submit *s = args;
    if (!s || s->size != sizeof(*s)) return STATUS_INVALID_PARAMETER;
    @autoreleasepool {
        pthread_mutex_lock(&lock);
        if (!ensure_open(s->error)) {
            pthread_mutex_unlock(&lock);
            return STATUS_UNSUCCESSFUL;
        }
        uint64_t t0 = now_ns();
        for (uint32_t i = 0; i < s->count; i++) {
            @autoreleasepool {
                id<MTLCommandBuffer> commands = [queue commandBufferWithUnretainedReferences];
                [commands commit];
                [commands waitUntilCompleted];
            }
        }
        s->total_ns = now_ns() - t0;
        pthread_mutex_unlock(&lock);
        return STATUS_SUCCESS;
    }
}

/* Wine looks this table up by name; the order is enum vm_call's. */
__attribute__((visibility("default"))) const unixlib_entry_t __wine_unix_call_funcs[] = {
    vm_noop,
    vm_open,
    vm_fill,
    vm_submit_empty,
};
_Static_assert(sizeof(__wine_unix_call_funcs) / sizeof(__wine_unix_call_funcs[0]) == VM_CALLS,
               "one entry per call");
