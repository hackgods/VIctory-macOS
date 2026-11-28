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

/* What VM_TURNS draws with, made the first time. Holds `lock`. */
static id<MTLComputePipelineState> args_pipeline;
static id<MTLRenderPipelineState> draw_pipeline;
static id<MTLBuffer> draw_indices;
static id<MTLTexture> draw_target;
#define TURNS_SIZE 64

static NSString *const turns_source =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void args(device uint *a [[buffer(0)]], constant uint &record [[buffer(1)]]) {\n"
     "    device uint *r = a + record * 5;\n"
     "    r[0] = 3; r[1] = 1; r[2] = 0; r[3] = 0; r[4] = 0;\n"
     "}\n"
     "vertex float4 corner(uint id [[vertex_id]]) {\n"
     "    float2 p = float2((id << 1) & 2, id & 2);\n"
     "    return float4(p * 2 - 1, 0, 1);\n"
     "}\n"
     "fragment float4 white() { return float4(1); }\n";

static BOOL ensure_turns(char *error) {
    if (draw_target) return YES;
    if (!ensure_open(error)) return NO;
    NSError *failure = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:turns_source options:nil error:&failure];
    if (!library) {
        set_error(error, [NSString stringWithFormat:@"turns library: %@", failure]);
        return NO;
    }
    args_pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"args"]
                                                          error:&failure];
    MTLRenderPipelineDescriptor *draw = [MTLRenderPipelineDescriptor new];
    draw.vertexFunction = [library newFunctionWithName:@"corner"];
    draw.fragmentFunction = [library newFunctionWithName:@"white"];
    draw.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    if (args_pipeline)
        draw_pipeline = [device newRenderPipelineStateWithDescriptor:draw error:&failure];
    if (!args_pipeline || !draw_pipeline) {
        set_error(error, [NSString stringWithFormat:@"turns pipelines: %@", failure]);
        return NO;
    }
    static const uint16_t indices[3] = {0, 1, 2};
    draw_indices = [device newBufferWithBytes:indices length:sizeof(indices) options:MTLResourceStorageModeShared];
    MTLTextureDescriptor *target =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:TURNS_SIZE
                                                          height:TURNS_SIZE
                                                       mipmapped:NO];
    target.usage = MTLTextureUsageRenderTarget;
    target.storageMode = MTLStorageModeShared;
    draw_target = [device newTextureWithDescriptor:target];
    if (!draw_indices || !draw_target) {
        set_error(error, @"turns resources");
        return NO;
    }
    return YES;
}

static NTSTATUS vm_turns(void *args) {
    struct vm_turns *t = args;
    if (!t || t->size != sizeof(*t)) return STATUS_INVALID_PARAMETER;
    if (!t->pairs) {
        set_error(t->error, @"no pairs");
        return STATUS_INVALID_PARAMETER;
    }
    @autoreleasepool {
        pthread_mutex_lock(&lock);
        if (!ensure_turns(t->error)) {
            pthread_mutex_unlock(&lock);
            return STATUS_UNSUCCESSFUL;
        }
        /* Fresh, zeroed arguments: what the draws find is what this
         * command buffer's dispatches wrote. Tracked, as D3D12's barriers
         * order the dispatches before the draws. */
        id<MTLBuffer> records = [device newBufferWithLength:(NSUInteger)t->pairs * 20
                                                    options:MTLResourceStorageModeShared];
        memset(records.contents, 0, records.length);
        MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = draw_target;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
        uint64_t t0 = now_ns();
        id<MTLCommandBuffer> commands = [queue commandBuffer];
        id<MTLComputeCommandEncoder> compute = nil;
        id<MTLRenderCommandEncoder> render = nil;
        for (uint32_t i = 0; i < t->pairs; i++) {
            if (!compute) {
                compute = [commands computeCommandEncoder];
                [compute setComputePipelineState:args_pipeline];
                [compute setBuffer:records offset:0 atIndex:0];
            }
            [compute setBytes:&i length:sizeof(i) atIndex:1];
            [compute dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            if (!t->batched) {
                [compute endEncoding];
                compute = nil;
                pass.colorAttachments[0].loadAction = i ? MTLLoadActionLoad : MTLLoadActionClear;
                render = [commands renderCommandEncoderWithDescriptor:pass];
                [render setRenderPipelineState:draw_pipeline];
                [render drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                    indexType:MTLIndexTypeUInt16
                                  indexBuffer:draw_indices
                            indexBufferOffset:0
                               indirectBuffer:records
                         indirectBufferOffset:(NSUInteger)i * 20];
                [render endEncoding];
                render = nil;
            }
        }
        if (t->batched) {
            [compute endEncoding];
            pass.colorAttachments[0].loadAction = MTLLoadActionClear;
            render = [commands renderCommandEncoderWithDescriptor:pass];
            [render setRenderPipelineState:draw_pipeline];
            for (uint32_t i = 0; i < t->pairs; i++)
                [render drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                    indexType:MTLIndexTypeUInt16
                                  indexBuffer:draw_indices
                            indexBufferOffset:0
                               indirectBuffer:records
                         indirectBufferOffset:(NSUInteger)i * 20];
            [render endEncoding];
        }
        uint64_t t1 = now_ns();
        [commands commit];
        [commands waitUntilCompleted];
        uint64_t t2 = now_ns();
        t->encode_ns = t1 - t0;
        t->wait_ns = t2 - t1;
        t->gpu_ns = (uint64_t)((commands.GPUEndTime - commands.GPUStartTime) * 1e9);
        NTSTATUS status = STATUS_SUCCESS;
        if (commands.status != MTLCommandBufferStatusCompleted) {
            set_error(t->error, [NSString stringWithFormat:@"command buffer: %@", commands.error]);
            status = STATUS_UNSUCCESSFUL;
        }
        const uint32_t *words = records.contents;
        for (uint32_t i = 0; status == STATUS_SUCCESS && i < t->pairs; i++) {
            const uint32_t *r = words + i * 5;
            if (r[0] != 3 || r[1] != 1 || r[2] || r[3] || r[4]) {
                set_error(t->error, [NSString stringWithFormat:@"record %u not as written", i]);
                status = STATUS_UNSUCCESSFUL;
            }
        }
        uint32_t pixel = 0;
        [draw_target getBytes:&pixel
                  bytesPerRow:TURNS_SIZE * 4
                   fromRegion:MTLRegionMake2D(TURNS_SIZE / 2, TURNS_SIZE / 2, 1, 1)
                  mipmapLevel:0];
        if (status == STATUS_SUCCESS && pixel != 0xffffffffu) {
            set_error(t->error, [NSString stringWithFormat:@"the target holds %08x, not white", pixel]);
            status = STATUS_UNSUCCESSFUL;
        }
        pthread_mutex_unlock(&lock);
        return status;
    }
}

/* Objects the Windows side holds by handle: no-copy buffers, shared
 * events and runs not waited for yet. Guarded by `lock`. */
static NSMutableDictionary<NSNumber *, id> *objects;
static uint64_t next_handle = 1;

static uint64_t keep(id object) {
    if (!objects) objects = [NSMutableDictionary new];
    uint64_t h = next_handle++;
    objects[@(h)] = object;
    return h;
}

static id<MTLBuffer> held_buffer(uint64_t handle) {
    id o = objects[@(handle)];
    return [o conformsToProtocol:@protocol(MTLBuffer)] ? o : nil;
}

static id<MTLSharedEvent> held_event(uint64_t handle) {
    id o = objects[@(handle)];
    return [o conformsToProtocol:@protocol(MTLSharedEvent)] ? o : nil;
}

static NTSTATUS vm_wrap(void *args) {
    struct vm_wrap *w = args;
    if (!w || w->size != sizeof(*w)) return STATUS_INVALID_PARAMETER;
    uint64_t page = (uint64_t)getpagesize();
    if (!w->length || w->address % page || w->length % page) {
        set_error(w->error, @"the range must be whole pages");
        return STATUS_INVALID_PARAMETER;
    }
    @autoreleasepool {
        pthread_mutex_lock(&lock);
        if (!ensure_open(w->error)) {
            pthread_mutex_unlock(&lock);
            return STATUS_UNSUCCESSFUL;
        }
        MTLResourceOptions options = MTLResourceStorageModeShared;
        if (w->flags & VM_WRAP_UNTRACKED) options |= MTLResourceHazardTrackingModeUntracked;
        uint64_t t0 = now_ns();
        id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:(void *)(uintptr_t)w->address
                                                         length:w->length
                                                        options:options
                                                    deallocator:nil];
        w->wrap_ns = now_ns() - t0;
        if (!buffer) {
            pthread_mutex_unlock(&lock);
            set_error(w->error, @"newBufferWithBytesNoCopy returned nil");
            return STATUS_UNSUCCESSFUL;
        }
        if ((uint64_t)(uintptr_t)buffer.contents != w->address || buffer.length != w->length) {
            pthread_mutex_unlock(&lock);
            set_error(w->error, @"the buffer is not the memory it was made over");
            return STATUS_UNSUCCESSFUL;
        }
        w->gpu_address = buffer.gpuAddress;
        w->handle = keep(buffer);
        pthread_mutex_unlock(&lock);
        return STATUS_SUCCESS;
    }
}

static NTSTATUS vm_release(void *args) {
    struct vm_release *r = args;
    if (!r || r->size != sizeof(*r)) return STATUS_INVALID_PARAMETER;
    @autoreleasepool {
        pthread_mutex_lock(&lock);
        BOOL known = objects[@(r->handle)] != nil;
        [objects removeObjectForKey:@(r->handle)];
        pthread_mutex_unlock(&lock);
        if (!known) {
            set_error(r->error, @"no such handle");
            return STATUS_INVALID_PARAMETER;
        }
        return STATUS_SUCCESS;
    }
}

static NTSTATUS vm_event(void *args) {
    struct vm_event *e = args;
    if (!e || e->size != sizeof(*e)) return STATUS_INVALID_PARAMETER;
    @autoreleasepool {
        pthread_mutex_lock(&lock);
        if (!ensure_open(e->error)) {
            pthread_mutex_unlock(&lock);
            return STATUS_UNSUCCESSFUL;
        }
        if (e->op == VM_EVENT_NEW) {
            id<MTLSharedEvent> made = [device newSharedEvent];
            if (made) e->handle = keep(made);
            pthread_mutex_unlock(&lock);
            if (!made) set_error(e->error, @"newSharedEvent returned nil");
            return made ? STATUS_SUCCESS : STATUS_UNSUCCESSFUL;
        }
        id<MTLSharedEvent> event = held_event(e->handle);
        pthread_mutex_unlock(&lock);
        if (!event) {
            set_error(e->error, @"no such event");
            return STATUS_INVALID_PARAMETER;
        }
        switch (e->op) {
        case VM_EVENT_SIGNAL:
            event.signaledValue = e->value;
            return STATUS_SUCCESS;
        case VM_EVENT_VALUE:
            e->value = event.signaledValue;
            return STATUS_SUCCESS;
        case VM_EVENT_WAIT: {
            /* Spins for 2 ms (the latency is what gets measured), then
             * sleeps between looks. */
            uint64_t t0 = now_ns();
            while (event.signaledValue < e->value) {
                uint64_t t = now_ns() - t0;
                if (t > e->timeout_ns) {
                    e->waited_ns = t;
                    set_error(e->error, [NSString stringWithFormat:@"the event stayed at %llu, below %llu",
                                                                   event.signaledValue, e->value]);
                    return STATUS_UNSUCCESSFUL;
                }
                if (t > 2000000) usleep(50);
            }
            e->waited_ns = now_ns() - t0;
            return STATUS_SUCCESS;
        }
        default:
            set_error(e->error, @"unknown event op");
            return STATUS_INVALID_PARAMETER;
        }
    }
}

/* The op kernels, made the first time a run needs them. Holds `lock`. */
static id<MTLComputePipelineState> write_pipeline, check_pipeline, copy_pipeline;

static NSString *const ops_source =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "struct op_args { uint value; uint words; };\n"
     "kernel void write_words(device uint *out [[buffer(0)]], constant op_args &a [[buffer(1)]],\n"
     "                        uint i [[thread_position_in_grid]], uint n [[threads_per_grid]]) {\n"
     "    for (uint k = i; k < a.words; k += n) out[k] = a.value + k;\n"
     "}\n"
     "kernel void check_words(device const uint *in [[buffer(0)]], constant op_args &a [[buffer(1)]],\n"
     "                        device atomic_uint *res [[buffer(2)]],\n"
     "                        uint i [[thread_position_in_grid]], uint n [[threads_per_grid]]) {\n"
     "    uint bad = 0, first = 0xffffffffu;\n"
     "    for (uint k = i; k < a.words; k += n)\n"
     "        if (in[k] != a.value + k) { bad++; first = min(first, k); }\n"
     "    if (bad) {\n"
     "        atomic_fetch_add_explicit(&res[0], bad, memory_order_relaxed);\n"
     "        atomic_fetch_min_explicit(&res[1], first, memory_order_relaxed);\n"
     "    }\n"
     "}\n"
     "kernel void copy_words(device uint *out [[buffer(0)]], constant op_args &a [[buffer(1)]],\n"
     "                       device const uint *in [[buffer(2)]],\n"
     "                       uint i [[thread_position_in_grid]], uint n [[threads_per_grid]]) {\n"
     "    for (uint k = i; k < a.words; k += n) out[k] = in[k];\n"
     "}\n";

static BOOL ensure_ops(char *error) {
    if (copy_pipeline) return YES;
    if (!ensure_open(error)) return NO;
    NSError *failure = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:ops_source options:nil error:&failure];
    if (!library) {
        set_error(error, [NSString stringWithFormat:@"ops library: %@", failure]);
        return NO;
    }
    write_pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"write_words"]
                                                           error:&failure];
    check_pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"check_words"]
                                                           error:&failure];
    copy_pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"copy_words"]
                                                          error:&failure];
    if (!write_pipeline || !check_pipeline || !copy_pipeline) {
        write_pipeline = check_pipeline = copy_pipeline = nil;
        set_error(error, [NSString stringWithFormat:@"ops pipelines: %@", failure]);
        return NO;
    }
    return YES;
}

/* A run on its way: its command buffer, the results of its checks, and
 * the caller's ops they go back into. */
@interface VMRun : NSObject
@property(nonatomic, strong) id<MTLCommandBuffer> commands;
@property(nonatomic, strong) id<MTLBuffer> results;
@property(nonatomic) struct vm_op *ops;
@property(nonatomic) uint32_t count;
/* What its ops use, kept alive here: the command buffer does not retain. */
@property(nonatomic, strong) NSMutableArray *used;
@end
@implementation VMRun
@end

/* Small shared buffers for checks' results, reused across runs. Guarded
 * by `lock`. */
#define RESULTS_BYTES 4096
static NSMutableArray<id<MTLBuffer>> *results_pool;

static id<MTLBuffer> take_results(NSUInteger bytes) {
    if (bytes <= RESULTS_BYTES && results_pool.count) {
        id<MTLBuffer> b = results_pool.lastObject;
        [results_pool removeLastObject];
        return b;
    }
    return [device newBufferWithLength:bytes > RESULTS_BYTES ? bytes : RESULTS_BYTES
                               options:MTLResourceStorageModeShared];
}

static void give_results(id<MTLBuffer> b) {
    if (b.length != RESULTS_BYTES) return;
    pthread_mutex_lock(&lock);
    if (!results_pool) results_pool = [NSMutableArray new];
    if (results_pool.count < 16) [results_pool addObject:b];
    pthread_mutex_unlock(&lock);
}

/* Waits for a run and writes its results back. */
static NTSTATUS finish_run(VMRun *run, uint64_t *wait_ns, uint64_t *gpu_ns, char *error) {
    uint64_t t0 = now_ns();
    [run.commands waitUntilCompleted];
    *wait_ns = now_ns() - t0;
    *gpu_ns = (uint64_t)((run.commands.GPUEndTime - run.commands.GPUStartTime) * 1e9);
    if (run.commands.status != MTLCommandBufferStatusCompleted) {
        set_error(error, [NSString stringWithFormat:@"command buffer: %@", run.commands.error]);
        return STATUS_UNSUCCESSFUL;
    }
    const uint32_t *res = run.results.contents;
    for (uint32_t i = 0; i < run.count; i++)
        if (run.ops[i].kind == VM_OP_CHECK) {
            run.ops[i].mismatches = res[2 * i];
            run.ops[i].first_bad = res[2 * i + 1];
        }
    give_results(run.results);
    run.results = nil;
    run.used = nil;
    return STATUS_SUCCESS;
}

static void dispatch_op(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline,
                        uint32_t value, uint32_t words) {
    uint32_t a[2] = {value, words};
    [encoder setComputePipelineState:pipeline];
    [encoder setBytes:a length:sizeof(a) atIndex:1];
    NSUInteger threads = words < (1u << 20) ? words : (1u << 20);
    [encoder dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

/* Records one op into `encoder`, or says what is wrong with it. */
static NSString *encode_op(id<MTLComputeCommandEncoder> encoder, VMRun *run, uint32_t i) {
    struct vm_op *op = &run.ops[i];
    id<MTLBuffer> buffer = held_buffer(op->buffer);
    if (!buffer) return [NSString stringWithFormat:@"op %u: no such buffer", i];
    if (op->offset % 4 || !op->words || op->words > (1u << 31) ||
        op->offset + (uint64_t)op->words * 4 > buffer.length)
        return [NSString stringWithFormat:@"op %u: the range is outside the buffer or unaligned", i];
    [run.used addObject:buffer];
    [encoder setBuffer:buffer offset:(NSUInteger)op->offset atIndex:0];
    switch (op->kind) {
    case VM_OP_WRITE:
        dispatch_op(encoder, write_pipeline, op->value, op->words);
        break;
    case VM_OP_CHECK:
        [encoder setBuffer:run.results offset:(NSUInteger)i * 8 atIndex:2];
        dispatch_op(encoder, check_pipeline, op->value, op->words);
        break;
    case VM_OP_COPY: {
        id<MTLBuffer> source = held_buffer(op->other);
        if (!source || op->other_value % 4 || op->other_value + (uint64_t)op->words * 4 > source.length)
            return [NSString stringWithFormat:@"op %u: no such source, or the range is outside it", i];
        [run.used addObject:source];
        [encoder setBuffer:source offset:(NSUInteger)op->other_value atIndex:2];
        dispatch_op(encoder, copy_pipeline, 0, op->words);
        break;
    }
    default:
        return [NSString stringWithFormat:@"op %u: unknown kind %u", i, op->kind];
    }
    return nil;
}

static NTSTATUS vm_run(void *args) {
    struct vm_run *r = args;
    if (!r || r->size != sizeof(*r)) return STATUS_INVALID_PARAMETER;
    if (!r->count || !r->ops) {
        set_error(r->error, @"no ops");
        return STATUS_INVALID_PARAMETER;
    }
    @autoreleasepool {
        pthread_mutex_lock(&lock);
        if (!ensure_ops(r->error)) {
            pthread_mutex_unlock(&lock);
            return STATUS_UNSUCCESSFUL;
        }
        uint64_t t0 = now_ns();
        VMRun *run = [VMRun new];
        run.ops = (struct vm_op *)(uintptr_t)r->ops;
        run.count = r->count;
        run.results = take_results((NSUInteger)r->count * 8);
        run.used = [NSMutableArray new];
        uint32_t *res = run.results.contents;
        for (uint32_t i = 0; i < r->count; i++) {
            res[2 * i] = 0;
            res[2 * i + 1] = 0xffffffffu;
        }
        run.commands = [queue commandBufferWithUnretainedReferences];
        [run.used addObject:run.results];
        id<MTLComputeCommandEncoder> encoder = nil;
        BOOL encoded = NO;
        NSString *bad = nil;
        for (uint32_t i = 0; i < r->count && !bad; i++) {
            uint32_t kind = run.ops[i].kind;
            if (kind == VM_OP_WAIT_EVENT || kind == VM_OP_SIGNAL_EVENT || kind == VM_OP_SPLIT) {
                [encoder endEncoding];
                encoder = nil;
                encoded = NO;
                if (kind == VM_OP_SPLIT) continue;
                id<MTLSharedEvent> event = held_event(run.ops[i].other);
                if (event) [run.used addObject:event];
                if (!event) {
                    bad = [NSString stringWithFormat:@"op %u: no such event", i];
                } else if (kind == VM_OP_WAIT_EVENT) {
                    [run.commands encodeWaitForEvent:event value:run.ops[i].other_value];
                } else {
                    [run.commands encodeSignalEvent:event value:run.ops[i].other_value];
                }
                continue;
            }
            if (!encoder) encoder = [run.commands computeCommandEncoder];
            /* One op after another, as a D3D12 UAV barrier between them
             * would have it. */
            if (encoded) [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            bad = encode_op(encoder, run, i);
            encoded = YES;
        }
        [encoder endEncoding];
        if (bad) {
            pthread_mutex_unlock(&lock);
            set_error(r->error, bad);
            return STATUS_INVALID_PARAMETER;
        }
        [run.commands commit];
        r->encode_ns = now_ns() - t0;
        if (!r->wait) r->handle = keep(run);
        pthread_mutex_unlock(&lock);
        if (!r->wait) return STATUS_SUCCESS;
        return finish_run(run, &r->wait_ns, &r->gpu_ns, r->error);
    }
}

static NTSTATUS vm_wait(void *args) {
    struct vm_wait *w = args;
    if (!w || w->size != sizeof(*w)) return STATUS_INVALID_PARAMETER;
    @autoreleasepool {
        pthread_mutex_lock(&lock);
        id o = objects[@(w->handle)];
        VMRun *run = [o isKindOfClass:[VMRun class]] ? o : nil;
        if (run) [objects removeObjectForKey:@(w->handle)];
        pthread_mutex_unlock(&lock);
        if (!run) {
            set_error(w->error, @"no such run");
            return STATUS_INVALID_PARAMETER;
        }
        /* Outside the lock: the run may wait for an event a later call
         * signals. */
        return finish_run(run, &w->wait_ns, &w->gpu_ns, w->error);
    }
}

/* Wine looks this table up by name; the order is enum vm_call's. */
__attribute__((visibility("default"))) const unixlib_entry_t __wine_unix_call_funcs[] = {
    vm_noop,
    vm_open,
    vm_fill,
    vm_submit_empty,
    vm_turns,
    vm_wrap,
    vm_release,
    vm_event,
    vm_run,
    vm_wait,
};
_Static_assert(sizeof(__wine_unix_call_funcs) / sizeof(__wine_unix_call_funcs[0]) == VM_CALLS,
               "one entry per call");
