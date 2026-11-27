// Bench: what one more dependent load at the heap-index choke point costs.
// mode 0: today's path: the view-stride table at the slot, then the heap at the slot.
// mode 1: proposed: the title's own record at the slot (tag, view, generation), the view
//         table at the view (generation, host slot, offset, bytes), then the heap at the host slot.
// mode 2: the heap at the slot, no table (floor).
// mode 3: no loads at all (loop and hash only).
// mode 4: proposed for views named whole in their records (#534: tag, address low, kind and
//         address high): the record at the slot, then the two-level title page table at the
//         address (the heap slot of the memory holding the page and its title address), then
//         the heap at that slot.
#define RS "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED), RootConstants(num32BitConstants=8, b0), SRV(t0), SRV(t1), SRV(t2), SRV(t3), SRV(t4), UAV(u0)"
cbuffer C : register(b0) {
    uint mode; uint accesses; uint slots; uint width;
    uint spread; uint null_slot; uint tag; uint pad;
};
StructuredBuffer<uint> strides : register(t0);
ByteAddressBuffer records : register(t1);
ByteAddressBuffer views : register(t2);
ByteAddressBuffer whole_records : register(t3);
StructuredBuffer<uint> pages : register(t4);
RWStructuredBuffer<uint> sums : register(u0);

uint hash(uint x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint thread = id.y * width + id.x;
    // Threads of one "draw" (spread threads) use the same slots, as a draw's pixels do.
    uint draw = thread / spread;
    uint acc = 0;
    for (uint k = 0; k < accesses; k++) {
        uint h = hash(draw * 0x9e3779b9u + k);
        uint slot = h % slots;
        uint at = (hash(thread + k * 977u) & 0x3ffu) * 4u;
        if (mode == 0u) {
            uint stride = strides[slot];
            uint index = stride != 0u ? slot : null_slot;
            ByteAddressBuffer b = ResourceDescriptorHeap[index];
            acc += b.Load(at);
        } else if (mode == 1u) {
            uint3 record = records.Load3(slot * 32u);
            uint4 view = views.Load4(record.y * 16u);
            bool live = record.x == tag && view.x == record.z;
            uint index = live ? view.y : null_slot;
            ByteAddressBuffer b = ResourceDescriptorHeap[index];
            acc += b.Load(view.z + min(at, view.w - 4u));
        } else if (mode == 4u) {
            uint3 record = whole_records.Load3(slot * 32u);
            // 40-bit address: low dword, high byte; 64 KiB pages, 4096 a block.
            uint block = pages[(record.z & 0xffu) << 4 | record.y >> 28];
            uint entry = (4096u + block * 4096u + ((record.y >> 16) & 0xfffu)) * 2u;
            uint index = record.x == tag ? pages[entry] : null_slot;
            ByteAddressBuffer b = ResourceDescriptorHeap[index];
            acc += b.Load(record.y - pages[entry + 1u] + at);
        } else if (mode == 2u) {
            ByteAddressBuffer b = ResourceDescriptorHeap[slot];
            acc += b.Load(at);
        } else {
            acc += h ^ at;
        }
    }
    sums[thread] = acc;
}
