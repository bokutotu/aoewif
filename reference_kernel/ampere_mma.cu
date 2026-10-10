#include <stdint.h>
#include <cuda/ptx>
#include <cuda_fp16.h>

extern "C" __global__ void ampere_mma(__half const* matrix_a, __half const* matrix_b, float* matrix_d) {
    uint32_t const group_id = (threadIdx.x / 4);
    uint32_t const thread_in_group = (threadIdx.x % 4);
    uint32_t fragment_a[4] = {};
    for (uint32_t packed_index = 0; (packed_index < 4); (packed_index = (packed_index + 1))) {
        uint32_t const row = (group_id + ((packed_index % 2) * 8));
        uint32_t const column = ((thread_in_group * 2) + ((packed_index / 2) * 8));
        (fragment_a[packed_index] = (static_cast<uint32_t>(__half_as_ushort(matrix_a[((row * 16) + column)])) | (static_cast<uint32_t>(__half_as_ushort(matrix_a[(((row * 16) + column) + 1)])) << 16)));
    }
    uint32_t fragment_b[2] = {};
    float fragment_d[4] = {};
    for (uint32_t packed_index = 0; (packed_index < 2); (packed_index = (packed_index + 1))) {
        uint32_t const row = ((thread_in_group * 2) + (packed_index * 8));
        uint32_t const column = group_id;
        (fragment_b[packed_index] = (static_cast<uint32_t>(__half_as_ushort(matrix_b[((row * 8) + column)])) | (static_cast<uint32_t>(__half_as_ushort(matrix_b[(((row + 1) * 8) + column)])) << 16)));
    }
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(fragment_d[0]), "+f"(fragment_d[1]), "+f"(fragment_d[2]), "+f"(fragment_d[3])
        : "r"(fragment_a[0]), "r"(fragment_a[1]), "r"(fragment_a[2]), "r"(fragment_a[3]), "r"(fragment_b[0]), "r"(fragment_b[1])
    );
    for (uint32_t element = 0; (element < 4); (element = (element + 1))) {
        uint32_t const row = (group_id + ((element / 2) * 8));
        uint32_t const column = ((thread_in_group * 2) + (element % 2));
        (matrix_d[((row * 8) + column)] = fragment_d[element]);
    }
}
