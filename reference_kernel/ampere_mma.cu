#include <stdint.h>
#include <cuda/ptx>
#include <cuda_fp16.h>

extern "C" __global__ void ampere_mma(__half const* matrixA, __half const* matrixB, float* matrixD) {
    uint32_t const groupId = (threadIdx.x / 4);
    uint32_t const threadInGroup = (threadIdx.x % 4);
    uint32_t fragmentA[4] = {};
    for (uint32_t packedIndex = 0; (packedIndex < 4); (packedIndex = (packedIndex + 1))) {
        uint32_t const row = (groupId + ((packedIndex % 2) * 8));
        uint32_t const column = ((threadInGroup * 2) + ((packedIndex / 2) * 8));
        (fragmentA[packedIndex] = (static_cast<uint32_t>(__half_as_ushort(matrixA[((row * 16) + column)])) | (static_cast<uint32_t>(__half_as_ushort(matrixA[(((row * 16) + column) + 1)])) << 16)));
    }
    uint32_t fragmentB[2] = {};
    float fragmentD[4] = {};
    for (uint32_t packedIndex = 0; (packedIndex < 2); (packedIndex = (packedIndex + 1))) {
        uint32_t const row = ((threadInGroup * 2) + (packedIndex * 8));
        uint32_t const column = groupId;
        (fragmentB[packedIndex] = (static_cast<uint32_t>(__half_as_ushort(matrixB[((row * 8) + column)])) | (static_cast<uint32_t>(__half_as_ushort(matrixB[(((row + 1) * 8) + column)])) << 16)));
    }
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(fragmentD[0]), "+f"(fragmentD[1]), "+f"(fragmentD[2]), "+f"(fragmentD[3])
        : "r"(fragmentA[0]), "r"(fragmentA[1]), "r"(fragmentA[2]), "r"(fragmentA[3]), "r"(fragmentB[0]), "r"(fragmentB[1])
    );
    for (uint32_t element = 0; (element < 4); (element = (element + 1))) {
        uint32_t const row = (groupId + ((element / 2) * 8));
        uint32_t const column = ((threadInGroup * 2) + (element % 2));
        (matrixD[((row * 8) + column)] = fragmentD[element]);
    }
}
