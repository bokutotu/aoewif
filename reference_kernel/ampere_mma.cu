#include <stdint.h>
#include <cuda/ptx>
#include <cuda_fp16.h>

extern "C" __global__ void ampere_mma(__half const* matrixA, __half const* matrixB, float* matrixD, uint32_t M, uint32_t N, uint32_t K) {
    uint32_t const lane = threadIdx.x;
    uint32_t const groupId = (lane / 4);
    uint32_t const threadInGroup = (lane % 4);
    uint32_t const tileRow = (blockIdx.y * 16);
    uint32_t const tileColumn = (blockIdx.x * 8);
    __shared__ __align__(16) __half sharedA[(16 * 16)];
    __shared__ __align__(16) __half sharedB[(16 * 8)];
    float fragmentD[4] = {};
    uint32_t const addressA = static_cast<uint32_t>(__cvta_generic_to_shared((sharedA + (((lane % 16) * 16) + ((lane / 16) * 8)))));
    uint32_t const addressB = static_cast<uint32_t>(__cvta_generic_to_shared((sharedB + ((lane % 16) * 8))));
    for (uint32_t tileK = 0; (tileK < K); (tileK = (tileK + 16))) {
        uint32_t const row = (tileRow + (lane / 2));
        uint32_t const column = (tileK + ((lane % 2) * 8));
        uint32_t const destination = static_cast<uint32_t>(__cvta_generic_to_shared((sharedA + (lane * 8))));
        __half const* const source = (matrixA + ((static_cast<size_t>(row) * K) + column));
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;\n"
            :
            : "r"(destination), "l"(source)
            : "memory"
        );
        if ((lane < 16)) {
            uint32_t const rowB = (tileK + lane);
            uint32_t const destinationB = static_cast<uint32_t>(__cvta_generic_to_shared((sharedB + (lane * 8))));
            __half const* const sourceB = (matrixB + ((static_cast<size_t>(rowB) * N) + tileColumn));
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16;\n"
                :
                : "r"(destinationB), "l"(sourceB)
                : "memory"
            );
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile(
            "cp.async.wait_group 0;\n"
            ::: "memory"
        );
        __syncwarp();
        uint32_t fragmentA[4] = {};
        uint32_t fragmentB[2] = {};
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(fragmentA[0]), "=r"(fragmentA[1]), "=r"(fragmentA[2]), "=r"(fragmentA[3])
            : "r"(addressA)
            : "memory"
        );
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(fragmentB[0]), "=r"(fragmentB[1])
            : "r"(addressB)
            : "memory"
        );
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(fragmentD[0]), "+f"(fragmentD[1]), "+f"(fragmentD[2]), "+f"(fragmentD[3])
            : "r"(fragmentA[0]), "r"(fragmentA[1]), "r"(fragmentA[2]), "r"(fragmentA[3]), "r"(fragmentB[0]), "r"(fragmentB[1])
        );
        __syncwarp();
    }
    for (uint32_t element = 0; (element < 4); (element = (element + 1))) {
        uint32_t const row = ((tileRow + groupId) + ((element / 2) * 8));
        uint32_t const column = ((tileColumn + (threadInGroup * 2)) + (element % 2));
        (matrixD[((static_cast<size_t>(row) * N) + column)] = fragmentD[element]);
    }
}
