#include <stdint.h>
#include <cuda/ptx>
#include <cuda_bf16.h>

// <<<1, 128>>>
extern "C" __global__ void gemm_bf16_128x32x16(
   __nv_bfloat16 const* A,
   __nv_bfloat16 const* B,
   float* D
) {
   __shared__ __align__(16) __nv_bfloat16 sA[2048];
   __shared__ __align__(16) __nv_bfloat16 sB[512];
   __shared__ __align__(16) size_t done[1];
   __shared__ uint32_t tmem[1];

   uint32_t tid = threadIdx.x;

   // TMEMの確保はwarp全体で実行する。
   if (tid < 32) {
       cuda::ptx::tcgen05_alloc(cuda::ptx::cta_group_1, tmem, 32);
   }
   if (tid == 0) {
       cuda::ptx::mbarrier_init(
           reinterpret_cast<uint64_t*>(&done[0]), 1);
   }

   // A/Bをno-swizzle、K-majorの8×8サブタイル配置へ変換する。
   for (uint32_t i = tid; i < 2048; i = i + 128) {
       uint32_t m = i / 16;
       uint32_t k = i % 16;
       uint32_t dst =
           (m / 8) * 128 + (k / 8) * 64 + (m % 8) * 8 + k % 8;
       sA[dst] = A[i];
   }
   for (uint32_t i = tid; i < 512; i = i + 128) {
       uint32_t k = i / 32;
       uint32_t n = i % 32;
       uint32_t dst =
           (n / 8) * 128 + (k / 8) * 64 + (n % 8) * 8 + k % 8;
       sB[dst] = B[i];
   }

   cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
   __syncthreads();
   cuda::ptx::tcgen05_fence_after_thread_sync();

   uint32_t base = tmem[0];

   // MMAとcommitは1スレッドだけが発行する。
   if (tid == 0) {
       uint32_t addrA =
           static_cast<uint32_t>(__cvta_generic_to_shared(sA));
       uint32_t addrB =
           static_cast<uint32_t>(__cvta_generic_to_shared(sB));

       // Leading offset = 128 bytes、stride offset = 256 bytes。
       // bit 46は仕様で定められた固定値。
       size_t descBits =
             (static_cast<size_t>(8) << 16)
           | (static_cast<size_t>(16) << 32)
           | (static_cast<size_t>(1) << 46);

       size_t descA =
           static_cast<size_t>((addrA & 0x3ffff) >> 4) | descBits;
       size_t descB =
           static_cast<size_t>((addrB & 0x3ffff) >> 4) | descBits;

       // M=128、N=32、BF16入力、FP32累積、dense、転置なし。
       uint32_t idesc =
             (8u << 24) | (4u << 17)
           | (1u << 10) | (1u << 7) | (1u << 4);

       uint32_t disableOutputLane[4] = {};

       cuda::ptx::tcgen05_mma(
           cuda::ptx::kind_f16,
           cuda::ptx::cta_group_1,
           base, descA, descB, idesc,
           disableOutputLane,
           false);  // 既存の累積値を読まず、D = A × Bにする。

       cuda::ptx::tcgen05_commit(
           cuda::ptx::cta_group_1,
           reinterpret_cast<uint64_t*>(&done[0]));
   }

   bool ready = false;
   for (; !ready;) {
       ready = cuda::ptx::mbarrier_try_wait_parity(
           reinterpret_cast<uint64_t*>(&done[0]), 0);
   }
   cuda::ptx::tcgen05_fence_after_thread_sync();

   // 各warpが担当する32 laneを読み、各threadがDの1行を書く。
   uint32_t warpBase = base + (((tid / 32) * 32) << 16);
   uint32_t value[1] = {};

   for (uint32_t n = 0; n < 32; n = n + 1) {
       cuda::ptx::tcgen05_ld_32x32b(value, warpBase + n);
       cuda::ptx::tcgen05_wait_ld();
       D[tid * 32 + n] = __uint_as_float(value[0]);
   }

   // 全warpの読み出し完了後に解放する。
   __syncthreads();
   if (tid < 32) {
       cuda::ptx::tcgen05_dealloc(cuda::ptx::cta_group_1, base, 32);
   }
}
