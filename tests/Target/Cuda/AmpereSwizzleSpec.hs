module Target.Cuda.AmpereSwizzleSpec (spec) where

import           Aoewif.Target.Cuda.Ampere
import qualified Aoewif.Target.Cuda.Codegen as Codegen
import           Aoewif.Target.Cuda.DSL
import           Test.Hspec                 (Spec, describe, it, shouldBe)

spec :: Spec
spec =
    describe "XOR swizzle layout" $ do
        it "renders a swizzled F16 GEMM over global A, B into an F32 C fragment" $ do
            Codegen.generateWith
                (Codegen.Config [Codegen.CudaFp16Header])
                ( kernel "swizzled_gemm" $ do
                    a <- parameter (Pointer (Const F16)) "A"
                    b <- parameter (Pointer (Const F16)) "B"
                    _ <- parameter (Pointer F32) "C"
                    n <- parameter USize "n"
                    _ <- parameter USize "m"
                    k <- parameter USize "k"
                    body $ do
                        smemA <- shared F16 "smemA" (int 128)
                        smemB <- shared F16 "smemB" (int 64)
                        accumulator <- declareFragment "c" 4
                        zeroFragment accumulator
                        smemA
                            ! swizzle 3 7 threadIdxX
                            .= a
                            ! ( (blockIdxY .* int 16 .+ (threadIdxX .>> int 3))
                                    .* k
                                    .+ (threadIdxX .& int 7)
                              )
                        if_ (threadIdxX .< int 64) $
                            smemB
                                ! swizzle 3 7 threadIdxX
                                .= b
                                ! ( (threadIdxX .>> int 3)
                                        .* n
                                        .+ blockIdxX
                                        .* int 8
                                        .+ (threadIdxX .& int 7)
                                  )
                        syncThreads
                        aFragment <- ldMatrix "a" LdX4 (smemA ! swizzle 3 7 (threadIdxX .+ int 16))
                        bFragment <- ldMatrix "b" LdX2 (smemB ! swizzle 3 7 (threadIdxX .+ int 8))
                        mma M16N8K8F16 aFragment bFragment accumulator
                        syncThreads
                )
                `shouldBe` """
                           #include <cuda_fp16.h>

                           extern "C" __global__ void swizzled_gemm(__half const* A, __half const* B, float* C, size_t n, size_t m, size_t k) {
                               __shared__ __half smemA[128];
                               __shared__ __half smemB[64];
                               uint32_t c0;
                               uint32_t c1;
                               uint32_t c2;
                               uint32_t c3;
                               (c0 = 0);
                               (c1 = 0);
                               (c2 = 0);
                               (c3 = 0);
                               (smemA[(threadIdx.x ^ ((threadIdx.x >> 3) & 7))] = A[((((blockIdx.y * 16) + (threadIdx.x >> 3)) * k) + (threadIdx.x & 7))]);
                               if ((threadIdx.x < 64)) {
                                   (smemB[(threadIdx.x ^ ((threadIdx.x >> 3) & 7))] = B[((((threadIdx.x >> 3) * n) + (blockIdx.x * 8)) + (threadIdx.x & 7))]);
                               }
                               __syncthreads();
                               uint32_t a0;
                               uint32_t a1;
                               uint32_t a2;
                               uint32_t a3;
                               asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                   : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                                   : "r"(__cvta_generic_to_shared(&smemA[((threadIdx.x + 16) ^ (((threadIdx.x + 16) >> 3) & 7))]))
                               );
                               uint32_t b0;
                               uint32_t b1;
                               asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                                   : "=r"(b0), "=r"(b1)
                                   : "r"(__cvta_generic_to_shared(&smemB[((threadIdx.x + 8) ^ (((threadIdx.x + 8) >> 3) & 7))]))
                               );
                               asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                   : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                                   : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
                               );
                               __syncthreads();
                           }

                           """
