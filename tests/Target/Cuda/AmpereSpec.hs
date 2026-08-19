module Target.Cuda.AmpereSpec (spec) where

import           Aoewif.Target.Cuda.Ampere
import           Aoewif.Target.Cuda.Ampere.Instruction (Fragment (..))
import qualified Aoewif.Target.Cuda.Codegen            as Codegen
import           Aoewif.Target.Cuda.DSL
import           Control.Monad                         (forM, forM_)
import           Test.Hspec                            (Spec, describe, it,
                                                        shouldBe)

spec :: Spec
spec =
    describe "Ampere tensor core instructions" $ do
        it "renders a swizzled GEMM pipelined with cp.async over a k-loop" $ do
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
                        -- 128B XOR swizzle, composed from the raw operators.
                        let swz index = index .^ ((index .>> int 3) .& int 7)
                        accumulator <- declareFragment "c" 4
                        zeroFragment accumulator
                        for_ (define U32 "kk" (int 0)) (.< k) (\kk -> kk .+ int 16) $ \kk -> do
                            cpAsync
                                CacheGlobal
                                Bytes16
                                Nothing
                                (smemA ! swz threadIdxX)
                                ( a
                                    ! ( (blockIdxY .* int 16 .+ (threadIdxX .>> int 3))
                                            .* k
                                            .+ kk
                                            .+ (threadIdxX .& int 7)
                                      )
                                )
                            cpAsync
                                CacheGlobal
                                Bytes16
                                Nothing
                                (smemB ! swz threadIdxX)
                                ( b
                                    ! ( (threadIdxX .>> int 3)
                                            .* n
                                            .+ blockIdxX
                                            .* int 8
                                            .+ kk
                                            .+ (threadIdxX .& int 7)
                                      )
                                )
                            commitGroup
                            waitGroup (Just 1)
                            syncThreads
                            aFragment <- ldMatrix "a" LdX4 (smemA ! swz (threadIdxX .+ int 16))
                            bFragment <- ldMatrix "b" LdX2 (smemB ! swz (threadIdxX .+ int 8))
                            mma
                                M16N8K8F16
                                (Fragment (take 2 (fragmentRegisters aFragment)))
                                (Fragment (take 1 (fragmentRegisters bFragment)))
                                accumulator
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
                               for (uint32_t kk = 0; (kk < k); (kk = (kk + 16))) {
                                   asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                                       :: "r"(__cvta_generic_to_shared(&smemA[(threadIdx.x ^ ((threadIdx.x >> 3) & 7))])), "l"(&A[(((((blockIdx.y * 16) + (threadIdx.x >> 3)) * k) + kk) + (threadIdx.x & 7))])
                                   );
                                   asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                                       :: "r"(__cvta_generic_to_shared(&smemB[(threadIdx.x ^ ((threadIdx.x >> 3) & 7))])), "l"(&B[(((((threadIdx.x >> 3) * n) + (blockIdx.x * 8)) + kk) + (threadIdx.x & 7))])
                                   );
                                   asm volatile("cp.async.commit_group;");
                                   asm volatile("cp.async.wait_group 1;");
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
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                                       : "r"(a0), "r"(a1), "r"(b0)
                                   );
                                   __syncthreads();
                               }
                           }

                           """

        -- 64x64 CTA, 4 warps (32x4), warp tile 32x32, k-stage 16, 2-stage double
        -- buffering. A and B stages are stored as 16B rows, one granule per thread,
        -- swizzled over 128B groups; ldmatrix addresses follow the per-thread row
        -- rule of the PTX spec.
        it "renders a double-buffered cp.async pipelined GEMM with ldmatrix and mma" $ do
            Codegen.generateWith
                (Codegen.Config [Codegen.CudaFp16Header])
                ( kernel "gemm_f16_pipeline2" $ do
                    a <- parameter (Pointer (Const F16)) "A"
                    b <- parameter (Pointer (Const F16)) "B"
                    c <- parameter (Pointer F32) "C"
                    n <- parameter USize "n"
                    _ <- parameter USize "m"
                    k <- parameter USize "k"
                    body $ do
                        smemA <- shared F16 "smemA" (int 2048)
                        smemB <- shared F16 "smemB" (int 2048)
                        accFrags <- forM [0 :: Int .. 7] $ \i -> declareFragment ("c" ++ show i) 4
                        mapM_ zeroFragment accFrags
                        let tid = threadIdxX
                            warp = threadIdxY
                            warpRow = warp .>> int 1
                            warpCol = warp .& int 1
                            -- 128B XOR swizzle on a 16B-granule index, composed
                            -- from the raw operators.
                            swz granule = granule .^ ((granule .>> int 3) .& int 7)
                            loadA kk =
                                cpAsync
                                    CacheGlobal
                                    Bytes16
                                    Nothing
                                    ( smemA
                                        ! ( ((kk .>> int 4) .& int 1)
                                                .* int 1024
                                                .+ swz tid
                                                .* int 8
                                          )
                                    )
                                    ( a
                                        ! ( (blockIdxY .* int 64 .+ (tid .>> int 1))
                                                .* k
                                                .+ kk
                                                .+ (tid .& int 1)
                                                .* int 8
                                          )
                                    )
                            loadB kk =
                                cpAsync
                                    CacheGlobal
                                    Bytes16
                                    Nothing
                                    ( smemB
                                        ! ( ((kk .>> int 4) .& int 1)
                                                .* int 1024
                                                .+ swz tid
                                                .* int 8
                                          )
                                    )
                                    ( b
                                        ! ( (kk .+ (tid .>> int 3))
                                                .* n
                                                .+ blockIdxX
                                                .* int 64
                                                .+ (tid .& int 7)
                                                .* int 8
                                          )
                                    )
                            computeStage kk = do
                                let stage = (kk .>> int 4) .& int 1
                                aFrags <-
                                    forM [0 :: Int, 1] $ \r16 ->
                                        ldMatrix
                                            ("a" ++ show r16)
                                            LdX4
                                            ( smemA
                                                ! ( stage
                                                        .* int 1024
                                                        .+ swz
                                                            ( warpRow
                                                                .* int 64
                                                                .+ int (fromIntegral r16 * 32)
                                                                .+ (tid .>> int 3)
                                                                .* int 8
                                                                .+ (tid .& int 7)
                                                            )
                                                        .* int 8
                                                  )
                                            )
                                bFrags <-
                                    forM [0 :: Int .. 3] $ \c8 ->
                                        ldMatrix
                                            ("b" ++ show c8)
                                            LdX2
                                            ( smemB
                                                ! ( stage
                                                        .* int 1024
                                                        .+ swz
                                                            ( warpCol
                                                                .* int 4
                                                                .+ int (fromIntegral c8)
                                                                .+ (tid .>> int 3)
                                                                .* int 8
                                                                .+ (tid .& int 7)
                                                            )
                                                        .* int 8
                                                  )
                                            )
                                forM_ [0 :: Int, 1] $ \r16 ->
                                    forM_ [0 :: Int .. 3] $ \c8 -> do
                                        let aRegs = fragmentRegisters (aFrags !! r16)
                                            bRegs = fragmentRegisters (bFrags !! c8)
                                            acc = accFrags !! (r16 * 4 + c8)
                                        mma M16N8K8F16 (Fragment [head aRegs, aRegs !! 2]) (Fragment [head bRegs]) acc
                                        mma M16N8K8F16 (Fragment [aRegs !! 1, aRegs !! 3]) (Fragment [bRegs !! 1]) acc
                        loadA (int 0)
                        loadB (int 0)
                        commitGroup
                        for_ (define U32 "kk" (int 0)) (.< k) (\kk -> kk .+ int 16) $ \kk -> do
                            ifElse_
                                (kk .+ int 16 .< k)
                                ( do
                                    loadA (kk .+ int 16)
                                    loadB (kk .+ int 16)
                                    commitGroup
                                    waitGroup (Just 1)
                                )
                                (waitGroup (Just 0))
                            syncThreads
                            computeStage kk
                            syncThreads
                        let storeTile r16 c8 = do
                                let frag = accFrags !! (r16 * 4 + c8)
                                forM_ [0 :: Integer .. 3] $ \j ->
                                    ( c
                                        ! ( ( blockIdxY
                                                .* int 64
                                                .+ warpRow
                                                .* int 32
                                                .+ int (fromIntegral r16 * 16)
                                                .+ (int j .>> int 1)
                                                .* int 8
                                                .+ (tid .>> int 2)
                                            )
                                                .* n
                                                .+ blockIdxX
                                                .* int 64
                                                .+ warpCol
                                                .* int 32
                                                .+ int (fromIntegral c8 * 8)
                                                .+ (int j .& int 1)
                                                .* int 2
                                                .+ (tid .& int 3)
                                                .* int 2
                                          )
                                    )
                                        .= bitcast F32 (fragmentRegisters frag !! fromIntegral j)
                        forM_ [0 :: Int, 1] $ \r16 ->
                            forM_ [0 :: Int .. 3] $ \c8 ->
                                storeTile r16 c8
                )
                `shouldBe` """
                           #include <cuda_fp16.h>

                           extern "C" __global__ void gemm_f16_pipeline2(__half const* A, __half const* B, float* C, size_t n, size_t m, size_t k) {
                               __shared__ __half smemA[2048];
                               __shared__ __half smemB[2048];
                               uint32_t c00;
                               uint32_t c01;
                               uint32_t c02;
                               uint32_t c03;
                               uint32_t c10;
                               uint32_t c11;
                               uint32_t c12;
                               uint32_t c13;
                               uint32_t c20;
                               uint32_t c21;
                               uint32_t c22;
                               uint32_t c23;
                               uint32_t c30;
                               uint32_t c31;
                               uint32_t c32;
                               uint32_t c33;
                               uint32_t c40;
                               uint32_t c41;
                               uint32_t c42;
                               uint32_t c43;
                               uint32_t c50;
                               uint32_t c51;
                               uint32_t c52;
                               uint32_t c53;
                               uint32_t c60;
                               uint32_t c61;
                               uint32_t c62;
                               uint32_t c63;
                               uint32_t c70;
                               uint32_t c71;
                               uint32_t c72;
                               uint32_t c73;
                               (c00 = 0);
                               (c01 = 0);
                               (c02 = 0);
                               (c03 = 0);
                               (c10 = 0);
                               (c11 = 0);
                               (c12 = 0);
                               (c13 = 0);
                               (c20 = 0);
                               (c21 = 0);
                               (c22 = 0);
                               (c23 = 0);
                               (c30 = 0);
                               (c31 = 0);
                               (c32 = 0);
                               (c33 = 0);
                               (c40 = 0);
                               (c41 = 0);
                               (c42 = 0);
                               (c43 = 0);
                               (c50 = 0);
                               (c51 = 0);
                               (c52 = 0);
                               (c53 = 0);
                               (c60 = 0);
                               (c61 = 0);
                               (c62 = 0);
                               (c63 = 0);
                               (c70 = 0);
                               (c71 = 0);
                               (c72 = 0);
                               (c73 = 0);
                               asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                                   :: "r"(__cvta_generic_to_shared(&smemA[((((0 >> 4) & 1) * 1024) + ((threadIdx.x ^ ((threadIdx.x >> 3) & 7)) * 8))])), "l"(&A[(((((blockIdx.y * 64) + (threadIdx.x >> 1)) * k) + 0) + ((threadIdx.x & 1) * 8))])
                               );
                               asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                                   :: "r"(__cvta_generic_to_shared(&smemB[((((0 >> 4) & 1) * 1024) + ((threadIdx.x ^ ((threadIdx.x >> 3) & 7)) * 8))])), "l"(&B[((((0 + (threadIdx.x >> 3)) * n) + (blockIdx.x * 64)) + ((threadIdx.x & 7) * 8))])
                               );
                               asm volatile("cp.async.commit_group;");
                               for (uint32_t kk = 0; (kk < k); (kk = (kk + 16))) {
                                   if (((kk + 16) < k)) {
                                       asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                                           :: "r"(__cvta_generic_to_shared(&smemA[(((((kk + 16) >> 4) & 1) * 1024) + ((threadIdx.x ^ ((threadIdx.x >> 3) & 7)) * 8))])), "l"(&A[(((((blockIdx.y * 64) + (threadIdx.x >> 1)) * k) + (kk + 16)) + ((threadIdx.x & 1) * 8))])
                                       );
                                       asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                                           :: "r"(__cvta_generic_to_shared(&smemB[(((((kk + 16) >> 4) & 1) * 1024) + ((threadIdx.x ^ ((threadIdx.x >> 3) & 7)) * 8))])), "l"(&B[(((((kk + 16) + (threadIdx.x >> 3)) * n) + (blockIdx.x * 64)) + ((threadIdx.x & 7) * 8))])
                                       );
                                       asm volatile("cp.async.commit_group;");
                                       asm volatile("cp.async.wait_group 1;");
                                   } else {
                                       asm volatile("cp.async.wait_group 0;");
                                   }
                                   __syncthreads();
                                   uint32_t a00;
                                   uint32_t a01;
                                   uint32_t a02;
                                   uint32_t a03;
                                   asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                       : "=r"(a00), "=r"(a01), "=r"(a02), "=r"(a03)
                                       : "r"(__cvta_generic_to_shared(&smemA[((((kk >> 4) & 1) * 1024) + (((((((threadIdx.y >> 1) * 64) + 0) + ((threadIdx.x >> 3) * 8)) + (threadIdx.x & 7)) ^ (((((((threadIdx.y >> 1) * 64) + 0) + ((threadIdx.x >> 3) * 8)) + (threadIdx.x & 7)) >> 3) & 7)) * 8))]))
                                   );
                                   uint32_t a10;
                                   uint32_t a11;
                                   uint32_t a12;
                                   uint32_t a13;
                                   asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                       : "=r"(a10), "=r"(a11), "=r"(a12), "=r"(a13)
                                       : "r"(__cvta_generic_to_shared(&smemA[((((kk >> 4) & 1) * 1024) + (((((((threadIdx.y >> 1) * 64) + 32) + ((threadIdx.x >> 3) * 8)) + (threadIdx.x & 7)) ^ (((((((threadIdx.y >> 1) * 64) + 32) + ((threadIdx.x >> 3) * 8)) + (threadIdx.x & 7)) >> 3) & 7)) * 8))]))
                                   );
                                   uint32_t b00;
                                   uint32_t b01;
                                   asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                                       : "=r"(b00), "=r"(b01)
                                       : "r"(__cvta_generic_to_shared(&smemB[((((kk >> 4) & 1) * 1024) + (((((((threadIdx.y & 1) * 4) + 0) + ((threadIdx.x >> 3) * 8)) + (threadIdx.x & 7)) ^ (((((((threadIdx.y & 1) * 4) + 0) + ((threadIdx.x >> 3) * 8)) + (threadIdx.x & 7)) >> 3) & 7)) * 8))]))
                                   );
                                   uint32_t b10;
                                   uint32_t b11;
                                   asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                                       : "=r"(b10), "=r"(b11)
                                       : "r"(__cvta_generic_to_shared(&smemB[((((kk >> 4) & 1) * 1024) + (((((((threadIdx.y & 1) * 4) + 1) + ((threadIdx.x >> 3) * 8)) + (threadIdx.x & 7)) ^ (((((((threadIdx.y & 1) * 4) + 1) + ((threadIdx.x >> 3) * 8)) + (threadIdx.x & 7)) >> 3) & 7)) * 8))]))
                                   );
                                   uint32_t b20;
                                   uint32_t b21;
                                   asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                                       : "=r"(b20), "=r"(b21)
                                       : "r"(__cvta_generic_to_shared(&smemB[((((kk >> 4) & 1) * 1024) + (((((((threadIdx.y & 1) * 4) + 2) + ((threadIdx.x >> 3) * 8)) + (threadIdx.x & 7)) ^ (((((((threadIdx.y & 1) * 4) + 2) + ((threadIdx.x >> 3) * 8)) + (threadIdx.x & 7)) >> 3) & 7)) * 8))]))
                                   );
                                   uint32_t b30;
                                   uint32_t b31;
                                   asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                                       : "=r"(b30), "=r"(b31)
                                       : "r"(__cvta_generic_to_shared(&smemB[((((kk >> 4) & 1) * 1024) + (((((((threadIdx.y & 1) * 4) + 3) + ((threadIdx.x >> 3) * 8)) + (threadIdx.x & 7)) ^ (((((((threadIdx.y & 1) * 4) + 3) + ((threadIdx.x >> 3) * 8)) + (threadIdx.x & 7)) >> 3) & 7)) * 8))]))
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+r"(c00), "+r"(c01), "+r"(c02), "+r"(c03)
                                       : "r"(a00), "r"(a02), "r"(b00)
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+r"(c00), "+r"(c01), "+r"(c02), "+r"(c03)
                                       : "r"(a01), "r"(a03), "r"(b01)
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+r"(c10), "+r"(c11), "+r"(c12), "+r"(c13)
                                       : "r"(a00), "r"(a02), "r"(b10)
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+r"(c10), "+r"(c11), "+r"(c12), "+r"(c13)
                                       : "r"(a01), "r"(a03), "r"(b11)
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+r"(c20), "+r"(c21), "+r"(c22), "+r"(c23)
                                       : "r"(a00), "r"(a02), "r"(b20)
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+r"(c20), "+r"(c21), "+r"(c22), "+r"(c23)
                                       : "r"(a01), "r"(a03), "r"(b21)
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+r"(c30), "+r"(c31), "+r"(c32), "+r"(c33)
                                       : "r"(a00), "r"(a02), "r"(b30)
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+r"(c30), "+r"(c31), "+r"(c32), "+r"(c33)
                                       : "r"(a01), "r"(a03), "r"(b31)
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+r"(c40), "+r"(c41), "+r"(c42), "+r"(c43)
                                       : "r"(a10), "r"(a12), "r"(b00)
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+r"(c40), "+r"(c41), "+r"(c42), "+r"(c43)
                                       : "r"(a11), "r"(a13), "r"(b01)
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+r"(c50), "+r"(c51), "+r"(c52), "+r"(c53)
                                       : "r"(a10), "r"(a12), "r"(b10)
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+r"(c50), "+r"(c51), "+r"(c52), "+r"(c53)
                                       : "r"(a11), "r"(a13), "r"(b11)
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+r"(c60), "+r"(c61), "+r"(c62), "+r"(c63)
                                       : "r"(a10), "r"(a12), "r"(b20)
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+r"(c60), "+r"(c61), "+r"(c62), "+r"(c63)
                                       : "r"(a11), "r"(a13), "r"(b21)
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+r"(c70), "+r"(c71), "+r"(c72), "+r"(c73)
                                       : "r"(a10), "r"(a12), "r"(b30)
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+r"(c70), "+r"(c71), "+r"(c72), "+r"(c73)
                                       : "r"(a11), "r"(a13), "r"(b31)
                                   );
                                   __syncthreads();
                               }
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((0 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 0) + ((0 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c00));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((1 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 0) + ((1 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c01));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((2 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 0) + ((2 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c02));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((3 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 0) + ((3 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c03));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((0 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 8) + ((0 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c10));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((1 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 8) + ((1 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c11));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((2 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 8) + ((2 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c12));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((3 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 8) + ((3 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c13));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((0 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 16) + ((0 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c20));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((1 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 16) + ((1 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c21));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((2 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 16) + ((2 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c22));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((3 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 16) + ((3 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c23));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((0 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 24) + ((0 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c30));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((1 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 24) + ((1 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c31));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((2 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 24) + ((2 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c32));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((3 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 24) + ((3 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c33));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((0 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 0) + ((0 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c40));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((1 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 0) + ((1 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c41));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((2 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 0) + ((2 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c42));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((3 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 0) + ((3 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c43));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((0 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 8) + ((0 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c50));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((1 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 8) + ((1 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c51));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((2 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 8) + ((2 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c52));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((3 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 8) + ((3 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c53));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((0 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 16) + ((0 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c60));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((1 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 16) + ((1 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c61));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((2 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 16) + ((2 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c62));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((3 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 16) + ((3 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c63));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((0 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 24) + ((0 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c70));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((1 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 24) + ((1 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c71));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((2 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 24) + ((2 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c72));
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((3 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 24) + ((3 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = *reinterpret_cast<float*>(&c73));
                           }

                           """

        it "renders cp.async with zero-fill for partial k-tiles" $ do
            Codegen.generateWith
                (Codegen.Config [Codegen.CudaFp16Header])
                ( kernel "zfill" $ do
                    source <- parameter (Pointer (Const F16)) "source"
                    remaining <- parameter U32 "remaining"
                    body $ do
                        tile <- shared F16 "tile" (int 64)
                        cpAsync
                            CacheAll
                            Bytes16
                            (Just remaining)
                            (tile ! threadIdxX)
                            (source ! threadIdxX)
                )
                `shouldBe` """
                           #include <cuda_fp16.h>

                           extern "C" __global__ void zfill(__half const* source, uint32_t remaining) {
                               __shared__ __half tile[64];
                               asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                                   :: "r"(__cvta_generic_to_shared(&tile[threadIdx.x])), "l"(&source[threadIdx.x]), "r"(remaining)
                               );
                           }

                           """
