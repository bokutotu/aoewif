module Target.Cuda.Sm80Spec (spec) where

import qualified Aoewif.Target.Cuda.Codegen          as Codegen
import           Aoewif.Target.Cuda.DSL
import           Aoewif.Target.Cuda.Sm80
import           Aoewif.Target.Cuda.Sm80.Instruction (Fragment (..))
import qualified Aoewif.Target.Cuda.Syntax           as Syntax
import           Aoewif.Target.Cuda.TensorCoreOp     (RenderOp (renderOp))
import           Control.Monad                       (forM, forM_)
import           Test.Hspec                          (Spec, describe, it,
                                                      shouldBe)

spec :: Spec
spec =
    describe "Sm80 tensor core instructions" $ do
        it "renders every floating-point MMA shape" $ do
            fmap
                (renderOp 0)
                [ Mma
                    M8N8K4F16
                    [var "a0", var "a1"]
                    [var "b0", var "b1"]
                    [ var "d0"
                    , var "d1"
                    , var "d2"
                    , var "d3"
                    , var "d4"
                    , var "d5"
                    , var "d6"
                    , var "d7"
                    ]
                , Mma
                    M16N8K8F16
                    [var "a0", var "a1"]
                    [var "b0"]
                    [var "d0", var "d1", var "d2", var "d3"]
                , Mma
                    M16N8K16F16
                    [var "a0", var "a1", var "a2", var "a3"]
                    [var "b0", var "b1"]
                    [var "d0", var "d1", var "d2", var "d3"]
                , Mma
                    M16N8K8BF16
                    [var "a0", var "a1"]
                    [var "b0"]
                    [var "d0", var "d1", var "d2", var "d3"]
                , Mma
                    M16N8K16BF16
                    [var "a0", var "a1", var "a2", var "a3"]
                    [var "b0", var "b1"]
                    [var "d0", var "d1", var "d2", var "d3"]
                ]
                `shouldBe` [ unlines
                                [ "asm volatile(\"mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\""
                                , "    : \"+f\"(d0), \"+f\"(d1), \"+f\"(d2), \"+f\"(d3), \"+f\"(d4), \"+f\"(d5), \"+f\"(d6), \"+f\"(d7)"
                                , "    : \"r\"(a0), \"r\"(a1), \"r\"(b0), \"r\"(b1)"
                                , ");"
                                ]
                           , unlines
                                [ "asm volatile(\"mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\""
                                , "    : \"+f\"(d0), \"+f\"(d1), \"+f\"(d2), \"+f\"(d3)"
                                , "    : \"r\"(a0), \"r\"(a1), \"r\"(b0)"
                                , ");"
                                ]
                           , unlines
                                [ "asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\""
                                , "    : \"+f\"(d0), \"+f\"(d1), \"+f\"(d2), \"+f\"(d3)"
                                , "    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"r\"(b0), \"r\"(b1)"
                                , ");"
                                ]
                           , unlines
                                [ "asm volatile(\"mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\""
                                , "    : \"+f\"(d0), \"+f\"(d1), \"+f\"(d2), \"+f\"(d3)"
                                , "    : \"r\"(a0), \"r\"(a1), \"r\"(b0)"
                                , ");"
                                ]
                           , unlines
                                [ "asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\""
                                , "    : \"+f\"(d0), \"+f\"(d1), \"+f\"(d2), \"+f\"(d3)"
                                , "    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"r\"(b0), \"r\"(b1)"
                                , ");"
                                ]
                           ]

        it "renders MMA register expressions with nested indentation" $ do
            renderOp
                2
                ( Mma
                    M16N8K16BF16
                    [ var "lhs" ! threadIdxX
                    , var "lhs" ! (threadIdxX .+ int 1)
                    , bitcast U32 (var "aPacked" ! int 0)
                    , call (var "loadA") [threadIdxX]
                    ]
                    [ var "rhs" ! (var "k" .% int 8)
                    , cast U32 (var "bits")
                    ]
                    [ var "acc" ! int 0
                    , var "acc" ! int 1
                    , var "acc" ! int 2
                    , var "acc" ! int 3
                    ]
                )
                `shouldBe` unlines
                    [ "        asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\""
                    , "            : \"+f\"(acc[0]), \"+f\"(acc[1]), \"+f\"(acc[2]), \"+f\"(acc[3])"
                    , "            : \"r\"(lhs[threadIdx.x]), \"r\"(lhs[(threadIdx.x + 1)]), \"r\"((*reinterpret_cast<uint32_t*>(&aPacked[0]))), \"r\"(loadA(threadIdx.x)), \"r\"(rhs[(k % 8)]), \"r\"(static_cast<uint32_t>(bits))"
                    , "        );"
                    ]

        it "renders every cp.async cache and size shape" $ do
            fmap
                (renderOp 0)
                [ CpAsync CacheAll4 Nothing (var "destination") (var "source")
                , CpAsync CacheAll8 Nothing (var "destination") (var "source")
                , CpAsync CacheAll16 Nothing (var "destination") (var "source")
                , CpAsync CacheGlobal16 Nothing (var "destination") (var "source")
                ]
                `shouldBe` [ unlines
                                [ "asm volatile(\"cp.async.ca.shared.global [%0], [%1], 4;\""
                                , "    :: \"l\"(__cvta_generic_to_shared(&destination)), \"l\"(&source)"
                                , ");"
                                ]
                           , unlines
                                [ "asm volatile(\"cp.async.ca.shared.global [%0], [%1], 8;\""
                                , "    :: \"l\"(__cvta_generic_to_shared(&destination)), \"l\"(&source)"
                                , ");"
                                ]
                           , unlines
                                [ "asm volatile(\"cp.async.ca.shared.global [%0], [%1], 16;\""
                                , "    :: \"l\"(__cvta_generic_to_shared(&destination)), \"l\"(&source)"
                                , ");"
                                ]
                           , unlines
                                [ "asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\""
                                , "    :: \"l\"(__cvta_generic_to_shared(&destination)), \"l\"(&source)"
                                , ");"
                                ]
                           ]

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
                        smemA <- shared Align16 F16 "smemA" (int 128)
                        smemB <- shared Align16 F16 "smemB" (int 64)
                        -- 128B XOR swizzle, composed from the raw operators.
                        let swz index = xor index (shiftR index (int 3) .&. int 7)
                        accumulator <- zeroArray F32 "c" [int 4]
                        let kk = var "kk"
                        for_
                            (Just (Syntax.VarDecl U32 (Syntax.Name "kk") (Just (int 0))))
                            (kk .< k)
                            (Just (Syntax.Binary Syntax.Assign kk (kk .+ int 16)))
                            $ do
                                cpAsync
                                    CacheGlobal16
                                    Nothing
                                    (smemA ! swz threadIdxX)
                                    ( a
                                        ! ( (blockIdxY .* int 16 .+ shiftR threadIdxX (int 3))
                                                .* k
                                                .+ kk
                                                .+ (threadIdxX .&. int 7)
                                          )
                                    )
                                cpAsync
                                    CacheGlobal16
                                    Nothing
                                    (smemB ! swz threadIdxX)
                                    ( b
                                        ! ( shiftR threadIdxX (int 3)
                                                .* n
                                                .+ blockIdxX
                                                .* int 8
                                                .+ kk
                                                .+ (threadIdxX .&. int 7)
                                          )
                                    )
                                commitGroup
                                waitGroup (Just 1)
                                syncThreads
                                aFragment <- ldMatrix "a" LdX4 LdMatrixNormal (smemA ! swz (threadIdxX .+ int 16))
                                bFragment <- ldMatrix "b" LdX2 LdMatrixTranspose (smemB ! swz (threadIdxX .+ int 8))
                                mma
                                    M16N8K8F16
                                    (Fragment (take 2 (fragmentRegisters aFragment)))
                                    (Fragment (take 1 (fragmentRegisters bFragment)))
                                    (Fragment [accumulator ! int index | index <- [0 .. 3]])
                                syncThreads
                )
                `shouldBe` """
                           #include <stdint.h>
                           #include <cuda_fp16.h>

                           extern "C" __global__ void swizzled_gemm(__half const* A, __half const* B, float* C, size_t n, size_t m, size_t k) {
                               __shared__ __align__(16) __half smemA[128];
                               __shared__ __align__(16) __half smemB[64];
                               float c[4] = {};
                               for (uint32_t kk = 0; (kk < k); (kk = (kk + 16))) {
                                   asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                                       :: "l"(__cvta_generic_to_shared(&smemA[(threadIdx.x ^ ((threadIdx.x >> 3) & 7))])), "l"(&A[(((((blockIdx.y * 16) + (threadIdx.x >> 3)) * k) + kk) + (threadIdx.x & 7))])
                                   );
                                   asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                                       :: "l"(__cvta_generic_to_shared(&smemB[(threadIdx.x ^ ((threadIdx.x >> 3) & 7))])), "l"(&B[(((((threadIdx.x >> 3) * n) + (blockIdx.x * 8)) + kk) + (threadIdx.x & 7))])
                                   );
                                   asm volatile("cp.async.commit_group;");
                                   asm volatile("cp.async.wait_group 1;");
                                   __syncthreads();
                                   uint32_t a[4] = {};
                                   asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                       : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                                       : "l"(__cvta_generic_to_shared(&smemA[((threadIdx.x + 16) ^ (((threadIdx.x + 16) >> 3) & 7))]))
                                   );
                                   uint32_t b[2] = {};
                                   asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                                       : "=r"(b[0]), "=r"(b[1])
                                       : "l"(__cvta_generic_to_shared(&smemB[((threadIdx.x + 8) ^ (((threadIdx.x + 8) >> 3) & 7))]))
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                                       : "r"(a[0]), "r"(a[1]), "r"(b[0])
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
                        smemA <- shared Align16 F16 "smemA" (int 2048)
                        smemB <- shared Align16 F16 "smemB" (int 2048)
                        accumulator <- zeroArray F32 "c" [int 2, int 4, int 4]
                        let accFrags =
                                [ [ Fragment [accumulator ! int r16 ! int c8 ! int j | j <- [0 .. 3]]
                                  | c8 <- [0 .. 3]
                                  ]
                                | r16 <- [0, 1]
                                ]
                            tid = threadIdxX
                            warp = threadIdxY
                            warpRow = shiftR warp (int 1)
                            warpCol = warp .&. int 1
                            -- 128B XOR swizzle on a 16B-granule index, composed
                            -- from the raw operators.
                            swz granule = xor granule (shiftR granule (int 3) .&. int 7)
                            loadA kk =
                                cpAsync
                                    CacheGlobal16
                                    Nothing
                                    ( smemA
                                        ! ( (shiftR kk (int 4) .&. int 1)
                                                .* int 1024
                                                .+ swz tid
                                                .* int 8
                                          )
                                    )
                                    ( a
                                        ! ( (blockIdxY .* int 64 .+ shiftR tid (int 1))
                                                .* k
                                                .+ kk
                                                .+ (tid .&. int 1)
                                                .* int 8
                                          )
                                    )
                            loadB kk =
                                cpAsync
                                    CacheGlobal16
                                    Nothing
                                    ( smemB
                                        ! ( (shiftR kk (int 4) .&. int 1)
                                                .* int 1024
                                                .+ swz tid
                                                .* int 8
                                          )
                                    )
                                    ( b
                                        ! ( (kk .+ shiftR tid (int 3))
                                                .* n
                                                .+ blockIdxX
                                                .* int 64
                                                .+ (tid .&. int 7)
                                                .* int 8
                                          )
                                    )
                            computeStage kk = do
                                let stage = shiftR kk (int 4) .&. int 1
                                aFrags <-
                                    forM [0 :: Int, 1] $ \r16 ->
                                        ldMatrix
                                            ("a" ++ show r16)
                                            LdX4
                                            LdMatrixNormal
                                            ( smemA
                                                ! ( stage
                                                        .* int 1024
                                                        .+ swz
                                                            ( warpRow
                                                                .* int 64
                                                                .+ int (fromIntegral r16 * 32)
                                                                .+ shiftR tid (int 3)
                                                                .* int 8
                                                                .+ (tid .&. int 7)
                                                            )
                                                        .* int 8
                                                  )
                                            )
                                bFrags <-
                                    forM [0 :: Int .. 3] $ \c8 ->
                                        ldMatrix
                                            ("b" ++ show c8)
                                            LdX2
                                            LdMatrixTranspose
                                            ( smemB
                                                ! ( stage
                                                        .* int 1024
                                                        .+ swz
                                                            ( warpCol
                                                                .* int 4
                                                                .+ int (fromIntegral c8)
                                                                .+ shiftR tid (int 3)
                                                                .* int 8
                                                                .+ (tid .&. int 7)
                                                            )
                                                        .* int 8
                                                  )
                                            )
                                forM_ [0 :: Int, 1] $ \r16 ->
                                    forM_ [0 :: Int .. 3] $ \c8 -> do
                                        let aRegs = fragmentRegisters (aFrags !! r16)
                                            bRegs = fragmentRegisters (bFrags !! c8)
                                            acc = accFrags !! r16 !! c8
                                        forM_
                                            (zip3 (take 2 aRegs) (drop 2 aRegs) bRegs)
                                            $ \(aLowRegister, aHighRegister, bRegister) ->
                                                mma
                                                    M16N8K8F16
                                                    (Fragment [aLowRegister, aHighRegister])
                                                    (Fragment [bRegister])
                                                    acc
                        loadA (int 0)
                        loadB (int 0)
                        commitGroup
                        let kk = var "kk"
                        for_
                            (Just (Syntax.VarDecl U32 (Syntax.Name "kk") (Just (int 0))))
                            (kk .< k)
                            (Just (Syntax.Binary Syntax.Assign kk (kk .+ int 16)))
                            $ do
                                ifElse_
                                    (kk .+ int 16 .< k)
                                    ( do
                                        loadA (kk .+ int 16)
                                        loadB (kk .+ int 16)
                                        commitGroup
                                        waitGroup (Just 1)
                                    )
                                    (waitGroup (Just 0))
                                namedBarrierSync (int 1) (blockDimX .* blockDimY .* blockDimZ)
                                computeStage kk
                                namedBarrierSync (int 1) (blockDimX .* blockDimY .* blockDimZ)
                        let storeTile r16 c8 =
                                forM_ [0 :: Integer .. 3] $ \j ->
                                    ( c
                                        ! ( ( blockIdxY
                                                .* int 64
                                                .+ warpRow
                                                .* int 32
                                                .+ int (fromIntegral r16 * 16)
                                                .+ shiftR (int j) (int 1)
                                                .* int 8
                                                .+ shiftR tid (int 2)
                                            )
                                                .* n
                                                .+ blockIdxX
                                                .* int 64
                                                .+ warpCol
                                                .* int 32
                                                .+ int (fromIntegral c8 * 8)
                                                .+ (int j .&. int 1)
                                                .* int 2
                                                .+ (tid .&. int 3)
                                                .* int 2
                                          )
                                    )
                                        .= accumulator
                                        ! int (fromIntegral r16)
                                        ! int (fromIntegral c8)
                                        ! int j
                        forM_ [0 :: Int, 1] $ \r16 ->
                            forM_ [0 :: Int .. 3] $ \c8 ->
                                storeTile r16 c8
                )
                `shouldBe` """
                           #include <stdint.h>
                           #include <cuda_fp16.h>

                           extern "C" __global__ void gemm_f16_pipeline2(__half const* A, __half const* B, float* C, size_t n, size_t m, size_t k) {
                               __shared__ __align__(16) __half smemA[2048];
                               __shared__ __align__(16) __half smemB[2048];
                               float c[2][4][4] = {};
                               asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                                   :: "l"(__cvta_generic_to_shared(&smemA[((((0 >> 4) & 1) * 1024) + ((threadIdx.x ^ ((threadIdx.x >> 3) & 7)) * 8))])), "l"(&A[(((((blockIdx.y * 64) + (threadIdx.x >> 1)) * k) + 0) + ((threadIdx.x & 1) * 8))])
                               );
                               asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                                   :: "l"(__cvta_generic_to_shared(&smemB[((((0 >> 4) & 1) * 1024) + ((threadIdx.x ^ ((threadIdx.x >> 3) & 7)) * 8))])), "l"(&B[((((0 + (threadIdx.x >> 3)) * n) + (blockIdx.x * 64)) + ((threadIdx.x & 7) * 8))])
                               );
                               asm volatile("cp.async.commit_group;");
                               for (uint32_t kk = 0; (kk < k); (kk = (kk + 16))) {
                                   if (((kk + 16) < k)) {
                                       asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                                           :: "l"(__cvta_generic_to_shared(&smemA[(((((kk + 16) >> 4) & 1) * 1024) + ((threadIdx.x ^ ((threadIdx.x >> 3) & 7)) * 8))])), "l"(&A[(((((blockIdx.y * 64) + (threadIdx.x >> 1)) * k) + (kk + 16)) + ((threadIdx.x & 1) * 8))])
                                       );
                                       asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                                           :: "l"(__cvta_generic_to_shared(&smemB[(((((kk + 16) >> 4) & 1) * 1024) + ((threadIdx.x ^ ((threadIdx.x >> 3) & 7)) * 8))])), "l"(&B[(((((kk + 16) + (threadIdx.x >> 3)) * n) + (blockIdx.x * 64)) + ((threadIdx.x & 7) * 8))])
                                       );
                                       asm volatile("cp.async.commit_group;");
                                       asm volatile("cp.async.wait_group 1;");
                                   } else {
                                       asm volatile("cp.async.wait_group 0;");
                                   }
                                   asm volatile("barrier.sync %0, %1;" :: "r"(1), "r"(((blockDim.x * blockDim.y) * blockDim.z)) : "memory");
                                   uint32_t a0[4] = {};
                                   asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                       : "=r"(a0[0]), "=r"(a0[1]), "=r"(a0[2]), "=r"(a0[3])
                                       : "l"(__cvta_generic_to_shared(&smemA[((((kk >> 4) & 1) * 1024) + (((((((threadIdx.y >> 1) * 64) + 0) + ((threadIdx.x >> 3) * 8)) + (threadIdx.x & 7)) ^ (((((((threadIdx.y >> 1) * 64) + 0) + ((threadIdx.x >> 3) * 8)) + (threadIdx.x & 7)) >> 3) & 7)) * 8))]))
                                   );
                                   uint32_t a1[4] = {};
                                   asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                       : "=r"(a1[0]), "=r"(a1[1]), "=r"(a1[2]), "=r"(a1[3])
                                       : "l"(__cvta_generic_to_shared(&smemA[((((kk >> 4) & 1) * 1024) + (((((((threadIdx.y >> 1) * 64) + 32) + ((threadIdx.x >> 3) * 8)) + (threadIdx.x & 7)) ^ (((((((threadIdx.y >> 1) * 64) + 32) + ((threadIdx.x >> 3) * 8)) + (threadIdx.x & 7)) >> 3) & 7)) * 8))]))
                                   );
                                   uint32_t b0[2] = {};
                                   asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                                       : "=r"(b0[0]), "=r"(b0[1])
                                       : "l"(__cvta_generic_to_shared(&smemB[((((kk >> 4) & 1) * 1024) + (((((((threadIdx.y & 1) * 4) + 0) + ((threadIdx.x >> 3) * 8)) + (threadIdx.x & 7)) ^ (((((((threadIdx.y & 1) * 4) + 0) + ((threadIdx.x >> 3) * 8)) + (threadIdx.x & 7)) >> 3) & 7)) * 8))]))
                                   );
                                   uint32_t b1[2] = {};
                                   asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                                       : "=r"(b1[0]), "=r"(b1[1])
                                       : "l"(__cvta_generic_to_shared(&smemB[((((kk >> 4) & 1) * 1024) + (((((((threadIdx.y & 1) * 4) + 1) + ((threadIdx.x >> 3) * 8)) + (threadIdx.x & 7)) ^ (((((((threadIdx.y & 1) * 4) + 1) + ((threadIdx.x >> 3) * 8)) + (threadIdx.x & 7)) >> 3) & 7)) * 8))]))
                                   );
                                   uint32_t b2[2] = {};
                                   asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                                       : "=r"(b2[0]), "=r"(b2[1])
                                       : "l"(__cvta_generic_to_shared(&smemB[((((kk >> 4) & 1) * 1024) + (((((((threadIdx.y & 1) * 4) + 2) + ((threadIdx.x >> 3) * 8)) + (threadIdx.x & 7)) ^ (((((((threadIdx.y & 1) * 4) + 2) + ((threadIdx.x >> 3) * 8)) + (threadIdx.x & 7)) >> 3) & 7)) * 8))]))
                                   );
                                   uint32_t b3[2] = {};
                                   asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                                       : "=r"(b3[0]), "=r"(b3[1])
                                       : "l"(__cvta_generic_to_shared(&smemB[((((kk >> 4) & 1) * 1024) + (((((((threadIdx.y & 1) * 4) + 3) + ((threadIdx.x >> 3) * 8)) + (threadIdx.x & 7)) ^ (((((((threadIdx.y & 1) * 4) + 3) + ((threadIdx.x >> 3) * 8)) + (threadIdx.x & 7)) >> 3) & 7)) * 8))]))
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+f"(c[0][0][0]), "+f"(c[0][0][1]), "+f"(c[0][0][2]), "+f"(c[0][0][3])
                                       : "r"(a0[0]), "r"(a0[2]), "r"(b0[0])
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+f"(c[0][0][0]), "+f"(c[0][0][1]), "+f"(c[0][0][2]), "+f"(c[0][0][3])
                                       : "r"(a0[1]), "r"(a0[3]), "r"(b0[1])
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+f"(c[0][1][0]), "+f"(c[0][1][1]), "+f"(c[0][1][2]), "+f"(c[0][1][3])
                                       : "r"(a0[0]), "r"(a0[2]), "r"(b1[0])
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+f"(c[0][1][0]), "+f"(c[0][1][1]), "+f"(c[0][1][2]), "+f"(c[0][1][3])
                                       : "r"(a0[1]), "r"(a0[3]), "r"(b1[1])
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+f"(c[0][2][0]), "+f"(c[0][2][1]), "+f"(c[0][2][2]), "+f"(c[0][2][3])
                                       : "r"(a0[0]), "r"(a0[2]), "r"(b2[0])
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+f"(c[0][2][0]), "+f"(c[0][2][1]), "+f"(c[0][2][2]), "+f"(c[0][2][3])
                                       : "r"(a0[1]), "r"(a0[3]), "r"(b2[1])
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+f"(c[0][3][0]), "+f"(c[0][3][1]), "+f"(c[0][3][2]), "+f"(c[0][3][3])
                                       : "r"(a0[0]), "r"(a0[2]), "r"(b3[0])
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+f"(c[0][3][0]), "+f"(c[0][3][1]), "+f"(c[0][3][2]), "+f"(c[0][3][3])
                                       : "r"(a0[1]), "r"(a0[3]), "r"(b3[1])
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+f"(c[1][0][0]), "+f"(c[1][0][1]), "+f"(c[1][0][2]), "+f"(c[1][0][3])
                                       : "r"(a1[0]), "r"(a1[2]), "r"(b0[0])
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+f"(c[1][0][0]), "+f"(c[1][0][1]), "+f"(c[1][0][2]), "+f"(c[1][0][3])
                                       : "r"(a1[1]), "r"(a1[3]), "r"(b0[1])
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+f"(c[1][1][0]), "+f"(c[1][1][1]), "+f"(c[1][1][2]), "+f"(c[1][1][3])
                                       : "r"(a1[0]), "r"(a1[2]), "r"(b1[0])
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+f"(c[1][1][0]), "+f"(c[1][1][1]), "+f"(c[1][1][2]), "+f"(c[1][1][3])
                                       : "r"(a1[1]), "r"(a1[3]), "r"(b1[1])
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+f"(c[1][2][0]), "+f"(c[1][2][1]), "+f"(c[1][2][2]), "+f"(c[1][2][3])
                                       : "r"(a1[0]), "r"(a1[2]), "r"(b2[0])
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+f"(c[1][2][0]), "+f"(c[1][2][1]), "+f"(c[1][2][2]), "+f"(c[1][2][3])
                                       : "r"(a1[1]), "r"(a1[3]), "r"(b2[1])
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+f"(c[1][3][0]), "+f"(c[1][3][1]), "+f"(c[1][3][2]), "+f"(c[1][3][3])
                                       : "r"(a1[0]), "r"(a1[2]), "r"(b3[0])
                                   );
                                   asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                       : "+f"(c[1][3][0]), "+f"(c[1][3][1]), "+f"(c[1][3][2]), "+f"(c[1][3][3])
                                       : "r"(a1[1]), "r"(a1[3]), "r"(b3[1])
                                   );
                                   asm volatile("barrier.sync %0, %1;" :: "r"(1), "r"(((blockDim.x * blockDim.y) * blockDim.z)) : "memory");
                               }
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((0 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 0) + ((0 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[0][0][0]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((1 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 0) + ((1 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[0][0][1]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((2 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 0) + ((2 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[0][0][2]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((3 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 0) + ((3 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[0][0][3]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((0 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 8) + ((0 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[0][1][0]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((1 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 8) + ((1 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[0][1][1]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((2 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 8) + ((2 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[0][1][2]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((3 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 8) + ((3 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[0][1][3]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((0 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 16) + ((0 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[0][2][0]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((1 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 16) + ((1 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[0][2][1]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((2 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 16) + ((2 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[0][2][2]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((3 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 16) + ((3 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[0][2][3]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((0 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 24) + ((0 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[0][3][0]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((1 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 24) + ((1 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[0][3][1]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((2 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 24) + ((2 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[0][3][2]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 0) + ((3 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 24) + ((3 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[0][3][3]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((0 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 0) + ((0 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[1][0][0]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((1 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 0) + ((1 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[1][0][1]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((2 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 0) + ((2 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[1][0][2]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((3 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 0) + ((3 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[1][0][3]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((0 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 8) + ((0 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[1][1][0]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((1 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 8) + ((1 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[1][1][1]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((2 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 8) + ((2 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[1][1][2]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((3 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 8) + ((3 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[1][1][3]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((0 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 16) + ((0 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[1][2][0]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((1 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 16) + ((1 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[1][2][1]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((2 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 16) + ((2 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[1][2][2]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((3 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 16) + ((3 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[1][2][3]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((0 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 24) + ((0 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[1][3][0]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((1 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 24) + ((1 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[1][3][1]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((2 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 24) + ((2 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[1][3][2]);
                               (C[(((((((((((blockIdx.y * 64) + ((threadIdx.y >> 1) * 32)) + 16) + ((3 >> 1) * 8)) + (threadIdx.x >> 2)) * n) + (blockIdx.x * 64)) + ((threadIdx.y & 1) * 32)) + 24) + ((3 & 1) * 2)) + ((threadIdx.x & 3) * 2))] = c[1][3][3]);
                           }

                           """

        it "renders an in-place movmatrix for every register in a fragment" $ do
            Codegen.generateWith
                (Codegen.Config [Codegen.CudaFp16Header])
                ( kernel "transpose_fragment" $ do
                    body $ do
                        tile <- shared Align16 F16 "tile" (int 64)
                        fragment <-
                            ldMatrix
                                "fragment"
                                LdX2
                                LdMatrixNormal
                                (tile ! threadIdxX)
                        movMatrix fragment
                )
                `shouldBe` """
                           #include <stdint.h>
                           #include <cuda_fp16.h>

                           extern "C" __global__ void transpose_fragment() {
                               __shared__ __align__(16) __half tile[64];
                               uint32_t fragment[2] = {};
                               asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                                   : "=r"(fragment[0]), "=r"(fragment[1])
                                   : "l"(__cvta_generic_to_shared(&tile[threadIdx.x]))
                               );
                               asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %0;"
                                   : "+r"(fragment[0])
                               );
                               asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %0;"
                                   : "+r"(fragment[1])
                               );
                           }

                           """

        it "renders cp.async with zero-fill for partial k-tiles" $ do
            Codegen.generateWith
                (Codegen.Config [Codegen.CudaFp16Header])
                ( kernel "zfill" $ do
                    source <- parameter (Pointer (Const F16)) "source"
                    remaining <- parameter U32 "remaining"
                    body $ do
                        tile <- shared Align16 F16 "tile" (int 64)
                        cpAsync
                            CacheAll16
                            (Just remaining)
                            (tile ! threadIdxX)
                            (source ! threadIdxX)
                )
                `shouldBe` """
                           #include <stdint.h>
                           #include <cuda_fp16.h>

                           extern "C" __global__ void zfill(__half const* source, uint32_t remaining) {
                               __shared__ __align__(16) __half tile[64];
                               asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                                   :: "l"(__cvta_generic_to_shared(&tile[threadIdx.x])), "l"(&source[threadIdx.x]), "r"(remaining)
                               );
                           }

                           """
