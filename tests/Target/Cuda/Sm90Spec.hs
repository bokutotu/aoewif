module Target.Cuda.Sm90Spec (spec) where

import qualified Aoewif.Target.Cuda.Codegen      as Codegen
import           Aoewif.Target.Cuda.DSL
import           Aoewif.Target.Cuda.Sm90
import qualified Aoewif.Target.Cuda.Syntax       as Syntax
import           Aoewif.Target.Cuda.TensorCoreOp (RenderOp (renderOp))
import           Test.Hspec                      (Spec, describe, it, shouldBe)

spec :: Spec
spec =
    describe "Sm90 instructions" $ do
        it "binds accumulator and register-A fragments to wgmma.fence" $ do
            renderOp
                0
                ( WgmmaFence
                    ( WgmmaBF16
                        WgmmaFloatN8
                        (WgmmaFragment [var "d0", var "d1", var "d2", var "d3"])
                        ( WgmmaHalfRegisterOperands
                            (WgmmaFragment [var "a0", var "a1", var "a2", var "a3"])
                            (WgmmaDescriptor (var "descriptorB"))
                            WgmmaNotTransposed
                        )
                        (int 1)
                        WgmmaScaleOne
                        WgmmaScaleOne
                    )
                )
                `shouldBe` unlines
                    [ "asm volatile(\"wgmma.fence.sync.aligned;\""
                    , "    : \"+f\"(d0), \"+f\"(d1), \"+f\"(d2), \"+f\"(d3), \"+r\"(a0), \"+r\"(a1), \"+r\"(a2), \"+r\"(a3)"
                    , "    :"
                    , "    : \"memory\""
                    , ");"
                    ]

        it "selects constraints from the cluster shared address width" $ do
            fmap
                (renderOp 0)
                [ TmaTensor
                    ( TmaTensorLoad
                        (TmaClusterShared ClusterAddressU32 (var "destination32"))
                        (TmaTensorMap (var "tensorMap"))
                        (TmaCoordinates1D (var "coordinate"))
                        (MBarrier (var "barrier32"))
                        Nothing
                    )
                , TmaTensor
                    ( TmaTensorLoad
                        (TmaClusterShared ClusterAddressU64 (var "destination64"))
                        (TmaTensorMap (var "tensorMap"))
                        (TmaCoordinates1D (var "coordinate"))
                        (MBarrier (var "barrier64"))
                        Nothing
                    )
                , MBarrierInstruction
                    ( MBarrierArriveRemote
                        ClusterAddressU64
                        (MBarrier (var "barrier"))
                        (Just (var "arrivalCount"))
                    )
                , MBarrierInstruction
                    ( MBarrierArriveExpectTxRemote
                        ClusterAddressU64
                        (MBarrier (var "barrier"))
                        (var "transactionCount")
                    )
                , MBarrierInstruction
                    ( MBarrierExpectTxRemote
                        ClusterAddressU64
                        (MBarrier (var "barrier"))
                        (var "transactionCount")
                    )
                ]
                `shouldBe` [ unlines
                                [ "asm volatile(\"cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2}], [%3];\""
                                , "    :: \"r\"(destination32), \"l\"(tensorMap), \"r\"(coordinate), \"r\"(barrier32)"
                                , "    : \"memory\""
                                , ");"
                                ]
                           , unlines
                                [ "asm volatile(\"cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2}], [%3];\""
                                , "    :: \"l\"(destination64), \"l\"(tensorMap), \"r\"(coordinate), \"l\"(barrier64)"
                                , "    : \"memory\""
                                , ");"
                                ]
                           , unlines
                                [ "asm volatile(\"mbarrier.arrive.shared::cluster.b64 _, [%0], %1;\""
                                , "    :: \"l\"(barrier), \"r\"(arrivalCount)"
                                , "    : \"memory\""
                                , ");"
                                ]
                           , unlines
                                [ "asm volatile(\"mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;\""
                                , "    :: \"l\"(barrier), \"r\"(transactionCount)"
                                , "    : \"memory\""
                                , ");"
                                ]
                           , unlines
                                [ "asm volatile(\"mbarrier.expect_tx.shared::cluster.b64 [%0], %1;\""
                                , "    :: \"l\"(barrier), \"r\"(transactionCount)"
                                , "    : \"memory\""
                                , ");"
                                ]
                           ]

        it "renders a 128-aligned 2x2-cluster BF16 GEMM smoke kernel" $ do
            let generated =
                    Codegen.generateWith
                        (Codegen.Config [Codegen.CudaBf16Header])
                        ( kernel "clustered_gemm" $ do
                            tensorMapA <- parameter USize "tensorMapA"
                            tensorMapB <- parameter USize "tensorMapB"
                            descriptorA <- parameter USize "descriptorA"
                            descriptorB <- parameter USize "descriptorB"
                            _ <- parameter USize "m"
                            _ <- parameter USize "n"
                            _ <- parameter USize "k"
                            body $ do
                                sharedA <- shared Align128 BF16 "sharedA" (int 8192)
                                sharedB <- shared Align128 BF16 "sharedB" (int 8192)
                                sharedFirstColumns <- shared Align16 BF16 "sharedFirstColumns" (int 1024)
                                barrierA <- shared NaturalAlignment USize "barrierA" (int 1)
                                barrierB <- shared NaturalAlignment USize "barrierB" (int 1)
                                ctaX <- declare U32 "ctaX"
                                ctaY <- declare U32 "ctaY"
                                readClusterSpecialRegister ctaX (ClusterCtaId ClusterX)
                                readClusterSpecialRegister ctaY (ClusterCtaId ClusterY)
                                if_ (threadIdxX .== int 0) $ do
                                    mBarrierInit (MBarrier barrierA) (int 1)
                                    mBarrierInit (MBarrier barrierB) (int 1)
                                    fenceMBarrierInit
                                clusterBarrierArrive ClusterBarrierRelease
                                clusterBarrierWait
                                if_ (threadIdxX .== int 0) $ do
                                    mBarrierArriveExpectTx
                                        Nothing
                                        (MBarrier barrierA)
                                        (int 16384)
                                    mBarrierArriveExpectTx
                                        Nothing
                                        (MBarrier barrierB)
                                        (int 16384)
                                clusterBarrierArrive ClusterBarrierRelease
                                clusterBarrierWait
                                multicastMaskA <-
                                    define
                                        U32
                                        "multicastMaskA"
                                        (ifElse (ctaY .== int 0) (int 3) (int 12))
                                multicastMaskB <-
                                    define
                                        U32
                                        "multicastMaskB"
                                        (ifElse (ctaX .== int 0) (int 5) (int 10))
                                if_
                                    ( (threadIdxX .== int 0)
                                        .&& (ctaX .== int 0)
                                    )
                                    ( tmaTensorLoad
                                        ( TmaMulticastClusterShared
                                            sharedA
                                            multicastMaskA
                                        )
                                        (TmaTensorMap tensorMapA)
                                        ( TmaCoordinates2D
                                            (int 0)
                                            (blockIdxY .* int 64)
                                        )
                                        (MBarrier barrierA)
                                        Nothing
                                    )
                                if_
                                    ( (threadIdxX .== int 0)
                                        .&& (ctaY .== int 0)
                                    )
                                    ( tmaTensorLoad
                                        ( TmaMulticastClusterShared
                                            sharedB
                                            multicastMaskB
                                        )
                                        (TmaTensorMap tensorMapB)
                                        ( TmaCoordinates2D
                                            (int 0)
                                            (blockIdxX .* int 64)
                                        )
                                        (MBarrier barrierB)
                                        Nothing
                                    )
                                let completeA = var "completeA"
                                    completeB = var "completeB"
                                for_
                                    (Just (Syntax.VarDecl U32 (Syntax.Name "completeA") (Just (int 0))))
                                    (completeA .== int 0)
                                    (Just (Syntax.Binary Syntax.Assign completeA completeA))
                                    ( mBarrierTryWaitParity
                                        completeA
                                        (MBarrier barrierA)
                                        (int 0)
                                        Nothing
                                    )
                                for_
                                    (Just (Syntax.VarDecl U32 (Syntax.Name "completeB") (Just (int 0))))
                                    (completeB .== int 0)
                                    (Just (Syntax.Binary Syntax.Assign completeB completeB))
                                    ( mBarrierTryWaitParity
                                        completeB
                                        (MBarrier barrierB)
                                        (int 0)
                                        Nothing
                                    )
                                accumulator <- declareWgmmaFragment F32 "d" 32
                                zeroWgmmaFragment accumulator
                                let mmaOperation =
                                        WgmmaBF16
                                            WgmmaFloatN64
                                            accumulator
                                            ( WgmmaHalfSharedOperands
                                                (WgmmaDescriptor descriptorA)
                                                (WgmmaDescriptor descriptorB)
                                                WgmmaNotTransposed
                                                WgmmaNotTransposed
                                            )
                                            (int 1)
                                            WgmmaScaleOne
                                            WgmmaScaleOne
                                wgmmaFence mmaOperation
                                wgmmaMmaAsync mmaOperation
                                wgmmaCommitGroup
                                wgmmaWaitGroup 0
                                warpId <-
                                    define
                                        U32
                                        "warpId"
                                        ((threadIdxX .+ blockDimX .* (threadIdxY .+ blockDimY .* threadIdxZ)) ./ int 32)
                                rowStride <- define U32 "rowStride" (int 16)
                                packedTop <-
                                    define
                                        U32
                                        "packedTop"
                                        ( cast U32 (call (var "__bfloat16_as_ushort") [call (var "__float2bfloat16_rn") [var "d0"]])
                                            .|. shiftL
                                                (cast U32 (call (var "__bfloat16_as_ushort") [call (var "__float2bfloat16_rn") [var "d1"]]))
                                                (int 16)
                                        )
                                packedBottom <-
                                    define
                                        U32
                                        "packedBottom"
                                        ( cast U32 (call (var "__bfloat16_as_ushort") [call (var "__float2bfloat16_rn") [var "d2"]])
                                            .|. shiftL
                                                (cast U32 (call (var "__bfloat16_as_ushort") [call (var "__float2bfloat16_rn") [var "d3"]]))
                                                (int 16)
                                        )
                                stMatrix
                                    (sharedFirstColumns .+ warpId .* int 16 .* rowStride)
                                    rowStride
                                    packedTop
                                stMatrix
                                    (sharedFirstColumns .+ (warpId .* int 16 .+ int 8) .* rowStride)
                                    rowStride
                                    packedBottom
                                ifElse_
                                    (warpId .== int 0)
                                    (namedBarrierSync (int 1) (int 128))
                                    (namedBarrierArrive (int 1) (int 128))
                        )
            generated
                `shouldBe` """
                           #include <stdint.h>
                           #include <cuda_bf16.h>

                           extern "C" __global__ void clustered_gemm(size_t tensorMapA, size_t tensorMapB, size_t descriptorA, size_t descriptorB, size_t m, size_t n, size_t k) {
                               __shared__ __align__(128) __nv_bfloat16 sharedA[8192];
                               __shared__ __align__(128) __nv_bfloat16 sharedB[8192];
                               __shared__ __align__(16) __nv_bfloat16 sharedFirstColumns[1024];
                               __shared__ size_t barrierA[1];
                               __shared__ size_t barrierB[1];
                               uint32_t ctaX;
                               uint32_t ctaY;
                               asm volatile("mov.u32 %0, %%cluster_ctaid.x;"
                                   : "=r"(ctaX)
                               );
                               asm volatile("mov.u32 %0, %%cluster_ctaid.y;"
                                   : "=r"(ctaY)
                               );
                               if ((threadIdx.x == 0)) {
                                   asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                                       :: "l"(__cvta_generic_to_shared(&barrierA)), "r"(1)
                                       : "memory"
                                   );
                                   asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                                       :: "l"(__cvta_generic_to_shared(&barrierB)), "r"(1)
                                       : "memory"
                                   );
                                   asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
                               }
                               asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
                               asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
                               if ((threadIdx.x == 0)) {
                                   asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                                       :: "l"(__cvta_generic_to_shared(&barrierA)), "r"(16384)
                                       : "memory"
                                   );
                                   asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                                       :: "l"(__cvta_generic_to_shared(&barrierB)), "r"(16384)
                                       : "memory"
                                   );
                               }
                               asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
                               asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
                               uint32_t multicastMaskA = ((ctaY == 0) ? 3 : 12);
                               uint32_t multicastMaskB = ((ctaX == 0) ? 5 : 10);
                               if (((threadIdx.x == 0) && (ctaX == 0))) {
                                   asm volatile("{ .reg .b16 cta_mask; .reg .b16 unused; mov.b32 {cta_mask, unused}, %5; cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], cta_mask; }"
                                       :: "l"(__cvta_generic_to_shared(&sharedA)), "l"(tensorMapA), "r"(0), "r"((blockIdx.y * 64)), "l"(__cvta_generic_to_shared(&barrierA)), "r"(multicastMaskA)
                                       : "memory"
                                   );
                               }
                               if (((threadIdx.x == 0) && (ctaY == 0))) {
                                   asm volatile("{ .reg .b16 cta_mask; .reg .b16 unused; mov.b32 {cta_mask, unused}, %5; cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], cta_mask; }"
                                       :: "l"(__cvta_generic_to_shared(&sharedB)), "l"(tensorMapB), "r"(0), "r"((blockIdx.x * 64)), "l"(__cvta_generic_to_shared(&barrierB)), "r"(multicastMaskB)
                                       : "memory"
                                   );
                               }
                               for (uint32_t completeA = 0; (completeA == 0); (completeA = completeA)) {
                                   asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.b32 %0, 1, 0, p; }"
                                       : "=r"(completeA)
                                       : "l"(__cvta_generic_to_shared(&barrierA)), "r"(0)
                                       : "memory"
                                   );
                               }
                               for (uint32_t completeB = 0; (completeB == 0); (completeB = completeB)) {
                                   asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.b32 %0, 1, 0, p; }"
                                       : "=r"(completeB)
                                       : "l"(__cvta_generic_to_shared(&barrierB)), "r"(0)
                                       : "memory"
                                   );
                               }
                               float d0;
                               float d1;
                               float d2;
                               float d3;
                               float d4;
                               float d5;
                               float d6;
                               float d7;
                               float d8;
                               float d9;
                               float d10;
                               float d11;
                               float d12;
                               float d13;
                               float d14;
                               float d15;
                               float d16;
                               float d17;
                               float d18;
                               float d19;
                               float d20;
                               float d21;
                               float d22;
                               float d23;
                               float d24;
                               float d25;
                               float d26;
                               float d27;
                               float d28;
                               float d29;
                               float d30;
                               float d31;
                               (d0 = 0);
                               (d1 = 0);
                               (d2 = 0);
                               (d3 = 0);
                               (d4 = 0);
                               (d5 = 0);
                               (d6 = 0);
                               (d7 = 0);
                               (d8 = 0);
                               (d9 = 0);
                               (d10 = 0);
                               (d11 = 0);
                               (d12 = 0);
                               (d13 = 0);
                               (d14 = 0);
                               (d15 = 0);
                               (d16 = 0);
                               (d17 = 0);
                               (d18 = 0);
                               (d19 = 0);
                               (d20 = 0);
                               (d21 = 0);
                               (d22 = 0);
                               (d23 = 0);
                               (d24 = 0);
                               (d25 = 0);
                               (d26 = 0);
                               (d27 = 0);
                               (d28 = 0);
                               (d29 = 0);
                               (d30 = 0);
                               (d31 = 0);
                               asm volatile("wgmma.fence.sync.aligned;"
                                   : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31)
                                   :
                                   : "memory"
                               );
                               asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1, 0, 0; }"
                                   : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31)
                                   : "l"(descriptorA), "l"(descriptorB), "r"(1)
                               );
                               asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                               asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                               uint32_t warpId = ((threadIdx.x + (blockDim.x * (threadIdx.y + (blockDim.y * threadIdx.z)))) / 32);
                               uint32_t rowStride = 16;
                               uint32_t packedTop = (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(d0))) | (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(d1))) << 16));
                               uint32_t packedBottom = (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(d2))) | (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(d3))) << 16));
                               // Derive the lane ID automatically from the block-local linear thread ID.
                               asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};"
                                   :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(((sharedFirstColumns + ((warpId * 16) * rowStride))) + ((threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) % 32 % 8) * (rowStride)))), "r"(packedTop)
                                   : "memory"
                               );
                               // Derive the lane ID automatically from the block-local linear thread ID.
                               asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};"
                                   :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(((sharedFirstColumns + (((warpId * 16) + 8) * rowStride))) + ((threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) % 32 % 8) * (rowStride)))), "r"(packedBottom)
                                   : "memory"
                               );
                               if ((warpId == 0)) {
                                   asm volatile("barrier.sync %0, %1;" :: "r"(1), "r"(128) : "memory");
                               } else {
                                   asm volatile("barrier.arrive %0, %1;" :: "r"(1), "r"(128) : "memory");
                               }
                           }

                           """
