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
                                accumulator <- zeroArray F32 "d" [int 32]
                                let mmaOperation =
                                        WgmmaBF16
                                            WgmmaFloatN64
                                            (WgmmaFragment [accumulator ! int index | index <- [0 .. 31]])
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
                                        ( cast U32 (call (var "__bfloat16_as_ushort") [call (var "__float2bfloat16_rn") [accumulator ! int 0]])
                                            .|. shiftL
                                                (cast U32 (call (var "__bfloat16_as_ushort") [call (var "__float2bfloat16_rn") [accumulator ! int 1]]))
                                                (int 16)
                                        )
                                packedBottom <-
                                    define
                                        U32
                                        "packedBottom"
                                        ( cast U32 (call (var "__bfloat16_as_ushort") [call (var "__float2bfloat16_rn") [accumulator ! int 2]])
                                            .|. shiftL
                                                (cast U32 (call (var "__bfloat16_as_ushort") [call (var "__float2bfloat16_rn") [accumulator ! int 3]]))
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
                           #include <cuda/ptx>
                           #include <cuda_bf16.h>

                           extern "C" __global__ void clustered_gemm(size_t tensorMapA, size_t tensorMapB, size_t descriptorA, size_t descriptorB, size_t m, size_t n, size_t k) {
                               __shared__ __align__(128) __nv_bfloat16 sharedA[8192];
                               __shared__ __align__(128) __nv_bfloat16 sharedB[8192];
                               __shared__ __align__(16) __nv_bfloat16 sharedFirstColumns[1024];
                               __shared__ size_t barrierA[1];
                               __shared__ size_t barrierB[1];
                               uint32_t ctaX;
                               uint32_t ctaY;
                               ctaX = cuda::ptx::get_sreg_cluster_ctaid_x();
                               ctaY = cuda::ptx::get_sreg_cluster_ctaid_y();
                               if ((threadIdx.x == 0)) {
                                   cuda::ptx::mbarrier_init(reinterpret_cast<uint64_t*>(&barrierA), 1);
                                   cuda::ptx::mbarrier_init(reinterpret_cast<uint64_t*>(&barrierB), 1);
                                   cuda::ptx::fence_mbarrier_init(cuda::ptx::sem_release, cuda::ptx::scope_cluster);
                               }
                               asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
                               asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
                               if ((threadIdx.x == 0)) {
                                   cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared, reinterpret_cast<uint64_t*>(&barrierA), 16384);
                                   cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared, reinterpret_cast<uint64_t*>(&barrierB), 16384);
                               }
                               asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
                               asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
                               uint32_t multicastMaskA = ((ctaY == 0) ? 3 : 12);
                               uint32_t multicastMaskB = ((ctaX == 0) ? 5 : 10);
                               if (((threadIdx.x == 0) && (ctaX == 0))) {
                                   cuda::ptx::cp_async_bulk_tensor(
                                       cuda::ptx::space_cluster, cuda::ptx::space_global,
                                       &sharedA, reinterpret_cast<const void*>(tensorMapA),
                                       {static_cast<int32_t>(0), static_cast<int32_t>((blockIdx.y * 64))}, reinterpret_cast<uint64_t*>(&barrierA), static_cast<uint16_t>(multicastMaskA));
                               }
                               if (((threadIdx.x == 0) && (ctaY == 0))) {
                                   cuda::ptx::cp_async_bulk_tensor(
                                       cuda::ptx::space_cluster, cuda::ptx::space_global,
                                       &sharedB, reinterpret_cast<const void*>(tensorMapB),
                                       {static_cast<int32_t>(0), static_cast<int32_t>((blockIdx.x * 64))}, reinterpret_cast<uint64_t*>(&barrierB), static_cast<uint16_t>(multicastMaskB));
                               }
                               for (uint32_t completeA = 0; (completeA == 0); (completeA = completeA)) {
                                   completeA = cuda::ptx::mbarrier_try_wait_parity(reinterpret_cast<uint64_t*>(&barrierA), 0);
                               }
                               for (uint32_t completeB = 0; (completeB == 0); (completeB = completeB)) {
                                   completeB = cuda::ptx::mbarrier_try_wait_parity(reinterpret_cast<uint64_t*>(&barrierB), 0);
                               }
                               float d[32] = {};
                               asm volatile("wgmma.fence.sync.aligned;"
                                   : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
                                   :
                                   : "memory"
                               );
                               asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1, 0, 0; }"
                                   : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
                                   : "l"(descriptorA), "l"(descriptorB), "r"(1)
                               );
                               asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                               asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                               uint32_t warpId = ((threadIdx.x + (blockDim.x * (threadIdx.y + (blockDim.y * threadIdx.z)))) / 32);
                               uint32_t rowStride = 16;
                               uint32_t packedTop = (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(d[0]))) | (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(d[1]))) << 16));
                               uint32_t packedBottom = (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(d[2]))) | (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(d[3]))) << 16));
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
