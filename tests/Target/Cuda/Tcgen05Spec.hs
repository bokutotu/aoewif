module Target.Cuda.Tcgen05Spec (spec) where

import qualified Aoewif.Target.Cuda.Codegen     as Codegen
import           Aoewif.Target.Cuda.DSL
import           Aoewif.Target.Cuda.Sm90
import qualified Aoewif.Target.Cuda.Syntax      as Syntax
import           Aoewif.Target.Cuda.Tcgen05.DSL
import           Test.Hspec                     (Spec, describe, it, shouldBe)

spec :: Spec
spec =
    describe "Tcgen05 instructions" $
        it "renders the currently supported parts of hoge.cu" $ do
            let generated =
                    Codegen.generateWith
                        (Codegen.Config [Codegen.CudaBf16Header])
                        ( kernel "gemm_bf16_128x32x16" $ do
                            a <- parameter (Pointer (Const BF16)) "A"
                            b <- parameter (Pointer (Const BF16)) "B"
                            d <- parameter (Pointer F32) "D"
                            body $ do
                                sA <- shared Align16 BF16 "sA" (int 2048)
                                sB <- shared Align16 BF16 "sB" (int 512)
                                done <- shared Align16 USize "done" (int 1)
                                tmem <- shared NaturalAlignment U32 "tmem" (int 1)
                                tid <- define U32 "tid" threadIdxX
                                if_ (tid .< int 32) $
                                    tcgen05Alloc tmem (int 32)
                                if_ (tid .== int 0) $
                                    mBarrierInit (MBarrier (done ! int 0)) (int 1)
                                let i = var "i"
                                for_
                                    (Just (Syntax.VarDecl U32 (Syntax.Name "i") (Just tid)))
                                    (i .< int 2048)
                                    (Just (Syntax.Binary Syntax.Assign i (i .+ int 128)))
                                    $ do
                                        m <- define U32 "m" (i ./ int 16)
                                        k <- define U32 "k" (i .% int 16)
                                        dst <-
                                            define U32 "dst" $
                                                (m ./ int 8)
                                                    .* int 128
                                                    .+ (k ./ int 8)
                                                    .* int 64
                                                    .+ (m .% int 8)
                                                    .* int 8
                                                    .+ k
                                                    .% int 8
                                        sA ! dst .= a ! i
                                for_
                                    (Just (Syntax.VarDecl U32 (Syntax.Name "i") (Just tid)))
                                    (i .< int 512)
                                    (Just (Syntax.Binary Syntax.Assign i (i .+ int 128)))
                                    $ do
                                        k <- define U32 "k" (i ./ int 32)
                                        n <- define U32 "n" (i .% int 32)
                                        dst <-
                                            define U32 "dst" $
                                                (n ./ int 8)
                                                    .* int 128
                                                    .+ (k ./ int 8)
                                                    .* int 64
                                                    .+ (n .% int 8)
                                                    .* int 8
                                                    .+ k
                                                    .% int 8
                                        sB ! dst .= b ! i
                                fenceProxyAsync SharedCta
                                syncThreads
                                tcgen05FenceAfterThreadSync
                                base <- define U32 "base" (tmem ! int 0)
                                if_ (tid .== int 0) $ do
                                    addrA <- define U32 "addrA" (cast U32 (call (var "__cvta_generic_to_shared") [sA]))
                                    addrB <- define U32 "addrB" (cast U32 (call (var "__cvta_generic_to_shared") [sB]))
                                    descBits <-
                                        define USize "descBits" $
                                            shiftL (cast USize (int 8)) (int 16)
                                                .|. shiftL (cast USize (int 16)) (int 32)
                                                .|. shiftL (cast USize (int 1)) (int 46)
                                    descA <- define USize "descA" (cast USize (shiftR (addrA .&. int 0x3ffff) (int 4)) .|. descBits)
                                    descB <- define USize "descB" (cast USize (shiftR (addrB .&. int 0x3ffff) (int 4)) .|. descBits)
                                    idesc <-
                                        define U32 "idesc" $
                                            shiftL (int 8) (int 24)
                                                .|. shiftL (int 4) (int 17)
                                                .|. shiftL (int 1) (int 10)
                                                .|. shiftL (int 1) (int 7)
                                                .|. shiftL (int 1) (int 4)
                                    disableOutputLane <- zeroArray U32 "disableOutputLane" [int 4]
                                    -- TODO: tcgen05_commit is not modeled yet.
                                    tcgen05Mma base descA descB idesc disableOutputLane (bool False)
                                ready <- define Bool "ready" (int 0)
                                for_ Nothing (not_ ready) Nothing $
                                    mBarrierTryWaitParity ready (MBarrier (done ! int 0)) (int 0) Nothing
                                tcgen05FenceAfterThreadSync
                                _ <- define U32 "warpBase" (base .+ shiftL ((tid ./ int 32) .* int 32) (int 16))
                                value <- zeroArray U32 "value" [int 1]
                                let n = var "n"
                                for_
                                    (Just (Syntax.VarDecl U32 (Syntax.Name "n") (Just (int 0))))
                                    (n .< int 32)
                                    (Just (Syntax.Binary Syntax.Assign n (n .+ int 1)))
                                    $ do
                                        -- TODO: tcgen05_ld_32x32b and tcgen05_wait_ld are not modeled yet.
                                        d ! (tid .* int 32 .+ n) .= call (var "__uint_as_float") [value ! int 0]
                                syncThreads
                                if_ (tid .< int 32) $ do
                                    -- TODO: tcgen05_dealloc is not modeled yet.
                                    pure ()
                        )
                expected =
                    unlines
                        [ "#include <stdint.h>"
                        , "#include <cuda/ptx>"
                        , "#include <cuda_bf16.h>"
                        , ""
                        , "extern \"C\" __global__ void gemm_bf16_128x32x16(__nv_bfloat16 const* A, __nv_bfloat16 const* B, float* D) {"
                        , "    __shared__ __align__(16) __nv_bfloat16 sA[2048];"
                        , "    __shared__ __align__(16) __nv_bfloat16 sB[512];"
                        , "    __shared__ __align__(16) size_t done[1];"
                        , "    __shared__ uint32_t tmem[1];"
                        , "    uint32_t tid = threadIdx.x;"
                        , "    if ((tid < 32)) {"
                        , "        cuda::ptx::tcgen05_alloc(cuda::ptx::cta_group_1, tmem, 32);"
                        , "    }"
                        , "    if ((tid == 0)) {"
                        , "        cuda::ptx::mbarrier_init(reinterpret_cast<uint64_t*>(&done[0]), 1);"
                        , "    }"
                        , "    for (uint32_t i = tid; (i < 2048); (i = (i + 128))) {"
                        , "        uint32_t m = (i / 16);"
                        , "        uint32_t k = (i % 16);"
                        , "        uint32_t dst = (((((m / 8) * 128) + ((k / 8) * 64)) + ((m % 8) * 8)) + (k % 8));"
                        , "        (sA[dst] = A[i]);"
                        , "    }"
                        , "    for (uint32_t i = tid; (i < 512); (i = (i + 128))) {"
                        , "        uint32_t k = (i / 32);"
                        , "        uint32_t n = (i % 32);"
                        , "        uint32_t dst = (((((n / 8) * 128) + ((k / 8) * 64)) + ((n % 8) * 8)) + (k % 8));"
                        , "        (sB[dst] = B[i]);"
                        , "    }"
                        , "    cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);"
                        , "    __syncthreads();"
                        , "    cuda::ptx::tcgen05_fence_after_thread_sync();"
                        , "    uint32_t base = tmem[0];"
                        , "    if ((tid == 0)) {"
                        , "        uint32_t addrA = static_cast<uint32_t>(__cvta_generic_to_shared(sA));"
                        , "        uint32_t addrB = static_cast<uint32_t>(__cvta_generic_to_shared(sB));"
                        , "        size_t descBits = (((static_cast<size_t>(8) << 16) | (static_cast<size_t>(16) << 32)) | (static_cast<size_t>(1) << 46));"
                        , "        size_t descA = (static_cast<size_t>(((addrA & 262143) >> 4)) | descBits);"
                        , "        size_t descB = (static_cast<size_t>(((addrB & 262143) >> 4)) | descBits);"
                        , "        uint32_t idesc = (((((8 << 24) | (4 << 17)) | (1 << 10)) | (1 << 7)) | (1 << 4));"
                        , "        uint32_t disableOutputLane[4] = {};"
                        , "        cuda::ptx::tcgen05_mma(cuda::ptx::kind_f16, cuda::ptx::cta_group_1, base, descA, descB, idesc, disableOutputLane, false);"
                        , "    }"
                        , "    bool ready = 0;"
                        , "    for (; (!ready); ) {"
                        , "        ready = cuda::ptx::mbarrier_try_wait_parity(reinterpret_cast<uint64_t*>(&done[0]), 0);"
                        , "    }"
                        , "    cuda::ptx::tcgen05_fence_after_thread_sync();"
                        , "    uint32_t warpBase = (base + (((tid / 32) * 32) << 16));"
                        , "    uint32_t value[1] = {};"
                        , "    for (uint32_t n = 0; (n < 32); (n = (n + 1))) {"
                        , "        (D[((tid * 32) + n)] = __uint_as_float(value[0]));"
                        , "    }"
                        , "    __syncthreads();"
                        , "    if ((tid < 32)) {"
                        , "    }"
                        , "}"
                        ]
            generated `shouldBe` expected
