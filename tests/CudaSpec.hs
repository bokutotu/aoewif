module CudaSpec (spec) where

import qualified Aoewif.Cuda         as Cuda
import qualified Aoewif.Cuda.Codegen as Codegen
import qualified Aoewif.Cuda.IR      as IR
import           Test.Hspec

spec :: Spec
spec = describe "CUDA eDSL and code generation" $ do
    it "builds the complete IR for a guarded dynamic vector copy" $ do
        Right actual <- pure $ Cuda.kernel "vector_copy" $ do
            size <- Cuda.dynamicExtent "size"
            source <- Cuda.input "source" [size]
            result <- Cuda.output "result" [size]
            grid <- Cuda.ceilDiv size 256
            Cuda.launch1D grid 256 $ \blockIndex threadIndex -> do
                let globalIndex =
                        Cuda.addIndex
                            (Cuda.multiplyIndex blockIndex (Cuda.indexLiteral 256))
                            threadIndex
                Cuda.when (Cuda.lessThan globalIndex (Cuda.extentIndex size)) $ do
                    value <- Cuda.load source [globalIndex]
                    Cuda.store result [globalIndex] value
        let globalIndex =
                IR.AddIndex
                    (IR.MulIndex IR.BlockIndexX (IR.ConstantIndex 256))
                    IR.ThreadIndexX
            size = IR.DynamicExtent (IR.SymbolId 0)
            expected =
                IR.Kernel
                    { IR.kernelName = "vector_copy"
                    , IR.kernelSymbols = [IR.Symbol (IR.SymbolId 0) "size"]
                    , IR.kernelBuffers =
                        [ IR.BufferDecl
                            (IR.BufferId 0)
                            "source"
                            IR.ReadOnly
                            [size]
                        , IR.BufferDecl
                            (IR.BufferId 1)
                            "result"
                            IR.ReadWrite
                            [size]
                        ]
                    , IR.kernelLaunch =
                        IR.Launch
                            (IR.CeilDivExtent size 256)
                            256
                    , IR.kernelBody =
                        [ IR.IfThen
                            (IR.IndexLessThan globalIndex (IR.ExtentIndex size))
                            [ IR.LetF32
                                (IR.ValueId 0)
                                (IR.LoadF32 (IR.BufferId 0) globalIndex)
                            , IR.StoreF32
                                (IR.BufferId 1)
                                globalIndex
                                (IR.F32Value (IR.ValueId 0))
                            ]
                        ]
                    }
        actual `shouldBe` expected

    it "generates a guarded CUDA kernel with its launch metadata" $ do
        Right cudaKernel <- pure $ Cuda.kernel "vector_copy" $ do
            size <- Cuda.dynamicExtent "size"
            source <- Cuda.input "source" [size]
            result <- Cuda.output "result" [size]
            grid <- Cuda.ceilDiv size 256
            Cuda.launch1D grid 256 $ \blockIndex threadIndex -> do
                let globalIndex =
                        Cuda.addIndex
                            (Cuda.multiplyIndex blockIndex (Cuda.indexLiteral 256))
                            threadIndex
                Cuda.when (Cuda.lessThan globalIndex (Cuda.extentIndex size)) $ do
                    value <- Cuda.load source [globalIndex]
                    Cuda.store result [globalIndex] value
        let generated = Codegen.generateCuda cudaKernel
        ( Codegen.cudaSourceText generated
            , Codegen.cudaKernelName generated
            , Codegen.cudaLaunch generated
            , Codegen.cudaSymbols generated
            , Codegen.cudaBuffers generated
            )
            `shouldBe` ( unlines
                            [ "#include <cuda_runtime.h>"
                            , "#include <math.h>"
                            , "#include <stddef.h>"
                            , ""
                            , "extern \"C\" __global__ void vector_copy(const float* source, float* result, size_t size) {"
                            , "    if (((((size_t)blockIdx.x) * 256ull) + ((size_t)threadIdx.x)) < size) {"
                            , "        float value0 = source[((((size_t)blockIdx.x) * 256ull) + ((size_t)threadIdx.x))];"
                            , "        result[((((size_t)blockIdx.x) * 256ull) + ((size_t)threadIdx.x))] = value0;"
                            , "    }"
                            , "}"
                            ]
                       , "vector_copy"
                       , IR.Launch
                            (IR.CeilDivExtent (IR.DynamicExtent (IR.SymbolId 0)) 256)
                            256
                       , [IR.Symbol (IR.SymbolId 0) "size"]
                       ,
                           [ IR.BufferDecl
                                (IR.BufferId 0)
                                "source"
                                IR.ReadOnly
                                [IR.DynamicExtent (IR.SymbolId 0)]
                           , IR.BufferDecl
                                (IR.BufferId 1)
                                "result"
                                IR.ReadWrite
                                [IR.DynamicExtent (IR.SymbolId 0)]
                           ]
                       )

    it "builds and generates a static shared-memory kernel" $ do
        Right actual <- pure $ Cuda.kernel "shared_stage" $ do
            let size = Cuda.staticExtent 256
            source <- Cuda.input "source" [size]
            result <- Cuda.output "result" [size]
            Cuda.launch1D (Cuda.staticExtent 1) 256 $ \_ threadIndex ->
                Cuda.shared "tile" [256] $ \tile -> do
                    value <- Cuda.load source [threadIndex]
                    Cuda.store tile [threadIndex] value
                    Cuda.syncThreads
                    stagedValue <- Cuda.load tile [threadIndex]
                    Cuda.store result [threadIndex] stagedValue
        let expectedIR =
                IR.Kernel
                    { IR.kernelName = "shared_stage"
                    , IR.kernelSymbols = []
                    , IR.kernelBuffers =
                        [ IR.BufferDecl
                            (IR.BufferId 0)
                            "source"
                            IR.ReadOnly
                            [IR.StaticExtent 256]
                        , IR.BufferDecl
                            (IR.BufferId 1)
                            "result"
                            IR.ReadWrite
                            [IR.StaticExtent 256]
                        ]
                    , IR.kernelLaunch = IR.Launch (IR.StaticExtent 1) 256
                    , IR.kernelBody =
                        [ IR.AllocateShared
                            (IR.SharedDecl (IR.BufferId 2) "tile" [256])
                            [ IR.LetF32
                                (IR.ValueId 0)
                                (IR.LoadF32 (IR.BufferId 0) IR.ThreadIndexX)
                            , IR.StoreF32
                                (IR.BufferId 2)
                                IR.ThreadIndexX
                                (IR.F32Value (IR.ValueId 0))
                            , IR.SyncThreads
                            , IR.LetF32
                                (IR.ValueId 1)
                                (IR.LoadF32 (IR.BufferId 2) IR.ThreadIndexX)
                            , IR.StoreF32
                                (IR.BufferId 1)
                                IR.ThreadIndexX
                                (IR.F32Value (IR.ValueId 1))
                            ]
                        ]
                    }
            expectedSource =
                unlines
                    [ "#include <cuda_runtime.h>"
                    , "#include <math.h>"
                    , "#include <stddef.h>"
                    , ""
                    , "extern \"C\" __global__ void shared_stage(const float* source, float* result) {"
                    , "    {"
                    , "        __shared__ float tile[256ull];"
                    , "        float value0 = source[((size_t)threadIdx.x)];"
                    , "        tile[((size_t)threadIdx.x)] = value0;"
                    , "        __syncthreads();"
                    , "        float value1 = tile[((size_t)threadIdx.x)];"
                    , "        result[((size_t)threadIdx.x)] = value1;"
                    , "    }"
                    , "}"
                    ]
        (actual, Codegen.cudaSourceText (Codegen.generateCuda actual))
            `shouldBe` (expectedIR, expectedSource)

    it "renders overflow-safe ceil division in a serial loop" $ do
        Right cudaKernel <- pure $ Cuda.kernel "ceil_loop" $ do
            size <- Cuda.dynamicExtent "size"
            tiles <- Cuda.ceilDiv size 256
            Cuda.launch1D (Cuda.staticExtent 1) 1 $ \_ _ ->
                Cuda.serial tiles (const (pure ()))
        let generated = Codegen.generateCuda cudaKernel
        ( Codegen.cudaSourceText generated
            , Codegen.cudaKernelName generated
            , Codegen.cudaLaunch generated
            , Codegen.cudaSymbols generated
            , Codegen.cudaBuffers generated
            )
            `shouldBe` ( unlines
                            [ "#include <cuda_runtime.h>"
                            , "#include <math.h>"
                            , "#include <stddef.h>"
                            , ""
                            , "extern \"C\" __global__ void ceil_loop(size_t size) {"
                            , "    for (size_t loop0 = 0; loop0 < (size / 256ull + (size % 256ull != 0ull)); ++loop0) {"
                            , "    }"
                            , "}"
                            ]
                       , "ceil_loop"
                       , IR.Launch (IR.StaticExtent 1) 1
                       , [IR.Symbol (IR.SymbolId 0) "size"]
                       , []
                       )

    it "rejects a barrier inside a conditional launch path" $ do
        Cuda.kernel
            "conditional_sync"
            ( Cuda.launch1D (Cuda.staticExtent 1) 1 $ \_ threadIndex ->
                Cuda.when
                    (Cuda.lessThan threadIndex (Cuda.indexLiteral 1))
                    Cuda.syncThreads
            )
            `shouldBe` Left Cuda.SyncThreadsInsideConditional
