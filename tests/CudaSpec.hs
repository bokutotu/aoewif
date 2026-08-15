{-# LANGUAGE OverloadedLabels #-}

module CudaSpec (spec) where

import qualified Aoewif.Cuda         as Cuda
import qualified Aoewif.Cuda.Codegen as Codegen
import           Test.Hspec

spec :: Spec
spec = describe "CUDA eDSL and code generation" $ do
    it "generates a guarded CUDA kernel" $ do
        Right cudaKernel <- pure $ Cuda.kernel #vector_copy $ do
            size <- Cuda.dynamicExtent #size
            source <- Cuda.input #source [size]
            result <- Cuda.output #result [size]
            grid <- Cuda.ceilDiv size 256
            Cuda.launch1D grid 256 $ \blockIndex threadIndex -> do
                let globalIndex =
                        Cuda.addIndex
                            (Cuda.multiplyIndex blockIndex (Cuda.indexLiteral 256))
                            threadIndex
                Cuda.when (Cuda.lessThan globalIndex (Cuda.extentIndex size)) $ do
                    value <- Cuda.load source [globalIndex]
                    Cuda.store result [globalIndex] value
        Codegen.cudaSourceText (Codegen.generateCuda cudaKernel)
            `shouldBe` unlines
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

    it "generates a static shared-memory kernel" $ do
        Right cudaKernel <- pure $ Cuda.kernel #shared_stage $ do
            let size = Cuda.staticExtent 256
            source <- Cuda.input #source [size]
            result <- Cuda.output #result [size]
            Cuda.launch1D (Cuda.staticExtent 1) 256 $ \_ threadIndex ->
                Cuda.shared #tile [256] $ \tile -> do
                    value <- Cuda.load source [threadIndex]
                    Cuda.store tile [threadIndex] value
                    Cuda.syncThreads
                    stagedValue <- Cuda.load tile [threadIndex]
                    Cuda.store result [threadIndex] stagedValue
        Codegen.cudaSourceText (Codegen.generateCuda cudaKernel)
            `shouldBe` unlines
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

    it "renders overflow-safe ceil division in a serial loop" $ do
        Right cudaKernel <- pure $ Cuda.kernel #ceil_loop $ do
            size <- Cuda.dynamicExtent #size
            tiles <- Cuda.ceilDiv size 256
            Cuda.launch1D (Cuda.staticExtent 1) 1 $ \_ _ ->
                Cuda.serial tiles (const (pure ()))
        Codegen.cudaSourceText (Codegen.generateCuda cudaKernel)
            `shouldBe` unlines
                [ "#include <cuda_runtime.h>"
                , "#include <math.h>"
                , "#include <stddef.h>"
                , ""
                , "extern \"C\" __global__ void ceil_loop(size_t size) {"
                , "    for (size_t loop0 = 0; loop0 < (size / 256ull + (size % 256ull != 0ull)); ++loop0) {"
                , "    }"
                , "}"
                ]
