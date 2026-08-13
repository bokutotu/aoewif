{-# LANGUAGE OverloadedLabels #-}
{-# LANGUAGE TypeApplications #-}

module IntegrationSpec (spec) where

import           Aoewif
import           Prelude    hiding (compare, exp, log, maximum, minimum)
import           Test.Hspec

data GemmAxes scope
    = GemmAxes
        (Axis scope Spatial)
        (Axis scope Spatial)

data MatrixAxes scope = MatrixAxes
    { matrixRowAxis    :: Axis scope Spatial
    , matrixColumnAxis :: Axis scope Spatial
    }

newtype VectorAxis scope = VectorAxis (Axis scope Spatial)

spec :: Spec
spec = describe "compute to source integration" $ do
    it "generates CPU source for strict GEMM" $ do
        let built = program #gemm $ do
                rows <- dim #rows
                columns <- dim #columns
                reductionSize <- dim #reduction
                left <- input @F32 #left (rows, reductionSize)
                right <- input @F32 #right (reductionSize, columns)
                output <- compute #output (rows, columns) $ \(row, column) ->
                    ( GemmAxes row column
                    , foldOver reductionSize 0 $ \inner accumulator ->
                        fma (left ! (row, inner)) (right ! (inner, column)) accumulator
                    )
                entry output
        Right gemmProgram <- pure built
        Right gemmSchedule <- pure $ cpu gemmProgram (\_ -> pure ())
        Right source <- pure $ generateC gemmSchedule
        (cSourceText source, cFunctionName source)
            `shouldBe` ( unlines
                            [ "#include <math.h>"
                            , "#include <stdbool.h>"
                            , "#include <stddef.h>"
                            , ""
                            , "#pragma STDC FP_CONTRACT OFF"
                            , ""
                            , "void gemm(const float* input0, const float* input1, float* output, size_t symbol0, size_t symbol1, size_t symbol2) {"
                            , "    for (size_t loop0 = 0; loop0 < symbol0; ++loop0) {"
                            , "        for (size_t loop1 = 0; loop1 < symbol1; ++loop1) {"
                            , "            float accumulator = 0.0f;"
                            , "            for (size_t loop2 = 0; loop2 < symbol2; ++loop2) {"
                            , "                float scalar1 = input0[((loop0) * symbol2 + loop2)];"
                            , "                float scalar2 = input1[((loop2) * symbol1 + loop1)];"
                            , "                float scalar3 = fmaf(scalar1, scalar2, accumulator);"
                            , "                accumulator = scalar3;"
                            , "            }"
                            , "            output[((loop0) * symbol1 + loop1)] = accumulator;"
                            , "        }"
                            , "    }"
                            , "}"
                            ]
                       , "gemm"
                       )

    it "preserves logical indices and guards the tail of a split CPU loop" $ do
        let built = program #choose_by_column $ do
                let rows = staticDim 2
                columns <- dim #columns
                left <- input @F32 #left (rows, columns)
                right <- input @F32 #right (rows, columns)
                output <- compute #output (rows, columns) $ \(row, column) ->
                    ( MatrixAxes row column
                    , select
                        (compare GreaterEqual (index column) (indexLiteral 4))
                        (left ! (row, column))
                        (right ! (row, column))
                    )
                entry output
        Right chooseProgram <- pure built
        Right chooseSchedule <- pure $ cpu chooseProgram $ \MatrixAxes{matrixColumnAxis} -> do
            column <- loop matrixColumnAxis
            _ <- split column 4
            pure ()
        Right source <- pure $ generateC chooseSchedule
        (cSourceText source, cFunctionName source)
            `shouldBe` ( unlines
                            [ "#include <math.h>"
                            , "#include <stdbool.h>"
                            , "#include <stddef.h>"
                            , ""
                            , "#pragma STDC FP_CONTRACT OFF"
                            , ""
                            , "void choose_by_column(const float* input0, const float* input1, float* output, size_t symbol0) {"
                            , "    for (size_t loop0 = 0; loop0 < 2; ++loop0) {"
                            , "        for (size_t loop2 = 0; loop2 < ((symbol0) + 3u) / 4u; ++loop2) {"
                            , "            for (size_t loop3 = 0; loop3 < 4; ++loop3) {"
                            , "                if (((loop2 * 4u) + loop3) < symbol0) {"
                            , "                    float scalar3 = input0[((loop0) * symbol0 + ((loop2 * 4u) + loop3))];"
                            , "                    float scalar4 = input1[((loop0) * symbol0 + ((loop2 * 4u) + loop3))];"
                            , "                    size_t scalar0 = ((loop2 * 4u) + loop3);"
                            , "                    size_t scalar1 = 4u;"
                            , "                    bool scalar2 = (scalar0 >= scalar1);"
                            , "                    float scalar5 = (scalar2 ? scalar3 : scalar4);"
                            , "                    output[((loop0) * symbol0 + ((loop2 * 4u) + loop3))] = scalar5;"
                            , "                }"
                            , "            }"
                            , "        }"
                            , "    }"
                            , "}"
                            ]
                       , "choose_by_column"
                       )

    it "emits typed scalar literals, arithmetic, intrinsics, and selection" $ do
        let built = program #scalar_ops $ do
                let size = staticDim 1
                output <- compute #output size $ \element ->
                    ( VectorAxis element
                    , select
                        (compare LessThan (index element) (indexLiteral 1))
                        (minimum (exp (f32 1.25 .+. f32 2.0)) (f32 3.0))
                        (maximum (log (f32 4.0 .-. f32 5.0)) ((f32 6.0 .*. f32 7.0) ./. f32 8.0))
                    )
                entry output
        Right scalarProgram <- pure built
        Right scalarSchedule <- pure $ cpu scalarProgram (\_ -> pure ())
        Right source <- pure $ generateC scalarSchedule
        (cSourceText source, cFunctionName source)
            `shouldBe` ( unlines
                            [ "#include <math.h>"
                            , "#include <stdbool.h>"
                            , "#include <stddef.h>"
                            , ""
                            , "#pragma STDC FP_CONTRACT OFF"
                            , ""
                            , "void scalar_ops(float* output) {"
                            , "    for (size_t loop0 = 0; loop0 < 1; ++loop0) {"
                            , "        size_t scalar0 = loop0;"
                            , "        size_t scalar1 = 1u;"
                            , "        bool scalar2 = (scalar0 < scalar1);"
                            , "        float scalar3 = 1.25f;"
                            , "        float scalar4 = 2.0f;"
                            , "        float scalar5 = (scalar3 + scalar4);"
                            , "        float scalar6 = expf(scalar5);"
                            , "        float scalar7 = 3.0f;"
                            , "        float scalar8 = fminf(scalar6, scalar7);"
                            , "        float scalar9 = 4.0f;"
                            , "        float scalar10 = 5.0f;"
                            , "        float scalar11 = (scalar9 - scalar10);"
                            , "        float scalar12 = logf(scalar11);"
                            , "        float scalar13 = 6.0f;"
                            , "        float scalar14 = 7.0f;"
                            , "        float scalar15 = (scalar13 * scalar14);"
                            , "        float scalar16 = 8.0f;"
                            , "        float scalar17 = (scalar15 / scalar16);"
                            , "        float scalar18 = fmaxf(scalar12, scalar17);"
                            , "        float scalar19 = (scalar2 ? scalar8 : scalar18);"
                            , "        output[loop0] = scalar19;"
                            , "    }"
                            , "}"
                            ]
                       , "scalar_ops"
                       )

    it "generates CUDA source from a scoped schedule" $ do
        let built = program #cuda_add $ do
                let rows = staticDim 5
                    columns = staticDim 10
                left <- input @F32 #left (rows, columns)
                right <- input @F32 #right (rows, columns)
                output <- compute #output (rows, columns) $ \(row, column) ->
                    (MatrixAxes row column, left ! (row, column) .+. right ! (row, column))
                entry output
        Right addProgram <- pure built
        Right addSchedule <- pure $ cuda defaultCudaTarget addProgram $ \MatrixAxes{matrixRowAxis, matrixColumnAxis} -> do
            row <- loop matrixRowAxis
            column <- loop matrixColumnAxis
            (blockY, threadY) <- split row 2
            (blockX, threadX) <- split column 4
            reorder [blockY, blockX, threadY, threadX]
            bind blockY BlockY
            bind blockX BlockX
            bind threadY ThreadY
            bind threadX ThreadX
        Right source <- pure $ generateCuda addSchedule
        (cudaSourceText source, cudaKernelName source)
            `shouldBe` ( unlines
                            [ "#include <cuda_runtime.h>"
                            , "#include <math.h>"
                            , "#include <stdbool.h>"
                            , "#include <stddef.h>"
                            , ""
                            , "__global__ void cuda_add(const float* input0, const float* input1, float* output) {"
                            , "    if (((((size_t)blockIdx.x) * 4u) + ((size_t)threadIdx.x)) < 10 && ((((size_t)blockIdx.y) * 2u) + ((size_t)threadIdx.y)) < 5) {"
                            , "        float scalar0 = input0[((((((size_t)blockIdx.y) * 2u) + ((size_t)threadIdx.y))) * 10 + ((((size_t)blockIdx.x) * 4u) + ((size_t)threadIdx.x)))];"
                            , "        float scalar1 = input1[((((((size_t)blockIdx.y) * 2u) + ((size_t)threadIdx.y))) * 10 + ((((size_t)blockIdx.x) * 4u) + ((size_t)threadIdx.x)))];"
                            , "        float scalar2 = __fadd_rn(scalar0, scalar1);"
                            , "        output[((((((size_t)blockIdx.y) * 2u) + ((size_t)threadIdx.y))) * 10 + ((((size_t)blockIdx.x) * 4u) + ((size_t)threadIdx.x)))] = scalar2;"
                            , "    }"
                            , "}"
                            ]
                       , "cuda_add"
                       )
