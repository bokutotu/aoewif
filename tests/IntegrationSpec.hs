{-# LANGUAGE OverloadedLabels #-}

module IntegrationSpec (spec) where

import           Aoewif.Codegen
import           Aoewif.Compute
import qualified Aoewif.Schedule as Schedule
import           Test.Hspec

spec :: Spec
spec = describe "compute to source integration" $ do
    it "generates CPU source for a pointwise block" $ do
        let computeIR = program #copy $ do
                size <- dim #size
                inputTensor <- input #source f32 [size]
                result <- output #result f32 [size]
                block #copy $ do
                    element <- spatial #i size
                    define result [element] (inputTensor ! [element])
            generated = generateC (Schedule.cpu computeIR (pure ()))
        (cSourceText generated, cFunctionName generated)
            `shouldBe` ( unlines
                            [ "#include <math.h>"
                            , "#include <stdbool.h>"
                            , "#include <stddef.h>"
                            , ""
                            , "#pragma STDC FP_CONTRACT OFF"
                            , ""
                            , "void copy(const float* input0, float* output0, size_t symbol0) {"
                            , "    for (size_t loop0 = 0; loop0 < symbol0; ++loop0) {"
                            , "        float scalar0 = input0[loop0];"
                            , "        output0[loop0] = scalar0;"
                            , "    }"
                            , "}"
                            ]
                       , "copy"
                       )

    it "uses canonical split bindings and guards a CPU tail" $ do
        let computeIR = program #choose_by_column $ do
                let rows = staticDim 2
                columns <- dim #columns
                left <- input #left f32 [rows, columns]
                right <- input #right f32 [rows, columns]
                result <- output #result f32 [rows, columns]
                block #choose $ do
                    row <- spatial #row rows
                    column <- spatial #column columns
                    define result [row, column] $
                        select
                            (compare_ GreaterEqual (index column) (indexLiteral 4))
                            (left ! [row, column])
                            (right ! [row, column])
            schedule = Schedule.cpu computeIR $ do
                chooseBlock <- Schedule.block #choose
                column <- Schedule.axis chooseBlock #column
                _ <- Schedule.split column 4
                pure ()
            source = generateC schedule
        cSourceText source
            `shouldBe` unlines
                [ "#include <math.h>"
                , "#include <stdbool.h>"
                , "#include <stddef.h>"
                , ""
                , "#pragma STDC FP_CONTRACT OFF"
                , ""
                , "void choose_by_column(const float* input0, const float* input1, float* output0, size_t symbol0) {"
                , "    for (size_t loop0 = 0; loop0 < 2; ++loop0) {"
                , "        for (size_t loop2 = 0; loop2 < ((symbol0) + 3u) / 4u; ++loop2) {"
                , "            for (size_t loop3 = 0; loop3 < 4; ++loop3) {"
                , "                if (((loop2 * 4u) + loop3) < symbol0) {"
                , "                    size_t scalar0 = ((loop2 * 4u) + loop3);"
                , "                    size_t scalar1 = 4u;"
                , "                    bool scalar2 = (scalar0 >= scalar1);"
                , "                    float scalar3 = input0[((loop0) * symbol0 + ((loop2 * 4u) + loop3))];"
                , "                    float scalar4 = input1[((loop0) * symbol0 + ((loop2 * 4u) + loop3))];"
                , "                    float scalar5 = (scalar2 ? scalar3 : scalar4);"
                , "                    output0[((loop0) * symbol0 + ((loop2 * 4u) + loop3))] = scalar5;"
                , "                }"
                , "            }"
                , "        }"
                , "    }"
                , "}"
                ]

    it "lowers an explicitly defined reduction after split and reorder" $ do
        let rows = staticDim 2
            columns = staticDim 3
            inner = staticDim 4
            computeIR = program #matmul $ do
                left <- input #left f32 [rows, inner]
                right <- input #right f32 [inner, columns]
                result <- output #result f32 [rows, columns]
                block #matmul $ do
                    row <- spatial #m rows
                    column <- spatial #n columns
                    reductionAxis <- reduction #k inner
                    define result [row, column] $
                        sumOver [reductionAxis] $
                            left ! [row, reductionAxis] * right ! [reductionAxis, column]
            schedule = Schedule.cpu computeIR $ do
                matmulBlock <- Schedule.block #matmul
                row <- Schedule.axis matmulBlock #m
                column <- Schedule.axis matmulBlock #n
                reductionAxis <- Schedule.axis matmulBlock #k
                (rowOuter, rowInner) <- Schedule.split row 1
                (columnOuter, columnInner) <- Schedule.split column 2
                (reductionOuter, reductionInner) <- Schedule.split reductionAxis 2
                Schedule.reorder matmulBlock [rowOuter, columnOuter, reductionOuter, rowInner, columnInner, reductionInner]
            source = cSourceText (generateC schedule)
        source
            `shouldBe` unlines
                [ "#include <math.h>"
                , "#include <stdbool.h>"
                , "#include <stddef.h>"
                , ""
                , "#pragma STDC FP_CONTRACT OFF"
                , ""
                , "void matmul(const float* input0, const float* input1, float* output0) {"
                , "    for (size_t loop3 = 0; loop3 < ((2) + 0u) / 1u; ++loop3) {"
                , "        for (size_t loop5 = 0; loop5 < ((3) + 1u) / 2u; ++loop5) {"
                , "            for (size_t loop7 = 0; loop7 < ((4) + 1u) / 2u; ++loop7) {"
                , "                for (size_t loop4 = 0; loop4 < 1; ++loop4) {"
                , "                    for (size_t loop6 = 0; loop6 < 2; ++loop6) {"
                , "                        for (size_t loop8 = 0; loop8 < 2; ++loop8) {"
                , "                            if (((loop5 * 2u) + loop6) < 3) {"
                , "                                if (((loop7 * 2u) + loop8) == 0) {"
                , "                                    float scalar0 = 0.0f;"
                , "                                    output0[((((loop3 * 1u) + loop4)) * 3 + ((loop5 * 2u) + loop6))] = scalar0;"
                , "                                }"
                , "                                float scalar1 = input0[((((loop3 * 1u) + loop4)) * 4 + ((loop7 * 2u) + loop8))];"
                , "                                float scalar2 = input1[((((loop7 * 2u) + loop8)) * 3 + ((loop5 * 2u) + loop6))];"
                , "                                float scalar3 = (scalar1 * scalar2);"
                , "                                float scalar4 = output0[((((loop3 * 1u) + loop4)) * 3 + ((loop5 * 2u) + loop6))];"
                , "                                float scalar5 = (scalar4 + scalar3);"
                , "                                output0[((((loop3 * 1u) + loop4)) * 3 + ((loop5 * 2u) + loop6))] = scalar5;"
                , "                            }"
                , "                        }"
                , "                    }"
                , "                }"
                , "            }"
                , "        }"
                , "    }"
                , "}"
                ]

    it "renders CPU execution and unroll metadata" $ do
        let computeIR = program #scaled $ do
                inputTensor <- input #source f32 [staticDim 8]
                result <- output #result f32 [staticDim 8]
                block #scaled $ do
                    element <- spatial #i (staticDim 8)
                    define result [element] (inputTensor ! [element] * 2)
            schedule = Schedule.cpu computeIR $ do
                scaledBlock <- Schedule.block #scaled
                element <- Schedule.axis scaledBlock #i
                Schedule.parallel element
                Schedule.unrollBy 4 element
            generated = cSourceText (generateC schedule)
        generated
            `shouldBe` unlines
                [ "#include <math.h>"
                , "#include <stdbool.h>"
                , "#include <stddef.h>"
                , ""
                , "#pragma STDC FP_CONTRACT OFF"
                , ""
                , "void scaled(const float* input0, float* output0) {"
                , "    #pragma omp parallel for"
                , "    #pragma GCC unroll 4"
                , "    for (size_t loop0 = 0; loop0 < 8; ++loop0) {"
                , "        float scalar0 = input0[loop0];"
                , "        float scalar1 = 2.0f;"
                , "        float scalar2 = (scalar0 * scalar1);"
                , "        output0[loop0] = scalar2;"
                , "    }"
                , "}"
                ]

    it "generates CUDA source from bound split loops" $ do
        let computeIR = program #cuda_add $ do
                let size = staticDim 10
                left <- input #left f32 [size]
                right <- input #right f32 [size]
                result <- output #result f32 [size]
                block #add $ do
                    element <- spatial #i size
                    define result [element] (left ! [element] + right ! [element])
            schedule = Schedule.cuda computeIR $ do
                addBlock <- Schedule.block #add
                element <- Schedule.axis addBlock #i
                (blockX, threadX) <- Schedule.split element 4
                Schedule.bind blockX Schedule.BlockX
                Schedule.bind threadX Schedule.ThreadX
            source = generateCuda schedule
        (cudaSourceText source, cudaKernelName source)
            `shouldBe` ( unlines
                            [ "#include <cuda_runtime.h>"
                            , "#include <math.h>"
                            , "#include <stdbool.h>"
                            , "#include <stddef.h>"
                            , ""
                            , "__global__ void cuda_add(const float* input0, const float* input1, float* output0) {"
                            , "    if (((((size_t)blockIdx.x) * 4u) + ((size_t)threadIdx.x)) < 10) {"
                            , "        float scalar0 = input0[((((size_t)blockIdx.x) * 4u) + ((size_t)threadIdx.x))];"
                            , "        float scalar1 = input1[((((size_t)blockIdx.x) * 4u) + ((size_t)threadIdx.x))];"
                            , "        float scalar2 = __fadd_rn(scalar0, scalar1);"
                            , "        output0[((((size_t)blockIdx.x) * 4u) + ((size_t)threadIdx.x))] = scalar2;"
                            , "    }"
                            , "}"
                            ]
                       , "cuda_add"
                       )

    it "renders CUDA unroll metadata on a lexical loop" $ do
        let computeIR = program #cuda_copy $ do
                inputTensor <- input #source f32 [staticDim 8]
                result <- output #result f32 [staticDim 8]
                block #copy $ do
                    element <- spatial #i (staticDim 8)
                    define result [element] (inputTensor ! [element])
            schedule = Schedule.cuda computeIR $ do
                copyBlock <- Schedule.block #copy
                element <- Schedule.axis copyBlock #i
                Schedule.unrollBy 2 element
            generated = cudaSourceText (generateCuda schedule)
        generated
            `shouldBe` unlines
                [ "#include <cuda_runtime.h>"
                , "#include <math.h>"
                , "#include <stdbool.h>"
                , "#include <stddef.h>"
                , ""
                , "__global__ void cuda_copy(const float* input0, float* output0) {"
                , "    #pragma unroll 2"
                , "    for (size_t loop0 = 0; loop0 < 8; ++loop0) {"
                , "        float scalar0 = input0[loop0];"
                , "        output0[loop0] = scalar0;"
                , "    }"
                , "}"
                ]
