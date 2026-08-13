{-# LANGUAGE OverloadedLabels #-}

module IntegrationSpec (spec) where

import           Aoewif.Codegen
import           Aoewif.Compute
import qualified Aoewif.Schedule as Schedule
import           Test.Hspec

spec :: Spec
spec = describe "compute to source integration" $ do
    it "generates C for a pointwise copy with a no-op schedule" $ do
        let computeIR = program #copy $ do
                size <- dim #size
                inputTensor <- input #source f32 [size]
                result <- output #result f32 [size]
                for #i size $ \element ->
                    block #copy $ do
                        value <- load inputTensor [element]
                        store result [element] value
            generated = generateC (Schedule.schedule computeIR (pure ()))
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

    it "loads once at its declared position and reuses the value after a store" $ do
        let computeIR = program #load_order $ do
                let size = staticDim 1
                source <- input #source f32 [size]
                result <- output #result f32 [size]
                for #i size $ \element ->
                    block #ordered $ do
                        original <- load result [element]
                        store result [element] 1
                        store result [element] (original + original)
                        store result [element] =<< load source [element]
            generated = generateC computeIR
        cSourceText generated
            `shouldBe` unlines
                [ "#include <math.h>"
                , "#include <stdbool.h>"
                , "#include <stddef.h>"
                , ""
                , "#pragma STDC FP_CONTRACT OFF"
                , ""
                , "void load_order(const float* input0, float* output0) {"
                , "    for (size_t loop0 = 0; loop0 < 1; ++loop0) {"
                , "        float scalar0 = output0[loop0];"
                , "        float scalar1 = 1.0f;"
                , "        output0[loop0] = scalar1;"
                , "        float scalar2 = (scalar0 + scalar0);"
                , "        output0[loop0] = scalar2;"
                , "        float scalar3 = input0[loop0];"
                , "        output0[loop0] = scalar3;"
                , "    }"
                , "}"
                ]

    it "generates distinct index, boolean, and float scalar temporaries" $ do
        let computeIR = program #scalar_types $ do
                let size = staticDim 4
                result <- output #result f32 [size]
                for #i size $ \element ->
                    block #choose $ do
                        let condition = compare_ Less (index element) (indexLiteral 2)
                        store result [element] (select condition 1 0)
            generated = generateC computeIR
        (cSourceText generated, cFunctionName generated)
            `shouldBe` ( unlines
                            [ "#include <math.h>"
                            , "#include <stdbool.h>"
                            , "#include <stddef.h>"
                            , ""
                            , "#pragma STDC FP_CONTRACT OFF"
                            , ""
                            , "void scalar_types(float* output0) {"
                            , "    for (size_t loop0 = 0; loop0 < 4; ++loop0) {"
                            , "        size_t scalar0 = loop0;"
                            , "        size_t scalar1 = 2u;"
                            , "        bool scalar2 = (scalar0 < scalar1);"
                            , "        float scalar3 = 1.0f;"
                            , "        float scalar4 = 0.0f;"
                            , "        float scalar5 = (scalar2 ? scalar3 : scalar4);"
                            , "        output0[loop0] = scalar5;"
                            , "    }"
                            , "}"
                            ]
                       , "scalar_types"
                       )

    it "lowers predicate and index selections" $ do
        let computeIR = program #split_categories $ do
                let size = staticDim 2
                source <- input #source f32 [size]
                result <- output #result f32 [size]
                for #i size $ \element ->
                    block #choose $ do
                        value <- load source [element]
                        let dataMatches = compare_ Equal value value
                            enabled = select dataMatches (boolean True) (boolean False)
                            chooseElement = compare_ Equal enabled (boolean True)
                            selectedIndex = select chooseElement (index element) (indexLiteral 0)
                            isFirst = compare_ Equal selectedIndex (indexLiteral 0)
                        store result [element] (select isFirst 1 0)
            generated = generateC computeIR
        cSourceText generated
            `shouldBe` unlines
                [ "#include <math.h>"
                , "#include <stdbool.h>"
                , "#include <stddef.h>"
                , ""
                , "#pragma STDC FP_CONTRACT OFF"
                , ""
                , "void split_categories(const float* input0, float* output0) {"
                , "    for (size_t loop0 = 0; loop0 < 2; ++loop0) {"
                , "        float scalar0 = input0[loop0];"
                , "        bool scalar1 = (scalar0 == scalar0);"
                , "        bool scalar2 = true;"
                , "        bool scalar3 = false;"
                , "        bool scalar4 = (scalar1 ? scalar2 : scalar3);"
                , "        bool scalar5 = true;"
                , "        bool scalar6 = (scalar4 == scalar5);"
                , "        size_t scalar7 = loop0;"
                , "        size_t scalar8 = 0u;"
                , "        size_t scalar9 = (scalar6 ? scalar7 : scalar8);"
                , "        size_t scalar10 = 0u;"
                , "        bool scalar11 = (scalar9 == scalar10);"
                , "        float scalar12 = 1.0f;"
                , "        float scalar13 = 0.0f;"
                , "        float scalar14 = (scalar11 ? scalar12 : scalar13);"
                , "        output0[loop0] = scalar14;"
                , "    }"
                , "}"
                ]

    it "guards the tail of a dynamically sized split loop in C" $ do
        let computeIR = program #dynamic_copy $ do
                size <- dim #size
                inputTensor <- input #source f32 [size]
                result <- output #result f32 [size]
                for #i size $ \element ->
                    block #copy $ do
                        value <- load inputTensor [element]
                        store result [element] value
            scheduled = Schedule.schedule computeIR $ do
                copyBlock <- Schedule.block #copy
                element <- Schedule.loopOf copyBlock #i
                _ <- Schedule.split element 4
                pure ()
            generated = generateC scheduled
        cSourceText generated
            `shouldBe` unlines
                [ "#include <math.h>"
                , "#include <stdbool.h>"
                , "#include <stddef.h>"
                , ""
                , "#pragma STDC FP_CONTRACT OFF"
                , ""
                , "void dynamic_copy(const float* input0, float* output0, size_t symbol0) {"
                , "    for (size_t loop1 = 0; loop1 < ((symbol0) + 3u) / 4u; ++loop1) {"
                , "        for (size_t loop2 = 0; loop2 < 4; ++loop2) {"
                , "            if (((loop1 * 4u) + loop2) < symbol0) {"
                , "                float scalar0 = input0[((loop1 * 4u) + loop2)];"
                , "                output0[((loop1 * 4u) + loop2)] = scalar0;"
                , "            }"
                , "        }"
                , "    }"
                , "}"
                ]

    it "keeps explicit matmul initialization before its update loops in C" $ do
        let rows = staticDim 2
            columns = staticDim 3
            inner = staticDim 4
            computeIR = program #matmul $ do
                left <- input #left f32 [rows, inner]
                right <- input #right f32 [inner, columns]
                result <- output #result f32 [rows, columns]
                for #m rows $ \row ->
                    for #n columns $ \column -> do
                        block #initialize $ store result [row, column] 0
                        for #k inner $ \reductionIndex ->
                            block #update $ do
                                lhs <- load left [row, reductionIndex]
                                rhs <- load right [reductionIndex, column]
                                update add result [row, column] (lhs * rhs)
            scheduled = Schedule.schedule computeIR $ do
                updateBlock <- Schedule.block #update
                row <- Schedule.loopOf updateBlock #m
                column <- Schedule.loopOf updateBlock #n
                reductionIndex <- Schedule.loopOf updateBlock #k
                (rowOuter, rowInner) <- Schedule.split row 1
                (columnOuter, columnInner) <- Schedule.split column 2
                _ <- Schedule.split reductionIndex 2
                Schedule.reorder [rowOuter, columnOuter, rowInner, columnInner]
            generated = generateC scheduled
        cSourceText generated
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
                , "            for (size_t loop4 = 0; loop4 < 1; ++loop4) {"
                , "                for (size_t loop6 = 0; loop6 < 2; ++loop6) {"
                , "                    if (((loop5 * 2u) + loop6) < 3) {"
                , "                        float scalar0 = 0.0f;"
                , "                        output0[((((loop3 * 1u) + loop4) * 3) + ((loop5 * 2u) + loop6))] = scalar0;"
                , "                    }"
                , "                    for (size_t loop7 = 0; loop7 < ((4) + 1u) / 2u; ++loop7) {"
                , "                        for (size_t loop8 = 0; loop8 < 2; ++loop8) {"
                , "                            if (((loop5 * 2u) + loop6) < 3) {"
                , "                                float scalar1 = input0[((((loop3 * 1u) + loop4) * 4) + ((loop7 * 2u) + loop8))];"
                , "                                float scalar2 = input1[((((loop7 * 2u) + loop8) * 3) + ((loop5 * 2u) + loop6))];"
                , "                                float scalar3 = (scalar1 * scalar2);"
                , "                                float scalar4 = output0[((((loop3 * 1u) + loop4) * 3) + ((loop5 * 2u) + loop6))];"
                , "                                float scalar5 = (scalar4 + scalar3);"
                , "                                output0[((((loop3 * 1u) + loop4) * 3) + ((loop5 * 2u) + loop6))] = scalar5;"
                , "                            }"
                , "                        }"
                , "                    }"
                , "                }"
                , "            }"
                , "        }"
                , "    }"
                , "}"
                ]

    it "generates CUDA arithmetic inside a lexical loop" $ do
        let computeIR = program #cuda_arithmetic $ do
                let size = staticDim 8
                left <- input #left f32 [size]
                right <- input #right f32 [size]
                result <- output #result f32 [size]
                for #i size $ \element ->
                    block #arithmetic $ do
                        lhs <- load left [element]
                        rhs <- load right [element]
                        store result [element] ((lhs + rhs) * 2)
            generated = generateCuda (Schedule.schedule computeIR (pure ()))
        (cudaSourceText generated, cudaKernelName generated)
            `shouldBe` ( unlines
                            [ "#include <cuda_runtime.h>"
                            , "#include <math.h>"
                            , "#include <stdbool.h>"
                            , "#include <stddef.h>"
                            , ""
                            , "__global__ void cuda_arithmetic(const float* input0, const float* input1, float* output0) {"
                            , "    for (size_t loop0 = 0; loop0 < 8; ++loop0) {"
                            , "        float scalar0 = input0[loop0];"
                            , "        float scalar1 = input1[loop0];"
                            , "        float scalar2 = __fadd_rn(scalar0, scalar1);"
                            , "        float scalar3 = 2.0f;"
                            , "        float scalar4 = __fmul_rn(scalar2, scalar3);"
                            , "        output0[loop0] = scalar4;"
                            , "    }"
                            , "}"
                            ]
                       , "cuda_arithmetic"
                       )
