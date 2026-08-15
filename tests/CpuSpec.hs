{-# LANGUAGE OverloadedLabels #-}

module CpuSpec (spec) where

import qualified Aoewif.Cpu         as Cpu
import qualified Aoewif.Cpu.Codegen as Codegen
import           Test.Hspec

spec :: Spec
spec = describe "CPU eDSL and code generation" $ do
    it "generates ordered OpenMP C for a dynamic extent" $ do
        Right cpuProgram <- pure $ Cpu.program #ordered_parallel $ do
            size <- Cpu.dynamicExtent #size
            source <- Cpu.input #source [size]
            result <- Cpu.output #result [size]
            Cpu.parallel #element size $ \element -> do
                sourceValue <- Cpu.load source [element]
                Cpu.store result [element] (Cpu.f32 1)
                storedValue <- Cpu.load result [element]
                Cpu.store result [element] (Cpu.add sourceValue storedValue)
        Codegen.cSourceText (Codegen.generateC cpuProgram)
            `shouldBe` unlines
                [ "#include <math.h>"
                , "#include <stddef.h>"
                , ""
                , "void ordered_parallel(const float* source, float* result, size_t size) {"
                , "    #pragma omp parallel for"
                , "    for (size_t element = 0; element < size; ++element) {"
                , "        float value0 = source[element];"
                , "        result[element] = 1.0f;"
                , "        float value1 = result[element];"
                , "        result[element] = (value0 + value1);"
                , "    }"
                , "}"
                ]

    it "generates nested 2D local arithmetic" $ do
        Right cpuProgram <- pure $ Cpu.program #local_math $ do
            let rows = Cpu.staticExtent 2
                columns = Cpu.staticExtent 3
            source <- Cpu.input #source [rows, columns]
            result <- Cpu.output #result [rows, columns]
            Cpu.local #tile [2, 3] $ \tile ->
                Cpu.serial #row rows $ \row ->
                    Cpu.serial #column columns $ \column -> do
                        value <- Cpu.load source [row, column]
                        let transformed =
                                Cpu.divide
                                    ( Cpu.mul
                                        ( Cpu.sub
                                            (Cpu.add value (Cpu.f32 1))
                                            (Cpu.f32 2)
                                        )
                                        (Cpu.f32 3)
                                    )
                                    (Cpu.f32 4)
                        Cpu.store tile [row, column] transformed
                        stagedValue <- Cpu.load tile [row, column]
                        Cpu.store result [row, column] stagedValue
        Codegen.cSourceText (Codegen.generateC cpuProgram)
            `shouldBe` unlines
                [ "#include <math.h>"
                , "#include <stddef.h>"
                , ""
                , "void local_math(const float* source, float* result) {"
                , "    {"
                , "        float tile[(2ull) * (3ull)];"
                , "        for (size_t row = 0; row < 2ull; ++row) {"
                , "            for (size_t column = 0; column < 3ull; ++column) {"
                , "                float value0 = source[((row) * (3ull) + (column))];"
                , "                tile[((row) * (3ull) + (column))] = ((((value0 + 1.0f) - 2.0f) * 3.0f) / 4.0f);"
                , "                float value1 = tile[((row) * (3ull) + (column))];"
                , "                result[((row) * (3ull) + (column))] = value1;"
                , "            }"
                , "        }"
                , "    }"
                , "}"
                ]
