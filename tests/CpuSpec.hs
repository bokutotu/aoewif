{-# LANGUAGE OverloadedLabels #-}

module CpuSpec (spec) where

import qualified Aoewif.Cpu         as Cpu
import qualified Aoewif.Cpu.Codegen as Codegen
import qualified Aoewif.Cpu.IR      as IR
import           Test.Hspec

spec :: Spec
spec = describe "CPU eDSL and code generation" $ do
    it "builds the complete IR for an ordered dynamic parallel program" $ do
        Right actual <- pure $ Cpu.program #ordered_parallel $ do
            size <- Cpu.dynamicExtent #size
            source <- Cpu.input #source [size]
            result <- Cpu.output #result [size]
            Cpu.parallel #element size $ \element -> do
                sourceValue <- Cpu.load source [element]
                Cpu.store result [element] (Cpu.f32 1)
                storedValue <- Cpu.load result [element]
                Cpu.store result [element] (Cpu.add sourceValue storedValue)
        let expected =
                IR.Program
                    { IR.programName = #ordered_parallel
                    , IR.programExtents = [#size]
                    , IR.programBuffers =
                        [ IR.Buffer
                            (IR.BufferId 0)
                            #source
                            IR.ReadOnly
                            [IR.DynamicExtent #size]
                        , IR.Buffer
                            (IR.BufferId 1)
                            #result
                            IR.ReadWrite
                            [IR.DynamicExtent #size]
                        ]
                    , IR.programBody =
                        [ IR.For
                            ( IR.Loop
                                { IR.loopKind = IR.Parallel
                                , IR.loopIndex = IR.IndexId 0
                                , IR.loopName = #element
                                , IR.loopExtent = IR.DynamicExtent #size
                                , IR.loopBody =
                                    [ IR.Let
                                        (IR.ValueId 0)
                                        (IR.BufferId 0)
                                        [IR.IndexId 0]
                                    , IR.Store
                                        (IR.BufferId 1)
                                        [IR.IndexId 0]
                                        (IR.F32Literal 1)
                                    , IR.Let
                                        (IR.ValueId 1)
                                        (IR.BufferId 1)
                                        [IR.IndexId 0]
                                    , IR.Store
                                        (IR.BufferId 1)
                                        [IR.IndexId 0]
                                        ( IR.AddExpr
                                            (IR.ValueExpr (IR.ValueId 0))
                                            (IR.ValueExpr (IR.ValueId 1))
                                        )
                                    ]
                                }
                            )
                        ]
                    }
        actual `shouldBe` expected

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
        let generated = Codegen.generateC cpuProgram
        (Codegen.cSourceText generated, Codegen.cFunctionName generated)
            `shouldBe` ( unlines
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
                       , "ordered_parallel"
                       )

    it "builds and generates nested 2D local arithmetic" $ do
        Right actual <- pure $ Cpu.program #local_math $ do
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
        let expectedIR =
                IR.Program
                    { IR.programName = #local_math
                    , IR.programExtents = []
                    , IR.programBuffers =
                        [ IR.Buffer
                            (IR.BufferId 0)
                            #source
                            IR.ReadOnly
                            [IR.StaticExtent 2, IR.StaticExtent 3]
                        , IR.Buffer
                            (IR.BufferId 1)
                            #result
                            IR.ReadWrite
                            [IR.StaticExtent 2, IR.StaticExtent 3]
                        ]
                    , IR.programBody =
                        [ IR.Allocate
                            ( IR.Buffer
                                (IR.BufferId 2)
                                #tile
                                IR.ReadWrite
                                [IR.StaticExtent 2, IR.StaticExtent 3]
                            )
                            [ IR.For
                                ( IR.Loop
                                    { IR.loopKind = IR.Serial
                                    , IR.loopIndex = IR.IndexId 0
                                    , IR.loopName = #row
                                    , IR.loopExtent = IR.StaticExtent 2
                                    , IR.loopBody =
                                        [ IR.For
                                            ( IR.Loop
                                                { IR.loopKind = IR.Serial
                                                , IR.loopIndex = IR.IndexId 1
                                                , IR.loopName = #column
                                                , IR.loopExtent = IR.StaticExtent 3
                                                , IR.loopBody =
                                                    [ IR.Let
                                                        (IR.ValueId 0)
                                                        (IR.BufferId 0)
                                                        [IR.IndexId 0, IR.IndexId 1]
                                                    , IR.Store
                                                        (IR.BufferId 2)
                                                        [IR.IndexId 0, IR.IndexId 1]
                                                        ( IR.DivExpr
                                                            ( IR.MulExpr
                                                                ( IR.SubExpr
                                                                    ( IR.AddExpr
                                                                        (IR.ValueExpr (IR.ValueId 0))
                                                                        (IR.F32Literal 1)
                                                                    )
                                                                    (IR.F32Literal 2)
                                                                )
                                                                (IR.F32Literal 3)
                                                            )
                                                            (IR.F32Literal 4)
                                                        )
                                                    , IR.Let
                                                        (IR.ValueId 1)
                                                        (IR.BufferId 2)
                                                        [IR.IndexId 0, IR.IndexId 1]
                                                    , IR.Store
                                                        (IR.BufferId 1)
                                                        [IR.IndexId 0, IR.IndexId 1]
                                                        (IR.ValueExpr (IR.ValueId 1))
                                                    ]
                                                }
                                            )
                                        ]
                                    }
                                )
                            ]
                        ]
                    }
            generated = Codegen.generateC actual
        ( actual
            , (Codegen.cSourceText generated, Codegen.cFunctionName generated)
            )
            `shouldBe` ( expectedIR
                       ,
                           ( unlines
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
                           , "local_math"
                           )
                       )

    it "rejects a buffer access with the wrong rank while building" $ do
        Cpu.program
            #bad_rank
            ( do
                let size = Cpu.staticExtent 4
                source <- Cpu.input #source [size]
                Cpu.parallel #element size $ \element -> do
                    _ <- Cpu.load source [element, element]
                    pure ()
            )
            `shouldBe` Left (Cpu.RankMismatch #source 1 2)

    it "rejects a local buffer whose byte size overflows" $ do
        Cpu.program
            #oversized_local
            (Cpu.local #tile [maxBound] (const (pure ())))
            `shouldBe` Left (Cpu.LocalSizeOverflow #tile)
