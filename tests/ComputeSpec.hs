{-# LANGUAGE OverloadedLabels #-}

module ComputeSpec (spec) where

import           Aoewif.Compute
import           Test.Hspec

spec :: Spec
spec = describe "compute eDSL" $ do
    it "elaborates matrix multiplication to a first-order reduction definition" $ do
        let rows = staticDim 4
            columns = staticDim 8
            inner = staticDim 16
            actual = program #matmul $ do
                left <- input #left f32 [rows, inner]
                right <- input #right f32 [inner, columns]
                result <- output #result f32 [rows, columns]
                block #matmul $ do
                    row <- spatial #m rows
                    column <- spatial #n columns
                    reductionAxis <- reduction #k inner
                    define result [row, column] $
                        reduce add 0 [reductionAxis] $
                            left ! [row, reductionAxis] * right ! [reductionAxis, column]
            expected =
                ComputeIR
                    { computeName = #matmul
                    , computeSymbols = []
                    , computeTensors =
                        [ TensorDecl (TensorId 0) #left F32Type [rows, inner] (InputTensor 0)
                        , TensorDecl (TensorId 1) #right F32Type [inner, columns] (InputTensor 1)
                        , TensorDecl (TensorId 2) #result F32Type [rows, columns] (OutputTensor 0)
                        ]
                    , computeBlocks =
                        [ ComputeBlock
                            { blockId = BlockId 0
                            , blockName = #matmul
                            , blockAxes =
                                [ AxisDecl (AxisId 0) #m Spatial (StaticDim 0) rows
                                , AxisDecl (AxisId 1) #n Spatial (StaticDim 0) columns
                                , AxisDecl (AxisId 2) #k Reduction (StaticDim 0) inner
                                ]
                            , blockDefinitions =
                                [ ReductionDef
                                    { definitionTarget = TensorId 2
                                    , definitionIndices = [AxisIndex (AxisId 0), AxisIndex (AxisId 1)]
                                    , definitionReducer = AddReducer
                                    , definitionIdentity = LiteralExpr (F32Literal 0)
                                    , definitionReduceAxes = [AxisId 2]
                                    , definitionValue =
                                        MulExpr
                                            (LoadExpr (TensorId 0) [AxisIndex (AxisId 0), AxisIndex (AxisId 2)])
                                            (LoadExpr (TensorId 1) [AxisIndex (AxisId 2), AxisIndex (AxisId 1)])
                                    }
                                ]
                            }
                        ]
                    }
        actual `shouldBe` expected

    it "elaborates ReLU to a pointwise definition" $ do
        let size = staticDim 32
            actual = program #relu $ do
                source <- input #source f32 [size]
                result <- output #result f32 [size]
                block #relu $ do
                    element <- spatial #i size
                    define result [element] (max_ 0 (source ! [element]))
            expected =
                ComputeIR
                    { computeName = #relu
                    , computeSymbols = []
                    , computeTensors =
                        [ TensorDecl (TensorId 0) #source F32Type [size] (InputTensor 0)
                        , TensorDecl (TensorId 1) #result F32Type [size] (OutputTensor 0)
                        ]
                    , computeBlocks =
                        [ ComputeBlock
                            { blockId = BlockId 0
                            , blockName = #relu
                            , blockAxes = [AxisDecl (AxisId 0) #i Spatial (StaticDim 0) size]
                            , blockDefinitions =
                                [ PointwiseDef
                                    { definitionTarget = TensorId 1
                                    , definitionIndices = [AxisIndex (AxisId 0)]
                                    , definitionValue =
                                        MaxExpr
                                            (LiteralExpr (F32Literal 0))
                                            (LoadExpr (TensorId 0) [AxisIndex (AxisId 0)])
                                    }
                                ]
                            }
                        ]
                    }
        actual `shouldBe` expected

    it "keeps declaration order separate from loop semantics" $ do
        let actual = program #copy $ do
                size <- dim #size
                source <- input #source f32 [size]
                result <- output #result f32 [size]
                block #copy $ do
                    element <- spatial #element size
                    define result [element] (source ! [element])
            expected =
                ComputeIR
                    { computeName = #copy
                    , computeSymbols = [Symbol (SymbolId 0) #size]
                    , computeTensors =
                        [ TensorDecl (TensorId 0) #source F32Type [SymbolDim (SymbolId 0)] (InputTensor 0)
                        , TensorDecl (TensorId 1) #result F32Type [SymbolDim (SymbolId 0)] (OutputTensor 0)
                        ]
                    , computeBlocks =
                        [ ComputeBlock
                            (BlockId 0)
                            #copy
                            [AxisDecl (AxisId 0) #element Spatial (StaticDim 0) (SymbolDim (SymbolId 0))]
                            [PointwiseDef (TensorId 1) [AxisIndex (AxisId 0)] (LoadExpr (TensorId 0) [AxisIndex (AxisId 0)])]
                        ]
                    }
        actual `shouldBe` expected
