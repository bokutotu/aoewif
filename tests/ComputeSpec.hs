{-# LANGUAGE OverloadedLabels #-}

module ComputeSpec (spec) where

import           Aoewif.Compute
import           Test.Hspec

spec :: Spec
spec = describe "compute eDSL" $ do
    it "builds an explicit matrix multiplication loop nest" $ do
        let rows = staticDim 4
            columns = staticDim 8
            inner = staticDim 16
            actual = program #matmul $ do
                left <- input #left f32 [rows, inner]
                right <- input #right f32 [inner, columns]
                result <- output #result f32 [rows, columns]
                for #m rows $ \row ->
                    for #n columns $ \column -> do
                        block #initialize $ do
                            store result [row, column] 0
                        for #k inner $ \reductionIndex ->
                            block #update $ do
                                lhs <- load left [row, reductionIndex]
                                rhs <- load right [reductionIndex, column]
                                update add result [row, column] (lhs * rhs)
            expected =
                IR
                    { irName = #matmul
                    , irSymbols = []
                    , irTensors =
                        [ TensorDecl (TensorId 0) #left F32Type [rows, inner] (InputTensor 0)
                        , TensorDecl (TensorId 1) #right F32Type [inner, columns] (InputTensor 1)
                        , TensorDecl (TensorId 2) #result F32Type [rows, columns] (OutputTensor 0)
                        ]
                    , irBody =
                        LoopIR
                            [ For
                                (Loop (LoopId 0) #m (StaticDim 0) rows)
                                ( LoopIR
                                    [ For
                                        (Loop (LoopId 1) #n (StaticDim 0) columns)
                                        ( LoopIR
                                            [ Execute
                                                ( Block
                                                    (BlockId 0)
                                                    #initialize
                                                    [ IndexBinding (IndexId 0) (LoopIndex (LoopId 0))
                                                    , IndexBinding (IndexId 1) (LoopIndex (LoopId 1))
                                                    ]
                                                    ( ComputeBlock
                                                        [ Store
                                                            (TensorId 2)
                                                            [ IterationIndex (IndexId 0)
                                                            , IterationIndex (IndexId 1)
                                                            ]
                                                            (DataLiteralExpr 0)
                                                        ]
                                                    )
                                                )
                                            , For
                                                (Loop (LoopId 2) #k (StaticDim 0) inner)
                                                ( LoopIR
                                                    [ Execute
                                                        ( Block
                                                            (BlockId 1)
                                                            #update
                                                            [ IndexBinding (IndexId 0) (LoopIndex (LoopId 0))
                                                            , IndexBinding (IndexId 1) (LoopIndex (LoopId 1))
                                                            , IndexBinding (IndexId 2) (LoopIndex (LoopId 2))
                                                            ]
                                                            ( ComputeBlock
                                                                [ Load
                                                                    (ComputeValueId 0)
                                                                    (TensorId 0)
                                                                    [ IterationIndex (IndexId 0)
                                                                    , IterationIndex (IndexId 2)
                                                                    ]
                                                                , Load
                                                                    (ComputeValueId 1)
                                                                    (TensorId 1)
                                                                    [ IterationIndex (IndexId 2)
                                                                    , IterationIndex (IndexId 1)
                                                                    ]
                                                                , Update
                                                                    AddReducer
                                                                    (TensorId 2)
                                                                    [ IterationIndex (IndexId 0)
                                                                    , IterationIndex (IndexId 1)
                                                                    ]
                                                                    ( MulExpr
                                                                        (ValueExpr (ComputeValueId 0))
                                                                        (ValueExpr (ComputeValueId 1))
                                                                    )
                                                                ]
                                                            )
                                                        )
                                                    ]
                                                )
                                            ]
                                        )
                                    ]
                                )
                            ]
                    }
        actual `shouldBe` expected

    it "keeps block statements in declaration order" $ do
        let size = staticDim 32
            actual = program #orderedBlock $ do
                left <- input #left f32 [size]
                right <- input #right f32 [size]
                result <- output #result f32 [size]
                for #element size $ \element ->
                    block #operations $ do
                        lhs <- load left [element]
                        rhs <- load right [element]
                        store result [element] lhs
                        update add result [element] rhs
                        store result [element] (lhs + rhs)
                        update multiply result [element] (lhs * rhs)
            expected =
                IR
                    { irName = #orderedBlock
                    , irSymbols = []
                    , irTensors =
                        [ TensorDecl (TensorId 0) #left F32Type [size] (InputTensor 0)
                        , TensorDecl (TensorId 1) #right F32Type [size] (InputTensor 1)
                        , TensorDecl (TensorId 2) #result F32Type [size] (OutputTensor 0)
                        ]
                    , irBody =
                        LoopIR
                            [ For
                                (Loop (LoopId 0) #element (StaticDim 0) size)
                                ( LoopIR
                                    [ Execute
                                        ( Block
                                            (BlockId 0)
                                            #operations
                                            [IndexBinding (IndexId 0) (LoopIndex (LoopId 0))]
                                            ( ComputeBlock
                                                [ Load
                                                    (ComputeValueId 0)
                                                    (TensorId 0)
                                                    [IterationIndex (IndexId 0)]
                                                , Load
                                                    (ComputeValueId 1)
                                                    (TensorId 1)
                                                    [IterationIndex (IndexId 0)]
                                                , Store
                                                    (TensorId 2)
                                                    [IterationIndex (IndexId 0)]
                                                    (ValueExpr (ComputeValueId 0))
                                                , Update
                                                    AddReducer
                                                    (TensorId 2)
                                                    [IterationIndex (IndexId 0)]
                                                    (ValueExpr (ComputeValueId 1))
                                                , Store
                                                    (TensorId 2)
                                                    [IterationIndex (IndexId 0)]
                                                    ( AddExpr
                                                        (ValueExpr (ComputeValueId 0))
                                                        (ValueExpr (ComputeValueId 1))
                                                    )
                                                , Update
                                                    MulReducer
                                                    (TensorId 2)
                                                    [IterationIndex (IndexId 0)]
                                                    ( MulExpr
                                                        (ValueExpr (ComputeValueId 0))
                                                        (ValueExpr (ComputeValueId 1))
                                                    )
                                                ]
                                            )
                                        )
                                    ]
                                )
                            ]
                    }
        actual `shouldBe` expected
