{-# LANGUAGE OverloadedLabels #-}

module ScheduleSpec (spec) where

import           Aoewif.Compute
import qualified Aoewif.Schedule as Schedule
import           Test.Hspec

spec :: Spec
spec = describe "schedule eDSL" $ do
    it "splits and reorders a perfect loop chain" $ do
        let rows = staticDim 8
            columns = staticDim 6
            computeIR = program #copy_2d $ do
                source <- input #source f32 [rows, columns]
                result <- output #result f32 [rows, columns]
                for #row rows $ \row ->
                    for #column columns $ \column ->
                        block #copy $ do
                            value <- load source [row, column]
                            store result [row, column] value
            actual = Schedule.schedule computeIR $ do
                copyBlock <- Schedule.block #copy
                row <- Schedule.loopOf copyBlock #row
                column <- Schedule.loopOf copyBlock #column
                (rowOuter, rowInner) <- Schedule.split row 4
                (columnOuter, columnInner) <- Schedule.split column 3
                Schedule.reorder [rowOuter, columnOuter, rowInner, columnInner]
            rowIndex =
                AddIndex
                    (MulIndex (LoopIndex (LoopId 2)) (ConstantIndex 4))
                    (LoopIndex (LoopId 3))
            columnIndex =
                AddIndex
                    (MulIndex (LoopIndex (LoopId 4)) (ConstantIndex 3))
                    (LoopIndex (LoopId 5))
            expected =
                IR
                    { irName = #copy_2d
                    , irSymbols = []
                    , irTensors =
                        [ TensorDecl (TensorId 0) #source F32Type [rows, columns] (InputTensor 0)
                        , TensorDecl (TensorId 1) #result F32Type [rows, columns] (OutputTensor 0)
                        ]
                    , irBody =
                        LoopIR
                            [ For (Loop (LoopId 2) #row_outer (StaticDim 0) (CeilDivDim rows 4)) $
                                LoopIR
                                    [ For (Loop (LoopId 4) #column_outer (StaticDim 0) (CeilDivDim columns 3)) $
                                        LoopIR
                                            [ For (Loop (LoopId 3) #row_inner (StaticDim 0) (StaticDim 4)) $
                                                LoopIR
                                                    [ For (Loop (LoopId 5) #column_inner (StaticDim 0) (StaticDim 3)) $
                                                        LoopIR
                                                            [ Execute
                                                                ( Block
                                                                    (BlockId 0)
                                                                    #copy
                                                                    [ IndexBinding (IndexId 0) rowIndex
                                                                    , IndexBinding (IndexId 1) columnIndex
                                                                    ]
                                                                    ( ComputeBlock
                                                                        [ Load
                                                                            (ComputeValueId 0)
                                                                            (TensorId 0)
                                                                            [ IterationIndex (IndexId 0)
                                                                            , IterationIndex (IndexId 1)
                                                                            ]
                                                                        , Store
                                                                            (TensorId 1)
                                                                            [ IterationIndex (IndexId 0)
                                                                            , IterationIndex (IndexId 1)
                                                                            ]
                                                                            (ValueExpr (ComputeValueId 0))
                                                                        ]
                                                                    )
                                                                )
                                                            ]
                                                    ]
                                            ]
                                    ]
                            ]
                    }
        actual `shouldBe` expected

    it "guards the inner loop when splitting a dynamic extent" $ do
        let computeIR = program #copy $ do
                size <- dim #size
                source <- input #source f32 [size]
                result <- output #result f32 [size]
                for #element size $ \element ->
                    block #copy $ do
                        value <- load source [element]
                        store result [element] value
            actual = Schedule.schedule computeIR $ do
                copyBlock <- Schedule.block #copy
                element <- Schedule.loopOf copyBlock #element
                _ <- Schedule.split element 32
                pure ()
            logicalIndex =
                AddIndex
                    (MulIndex (LoopIndex (LoopId 1)) (ConstantIndex 32))
                    (LoopIndex (LoopId 2))
            extent = SymbolDim (SymbolId 0)
            expected =
                IR
                    { irName = #copy
                    , irSymbols = [Symbol (SymbolId 0) #size]
                    , irTensors =
                        [ TensorDecl (TensorId 0) #source F32Type [extent] (InputTensor 0)
                        , TensorDecl (TensorId 1) #result F32Type [extent] (OutputTensor 0)
                        ]
                    , irBody =
                        LoopIR
                            [ For (Loop (LoopId 1) #element_outer (StaticDim 0) (CeilDivDim extent 32)) $
                                LoopIR
                                    [ For (Loop (LoopId 2) #element_inner (StaticDim 0) (StaticDim 32)) $
                                        LoopIR
                                            [ Guard
                                                (IndexLessThan logicalIndex (DimensionIndex extent))
                                                ( LoopIR
                                                    [ Execute
                                                        ( Block
                                                            (BlockId 0)
                                                            #copy
                                                            [IndexBinding (IndexId 0) logicalIndex]
                                                            ( ComputeBlock
                                                                [ Load
                                                                    (ComputeValueId 0)
                                                                    (TensorId 0)
                                                                    [IterationIndex (IndexId 0)]
                                                                , Store
                                                                    (TensorId 1)
                                                                    [IterationIndex (IndexId 0)]
                                                                    (ValueExpr (ComputeValueId 0))
                                                                ]
                                                            )
                                                        )
                                                    ]
                                                )
                                            ]
                                    ]
                            ]
                    }
        actual `shouldBe` expected

    it "reorders split loops before guarding dynamic blocks" $ do
        let computeIR = program #copy_2d $ do
                rows <- dim #rows
                columns <- dim #columns
                source <- input #source f32 [rows, columns]
                result <- output #result f32 [rows, columns]
                for #row rows $ \row ->
                    for #column columns $ \column ->
                        block #copy $ do
                            value <- load source [row, column]
                            store result [row, column] value
            actual = Schedule.schedule computeIR $ do
                copyBlock <- Schedule.block #copy
                row <- Schedule.loopOf copyBlock #row
                column <- Schedule.loopOf copyBlock #column
                (rowOuter, rowInner) <- Schedule.split row 4
                (columnOuter, columnInner) <- Schedule.split column 8
                Schedule.reorder [rowOuter, columnOuter, rowInner, columnInner]
            rowExtent = SymbolDim (SymbolId 0)
            columnExtent = SymbolDim (SymbolId 1)
            rowIndex =
                AddIndex
                    (MulIndex (LoopIndex (LoopId 2)) (ConstantIndex 4))
                    (LoopIndex (LoopId 3))
            columnIndex =
                AddIndex
                    (MulIndex (LoopIndex (LoopId 4)) (ConstantIndex 8))
                    (LoopIndex (LoopId 5))
            expectedBlock =
                Execute
                    ( Block
                        (BlockId 0)
                        #copy
                        [ IndexBinding (IndexId 0) rowIndex
                        , IndexBinding (IndexId 1) columnIndex
                        ]
                        ( ComputeBlock
                            [ Load
                                (ComputeValueId 0)
                                (TensorId 0)
                                [IterationIndex (IndexId 0), IterationIndex (IndexId 1)]
                            , Store
                                (TensorId 1)
                                [IterationIndex (IndexId 0), IterationIndex (IndexId 1)]
                                (ValueExpr (ComputeValueId 0))
                            ]
                        )
                    )
            expected =
                IR
                    { irName = #copy_2d
                    , irSymbols =
                        [ Symbol (SymbolId 0) #rows
                        , Symbol (SymbolId 1) #columns
                        ]
                    , irTensors =
                        [ TensorDecl (TensorId 0) #source F32Type [rowExtent, columnExtent] (InputTensor 0)
                        , TensorDecl (TensorId 1) #result F32Type [rowExtent, columnExtent] (OutputTensor 0)
                        ]
                    , irBody =
                        LoopIR
                            [ For (Loop (LoopId 2) #row_outer (StaticDim 0) (CeilDivDim rowExtent 4)) $
                                LoopIR
                                    [ For (Loop (LoopId 4) #column_outer (StaticDim 0) (CeilDivDim columnExtent 8)) $
                                        LoopIR
                                            [ For (Loop (LoopId 3) #row_inner (StaticDim 0) (StaticDim 4)) $
                                                LoopIR
                                                    [ For (Loop (LoopId 5) #column_inner (StaticDim 0) (StaticDim 8)) $
                                                        LoopIR
                                                            [ Guard
                                                                ( IndexLessThan
                                                                    rowIndex
                                                                    (DimensionIndex rowExtent)
                                                                )
                                                                ( LoopIR
                                                                    [ Guard
                                                                        ( IndexLessThan
                                                                            columnIndex
                                                                            (DimensionIndex columnExtent)
                                                                        )
                                                                        (LoopIR [expectedBlock])
                                                                    ]
                                                                )
                                                            ]
                                                    ]
                                            ]
                                    ]
                            ]
                    }
        actual `shouldBe` expected

    it "reorders a contiguous subset of a loop chain" $ do
        let size = staticDim 4
            computeIR = program #copy_3d $ do
                source <- input #source f32 [size, size, size]
                result <- output #result f32 [size, size, size]
                for #outer size $ \outer ->
                    for #middle size $ \middle ->
                        for #inner size $ \inner ->
                            block #copy $ do
                                value <- load source [outer, middle, inner]
                                store result [outer, middle, inner] value
            actual = Schedule.schedule computeIR $ do
                copyBlock <- Schedule.block #copy
                middle <- Schedule.loopOf copyBlock #middle
                inner <- Schedule.loopOf copyBlock #inner
                Schedule.reorder [inner, middle]
            operation =
                ComputeBlock
                    [ Load
                        (ComputeValueId 0)
                        (TensorId 0)
                        [ IterationIndex (IndexId 0)
                        , IterationIndex (IndexId 1)
                        , IterationIndex (IndexId 2)
                        ]
                    , Store
                        (TensorId 1)
                        [ IterationIndex (IndexId 0)
                        , IterationIndex (IndexId 1)
                        , IterationIndex (IndexId 2)
                        ]
                        (ValueExpr (ComputeValueId 0))
                    ]
            expected =
                IR
                    { irName = #copy_3d
                    , irSymbols = []
                    , irTensors =
                        [ TensorDecl (TensorId 0) #source F32Type [size, size, size] (InputTensor 0)
                        , TensorDecl (TensorId 1) #result F32Type [size, size, size] (OutputTensor 0)
                        ]
                    , irBody =
                        LoopIR
                            [ For (Loop (LoopId 0) #outer (StaticDim 0) size) $
                                LoopIR
                                    [ For (Loop (LoopId 2) #inner (StaticDim 0) size) $
                                        LoopIR
                                            [ For (Loop (LoopId 1) #middle (StaticDim 0) size) $
                                                LoopIR
                                                    [ Execute
                                                        ( Block
                                                            (BlockId 0)
                                                            #copy
                                                            [ IndexBinding (IndexId 0) (LoopIndex (LoopId 0))
                                                            , IndexBinding (IndexId 1) (LoopIndex (LoopId 1))
                                                            , IndexBinding (IndexId 2) (LoopIndex (LoopId 2))
                                                            ]
                                                            operation
                                                        )
                                                    ]
                                            ]
                                    ]
                            ]
                    }
        actual `shouldBe` expected

    it "splits a loop with a non-zero lower bound" $ do
        let size = staticDim 5
            computeIR =
                IR
                    { irName = #offset
                    , irSymbols = []
                    , irTensors =
                        [TensorDecl (TensorId 0) #result F32Type [staticDim 12] (OutputTensor 0)]
                    , irBody =
                        LoopIR
                            [ For (Loop (LoopId 0) #element (StaticDim 7) size) $
                                LoopIR
                                    [ Execute
                                        ( Block
                                            (BlockId 0)
                                            #write
                                            [IndexBinding (IndexId 0) (LoopIndex (LoopId 0))]
                                            ( ComputeBlock
                                                [ Store
                                                    (TensorId 0)
                                                    [IterationIndex (IndexId 0)]
                                                    (DataLiteralExpr 1)
                                                ]
                                            )
                                        )
                                    ]
                            ]
                    }
            actual = Schedule.schedule computeIR $ do
                writeBlock <- Schedule.block #write
                element <- Schedule.loopOf writeBlock #element
                _ <- Schedule.split element 2
                pure ()
            logicalIndex =
                AddIndex
                    (DimensionIndex (StaticDim 7))
                    ( AddIndex
                        (MulIndex (LoopIndex (LoopId 1)) (ConstantIndex 2))
                        (LoopIndex (LoopId 2))
                    )
            expected =
                computeIR
                    { irBody =
                        LoopIR
                            [ For (Loop (LoopId 1) #element_outer (StaticDim 0) (CeilDivDim size 2)) $
                                LoopIR
                                    [ For (Loop (LoopId 2) #element_inner (StaticDim 0) (StaticDim 2)) $
                                        LoopIR
                                            [ Guard
                                                ( IndexLessThan
                                                    logicalIndex
                                                    ( AddIndex
                                                        (DimensionIndex (StaticDim 7))
                                                        (DimensionIndex size)
                                                    )
                                                )
                                                ( LoopIR
                                                    [ Execute
                                                        ( Block
                                                            (BlockId 0)
                                                            #write
                                                            [IndexBinding (IndexId 0) logicalIndex]
                                                            ( ComputeBlock
                                                                [ Store
                                                                    (TensorId 0)
                                                                    [IterationIndex (IndexId 0)]
                                                                    (DataLiteralExpr 1)
                                                                ]
                                                            )
                                                        )
                                                    ]
                                                )
                                            ]
                                    ]
                            ]
                    }
        actual `shouldBe` expected
