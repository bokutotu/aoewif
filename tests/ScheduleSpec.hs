{-# LANGUAGE OverloadedLabels #-}

module ScheduleSpec (spec) where

import           Aoewif.Compute
import qualified Aoewif.Schedule as Schedule
import           Test.Hspec

spec :: Spec
spec = describe "schedule eDSL" $ do
    it "builds a canonical tiled schedule tree with logical-axis bindings" $ do
        let rows = staticDim 256
            columns = staticDim 256
            inner = staticDim 64
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
            actual = Schedule.cpu computeIR $ do
                matrixMultiply <- Schedule.block #matmul
                row <- Schedule.axis matrixMultiply #m
                column <- Schedule.axis matrixMultiply #n
                reductionAxis <- Schedule.axis matrixMultiply #k
                ((rowOuter, rowInner), (columnOuter, columnInner), (reductionOuter, reductionInner)) <-
                    Schedule.tile3 (row, column, reductionAxis) (128, 128, 32)
                Schedule.reorder matrixMultiply [rowOuter, columnOuter, reductionOuter, rowInner, columnInner, reductionInner]
                Schedule.parallel rowOuter
                Schedule.parallel columnOuter
                Schedule.unrollBy 4 reductionInner
            expected =
                Schedule.ScheduleIR
                    { Schedule.scheduleRoot =
                        Schedule.Sequence
                            [ Schedule.Band
                                [ Schedule.LoopDim (Schedule.LoopId 3) #m_outer (StaticDim 0) (CeilDivDim rows 128) Schedule.Parallel Nothing Nothing
                                , Schedule.LoopDim (Schedule.LoopId 5) #n_outer (StaticDim 0) (CeilDivDim columns 128) Schedule.Parallel Nothing Nothing
                                , Schedule.LoopDim (Schedule.LoopId 7) #k_outer (StaticDim 0) (CeilDivDim inner 32) Schedule.Serial Nothing Nothing
                                , Schedule.LoopDim (Schedule.LoopId 4) #m_inner (StaticDim 0) (StaticDim 128) Schedule.Serial Nothing Nothing
                                , Schedule.LoopDim (Schedule.LoopId 6) #n_inner (StaticDim 0) (StaticDim 128) Schedule.Serial Nothing Nothing
                                , Schedule.LoopDim (Schedule.LoopId 8) #k_inner (StaticDim 0) (StaticDim 32) Schedule.Serial (Just 4) Nothing
                                ]
                                ( Schedule.Leaf
                                    (BlockId 0)
                                    [ Schedule.AxisBinding
                                        (AxisId 0)
                                        ( Schedule.AddIndex
                                            (Schedule.MulIndex (Schedule.LoopIndex (Schedule.LoopId 3)) (Schedule.ConstantIndex 128))
                                            (Schedule.LoopIndex (Schedule.LoopId 4))
                                        )
                                    , Schedule.AxisBinding
                                        (AxisId 1)
                                        ( Schedule.AddIndex
                                            (Schedule.MulIndex (Schedule.LoopIndex (Schedule.LoopId 5)) (Schedule.ConstantIndex 128))
                                            (Schedule.LoopIndex (Schedule.LoopId 6))
                                        )
                                    , Schedule.AxisBinding
                                        (AxisId 2)
                                        ( Schedule.AddIndex
                                            (Schedule.MulIndex (Schedule.LoopIndex (Schedule.LoopId 7)) (Schedule.ConstantIndex 32))
                                            (Schedule.LoopIndex (Schedule.LoopId 8))
                                        )
                                    ]
                                )
                            ]
                    }
        Schedule.cpuScheduleIR actual `shouldBe` expected

    it "keeps CUDA bindings on loop dimensions and guards a split tail" $ do
        let computeIR = program #copy $ do
                size <- dim #size
                source <- input #source f32 [size]
                result <- output #result f32 [size]
                block #copy $ do
                    element <- spatial #i size
                    define result [element] (source ! [element])
            actual = Schedule.cuda computeIR $ do
                copyBlock <- Schedule.block #copy
                element <- Schedule.axis copyBlock #i
                (blockX, threadX) <- Schedule.split element 32
                Schedule.bind blockX Schedule.BlockX
                Schedule.bind threadX Schedule.ThreadX
            logicalIndex =
                Schedule.AddIndex
                    (Schedule.MulIndex (Schedule.LoopIndex (Schedule.LoopId 1)) (Schedule.ConstantIndex 32))
                    (Schedule.LoopIndex (Schedule.LoopId 2))
            expected =
                Schedule.ScheduleIR
                    { Schedule.scheduleRoot =
                        Schedule.Sequence
                            [ Schedule.Band
                                [ Schedule.LoopDim (Schedule.LoopId 1) #i_outer (StaticDim 0) (CeilDivDim (SymbolDim (SymbolId 0)) 32) Schedule.Serial Nothing (Just Schedule.BlockX)
                                , Schedule.LoopDim (Schedule.LoopId 2) #i_inner (StaticDim 0) (StaticDim 32) Schedule.Serial Nothing (Just Schedule.ThreadX)
                                ]
                                ( Schedule.Guard
                                    (Schedule.IndexLessThan logicalIndex (SymbolDim (SymbolId 0)))
                                    (Schedule.Leaf (BlockId 0) [Schedule.AxisBinding (AxisId 0) logicalIndex])
                                )
                            ]
                    }
        Schedule.cudaScheduleIR actual `shouldBe` expected
