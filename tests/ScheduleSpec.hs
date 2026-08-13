{-# LANGUAGE OverloadedLabels #-}
{-# LANGUAGE TypeApplications #-}

module ScheduleSpec (spec) where

import           Aoewif.Compute
import           Aoewif.Schedule
import           Data.Functor    (void)
import           Test.Hspec

data VectorAxis scope = VectorAxis
    { vectorAxis :: Axis scope Spatial
    }

data MatrixAxes scope = MatrixAxes
    { rowAxis    :: Axis scope Spatial
    , columnAxis :: Axis scope Spatial
    }

spec :: Spec
spec = describe "schedule eDSL" $ do
    it "splits, reorders, and binds spatial loops" $ do
        Right sourceProgram <- pure $ program #copy_2d $ do
            let rows = staticDim 128
                columns = staticDim 70
            source <- input @F32 #source (rows, columns)
            output <- compute #output (rows, columns) $ \(row, column) ->
                (MatrixAxes row column, source ! (row, column))
            entry output
        void
            ( cuda defaultCudaTarget sourceProgram $ \MatrixAxes{rowAxis, columnAxis} -> do
                row <- loop rowAxis
                column <- loop columnAxis
                (blockY, threadY) <- split row 16
                (blockX, threadX) <- split column 32
                reorder [blockY, blockX, threadY, threadX]
                bind blockY BlockY
                bind blockX BlockX
                bind threadY ThreadY
                bind threadX ThreadX
            )
            `shouldBe` Right ()

    it "reorders spatial loops while preserving a hidden reduction loop" $ do
        Right sourceProgram <- pure $ program #row_sum $ do
            let rows = staticDim 8
                columns = staticDim 16
                reductionSize = staticDim 4
            source <- input @F32 #source (rows, columns, reductionSize)
            output <- compute #output (rows, columns) $ \(row, column) ->
                ( MatrixAxes row column
                , foldOver reductionSize 0 $ \reductionAxis accumulator ->
                    source ! (row, column, reductionAxis) .+. accumulator
                )
            entry output
        void
            ( cpu sourceProgram $ \MatrixAxes{rowAxis, columnAxis} -> do
                row <- loop rowAxis
                column <- loop columnAxis
                reorder [column, row]
            )
            `shouldBe` Right ()

    it "rejects a zero split factor at the transformation" $ do
        Right sourceProgram <- pure $ program #copy $ do
            let size = staticDim 8
            source <- input @F32 #source size
            output <- compute #output size $ \element ->
                (VectorAxis element, source ! element)
            entry output
        void
            ( cpu sourceProgram $ \VectorAxis{vectorAxis} -> do
                element <- loop vectorAxis
                _ <- split element 0
                pure ()
            )
            `shouldBe` Left ZeroSplitFactor

    it "rejects a loop handle after that loop has been split" $ do
        Right sourceProgram <- pure $ program #copy $ do
            let size = staticDim 8
            source <- input @F32 #source size
            output <- compute #output size $ \element ->
                (VectorAxis element, source ! element)
            entry output
        void
            ( cpu sourceProgram $ \VectorAxis{vectorAxis} -> do
                element <- loop vectorAxis
                _ <- split element 4
                _ <- split element 2
                pure ()
            )
            `shouldBe` Left (UnknownLoop 0)

    it "checks a zero split factor before a stale loop handle" $ do
        Right sourceProgram <- pure $ program #copy $ do
            let size = staticDim 8
            source <- input @F32 #source size
            output <- compute #output size $ \element ->
                (VectorAxis element, source ! element)
            entry output
        void
            ( cpu sourceProgram $ \VectorAxis{vectorAxis} -> do
                element <- loop vectorAxis
                _ <- split element 4
                _ <- split element 0
                pure ()
            )
            `shouldBe` Left ZeroSplitFactor

    it "rejects incomplete and duplicate loop orders" $ do
        Right sourceProgram <- pure $ program #copy_2d $ do
            let rows = staticDim 8
                columns = staticDim 16
            source <- input @F32 #source (rows, columns)
            output <- compute #output (rows, columns) $ \(row, column) ->
                (MatrixAxes row column, source ! (row, column))
            entry output
        void
            ( cpu sourceProgram $ \MatrixAxes{rowAxis} -> do
                row <- loop rowAxis
                reorder [row]
            )
            `shouldBe` Left (IncompleteLoopOrder 2 1)
        void
            ( cpu sourceProgram $ \MatrixAxes{rowAxis} -> do
                row <- loop rowAxis
                reorder [row, row]
            )
            `shouldBe` Left (DuplicateLoop 0)

    it "rejects splitting a loop after CUDA binding" $ do
        Right sourceProgram <- pure $ program #copy $ do
            let size = staticDim 8
            source <- input @F32 #source size
            output <- compute #output size $ \element ->
                (VectorAxis element, source ! element)
            entry output
        void
            ( cuda defaultCudaTarget sourceProgram $ \VectorAxis{vectorAxis} -> do
                element <- loop vectorAxis
                bind element ThreadX
                _ <- split element 2
                pure ()
            )
            `shouldBe` Left (BoundLoopSplitUnsupported "axis0")

    it "rejects rebinding a loop and reusing a CUDA binding" $ do
        Right sourceProgram <- pure $ program #copy_2d $ do
            let rows = staticDim 8
                columns = staticDim 16
            source <- input @F32 #source (rows, columns)
            output <- compute #output (rows, columns) $ \(row, column) ->
                (MatrixAxes row column, source ! (row, column))
            entry output
        void
            ( cuda defaultCudaTarget sourceProgram $ \MatrixAxes{rowAxis} -> do
                row <- loop rowAxis
                bind row ThreadX
                bind row ThreadY
            )
            `shouldBe` Left (LoopAlreadyBound "axis0" ThreadX)
        void
            ( cuda defaultCudaTarget sourceProgram $ \MatrixAxes{rowAxis, columnAxis} -> do
                row <- loop rowAxis
                column <- loop columnAxis
                bind row ThreadX
                bind column ThreadX
            )
            `shouldBe` Left (BindingAlreadyUsed ThreadX "axis0")

    it "allows a dynamic grid extent after splitting off static threads" $ do
        Right sourceProgram <- pure $ program #dynamic_copy $ do
            size <- dim #size
            source <- input @F32 #source size
            output <- compute #output size $ \element ->
                (VectorAxis element, source ! element)
            entry output
        void
            ( cuda defaultCudaTarget sourceProgram $ \VectorAxis{vectorAxis} -> do
                element <- loop vectorAxis
                (blockX, threadX) <- split element 32
                bind blockX BlockX
                bind threadX ThreadX
            )
            `shouldBe` Right ()

    it "rejects a dynamic CUDA thread extent when binding" $ do
        Right sourceProgram <- pure $ program #dynamic_copy $ do
            size <- dim #size
            source <- input @F32 #source size
            output <- compute #output size $ \element ->
                (VectorAxis element, source ! element)
            entry output
        void
            ( cuda defaultCudaTarget sourceProgram $ \VectorAxis{vectorAxis} -> do
                element <- loop vectorAxis
                bind element ThreadX
            )
            `shouldBe` Left (DynamicCudaThreadExtent "axis0")

    it "rejects zero CUDA launch dimensions when binding" $ do
        Right sourceProgram <- pure $ program #empty_copy $ do
            let size = staticDim 0
            source <- input @F32 #source size
            output <- compute #output size $ \element ->
                (VectorAxis element, source ! element)
            entry output
        void
            ( cuda defaultCudaTarget sourceProgram $ \VectorAxis{vectorAxis} -> do
                element <- loop vectorAxis
                bind element ThreadX
            )
            `shouldBe` Left (ZeroCudaLaunchDimension ThreadX)

    it "rejects zero CUDA target limits before scheduling" $ do
        Right sourceProgram <- pure $ program #copy $ do
            let size = staticDim 1
            source <- input @F32 #source size
            output <- compute #output size $ \element ->
                (VectorAxis element, source ! element)
            entry output
        let block = newCudaDim3 1024 1024 64
            grid = newCudaDim3 2147483647 65535 65535
            results =
                [ cuda (newCudaTarget 0 block grid) sourceProgram (\_ -> pure ())
                , cuda (newCudaTarget 1024 (newCudaDim3 0 1024 64) grid) sourceProgram (\_ -> pure ())
                , cuda (newCudaTarget 1024 (newCudaDim3 1024 0 64) grid) sourceProgram (\_ -> pure ())
                , cuda (newCudaTarget 1024 (newCudaDim3 1024 1024 0) grid) sourceProgram (\_ -> pure ())
                , cuda (newCudaTarget 1024 block (newCudaDim3 0 65535 65535)) sourceProgram (\_ -> pure ())
                , cuda (newCudaTarget 1024 block (newCudaDim3 2147483647 0 65535)) sourceProgram (\_ -> pure ())
                , cuda (newCudaTarget 1024 block (newCudaDim3 2147483647 65535 0)) sourceProgram (\_ -> pure ())
                ]
        map void results
            `shouldBe` [ Left (InvalidCudaTargetLimit "max threads per block")
                       , Left (InvalidCudaTargetLimit "max block dimension x")
                       , Left (InvalidCudaTargetLimit "max block dimension y")
                       , Left (InvalidCudaTargetLimit "max block dimension z")
                       , Left (InvalidCudaTargetLimit "max grid dimension x")
                       , Left (InvalidCudaTargetLimit "max grid dimension y")
                       , Left (InvalidCudaTargetLimit "max grid dimension z")
                       ]

    it "enforces CUDA block and grid dimension limits" $ do
        Right sourceProgram <- pure $ program #copy $ do
            let size = staticDim 33
            source <- input @F32 #source size
            output <- compute #output size $ \element ->
                (VectorAxis element, source ! element)
            entry output
        void
            ( cuda (newCudaTarget 1024 (newCudaDim3 32 1024 64) (newCudaDim3 1024 1024 1024)) sourceProgram $ \VectorAxis{vectorAxis} -> do
                element <- loop vectorAxis
                bind element ThreadX
            )
            `shouldBe` Left (CudaDimensionExceeded ThreadX 33 32)
        void
            ( cuda (newCudaTarget 1024 (newCudaDim3 1024 1024 64) (newCudaDim3 32 1024 1024)) sourceProgram $ \VectorAxis{vectorAxis} -> do
                element <- loop vectorAxis
                bind element BlockX
            )
            `shouldBe` Left (CudaDimensionExceeded BlockX 33 32)

    it "enforces the total CUDA threads per block" $ do
        Right sourceProgram <- pure $ program #copy_2d $ do
            let rows = staticDim 32
                columns = staticDim 16
            source <- input @F32 #source (rows, columns)
            output <- compute #output (rows, columns) $ \(row, column) ->
                (MatrixAxes row column, source ! (row, column))
            entry output
        void
            ( cuda (newCudaTarget 256 (newCudaDim3 1024 1024 64) (newCudaDim3 1024 1024 1024)) sourceProgram $ \MatrixAxes{rowAxis, columnAxis} -> do
                row <- loop rowAxis
                column <- loop columnAxis
                bind row ThreadX
                bind column ThreadY
            )
            `shouldBe` Left (CudaThreadsPerBlockExceeded 512 256)

    it "checks Word64 overflow in nested dynamic split divisors" $ do
        Right sourceProgram <- pure $ program #dynamic_copy $ do
            size <- dim #size
            source <- input @F32 #source size
            output <- compute #output size $ \element ->
                (VectorAxis element, source ! element)
            entry output
        void
            ( cpu sourceProgram $ \VectorAxis{vectorAxis} -> do
                element <- loop vectorAxis
                (outer, _) <- split element maxBound
                _ <- split outer 2
                pure ()
            )
            `shouldBe` Left (ArithmeticOverflow "split loop divisor")
