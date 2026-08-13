{-# LANGUAGE OverloadedLabels #-}
{-# LANGUAGE TypeApplications #-}

module ComputeSpec (spec) where

import           Aoewif.Compute
import           Data.Functor         (void)
import           GHC.OverloadedLabels (fromLabel)
import           Prelude              hiding (compare, exp, log, maximum,
                                       minimum)
import           Test.Hspec

newtype VectorAxis scope = VectorAxis (Axis scope Spatial)

data VolumeAxes scope
    = VolumeAxes
        (Axis scope Spatial)
        (Axis scope Spatial)
        (Axis scope Spatial)

spec :: Spec
spec = describe "compute eDSL" $ do
    it "constructs a scoped elementwise program" $ do
        void
            ( program #copy $ do
                let size = staticDim 4
                source <- input @F32 #source size
                output <- compute #output size $ \element ->
                    (VectorAxis element, source ! element)
                entry output
            )
            `shouldBe` Right ()

    it "infers vector, matrix, and volume ranks from their shapes" $ do
        void
            ( program #ranked_shapes $ do
                depth <- dim #depth
                let rows = staticDim 3
                    columns = staticDim 5
                vector <- input @F32 #vector depth
                matrix <- input @F32 #matrix (depth, rows)
                volume <- input @F32 #volume (depth, rows, columns)
                output <- compute #output (depth, rows, columns) $ \(depthAxis, rowAxis, columnAxis) ->
                    ( VolumeAxes depthAxis rowAxis columnAxis
                    , vector
                        ! depthAxis
                        .+. matrix
                        ! (depthAxis, rowAxis)
                        .+. volume
                        ! (depthAxis, rowAxis, columnAxis)
                    )
                entry output
            )
            `shouldBe` Right ()

    it "lowers typed scalar literals, arithmetic, comparisons, and selection" $ do
        void
            ( program #scalar_expressions $ do
                output <- compute #output (staticDim 4) $ \element ->
                    ( VectorAxis element
                    , select
                        (boolean True)
                        ( select
                            (compare LessThan (index element) (indexLiteral 2))
                            ( maximum
                                ( minimum
                                    (exp (log ((((f32 8 ./. f32 2) .*. f32 3) .-. f32 1) .+. f32 4)))
                                    (f32 20)
                                )
                                (f32 0)
                            )
                            (fma (f32 2) (f32 3) (f32 4))
                        )
                        (f32 (-1))
                    )
                entry output
            )
            `shouldBe` Right ()

    it "lowers a reduction fold with a typed accumulator" $ do
        void
            ( program #sum $ do
                size <- dim #size
                source <- input @F32 #source size
                output <- compute #output (staticDim 1) $ \element ->
                    ( VectorAxis element
                    , foldOver size 0 $ \reductionAxis accumulator ->
                        accumulator .+. source ! reductionAxis
                    )
                entry output
            )
            `shouldBe` Right ()

    it "fails while lowering an inconsistent tensor access" $ do
        void
            ( program #mismatched_shape $ do
                rows <- dim #rows
                columns <- dim #columns
                source <- input @F32 #source rows
                output <- compute #output columns $ \column ->
                    (VectorAxis column, source ! column)
                entry output
            )
            `shouldBe` Left DimensionMismatch

    it "fails while lowering a reduction over the wrong tensor dimension" $ do
        void
            ( program #mismatched_reduction $ do
                sourceSize <- dim #source_size
                reductionSize <- dim #reduction_size
                source <- input @F32 #source sourceSize
                output <- compute #output (staticDim 1) $ \element ->
                    ( VectorAxis element
                    , foldOver reductionSize 0 $ \reductionAxis accumulator ->
                        accumulator .+. source ! reductionAxis
                    )
                entry output
            )
            `shouldBe` Left DimensionMismatch

    it "rejects an invalid generated-function identifier at the input boundary" $ do
        void
            ( program (fromLabel @"invalid-name") $ do
                let size = staticDim 1
                output <- compute #output size $ \element ->
                    ( VectorAxis element
                    , select
                        (compare Equal (index element) (indexLiteral 0))
                        (f32 0)
                        (f32 0)
                    )
                entry output
            )
            `shouldBe` Left (InvalidFunctionIdentifier "invalid-name")
