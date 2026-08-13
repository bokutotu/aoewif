{-# LANGUAGE OverloadedLabels #-}
{-# LANGUAGE TypeApplications #-}

module CodegenSpec (spec) where

import           Aoewif
import           GHC.OverloadedLabels (fromLabel)
import           Prelude              hiding (compare)
import           Test.Hspec

newtype VectorAxis scope = VectorAxis (Axis scope Spatial)

spec :: Spec
spec = describe "codegen errors" $ do
    it "rejects an invalid C identifier at the code generation boundary" $ do
        Right invalidProgram <- pure $ program (fromLabel @"invalid-name") $ do
            let size = staticDim 1
            output <- compute #output size $ \element ->
                ( VectorAxis element
                , select
                    (compare Equal (index element) (indexLiteral 0))
                    (f32 0)
                    (f32 0)
                )
            entry output
        Right invalidSchedule <- pure $ cpu invalidProgram (\_ -> pure ())
        generateC invalidSchedule
            `shouldBe` Left (InvalidFunctionIdentifier "invalid-name")
