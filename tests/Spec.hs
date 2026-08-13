module Main (main) where

import qualified CodegenSpec
import qualified ComputeSpec
import qualified IntegrationSpec
import qualified ScheduleSpec
import           Test.Hspec      (hspec)

main :: IO ()
main = hspec $ do
    ComputeSpec.spec
    ScheduleSpec.spec
    CodegenSpec.spec
    IntegrationSpec.spec
