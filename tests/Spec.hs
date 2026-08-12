module Main (main) where

import qualified CodegenSpec
import qualified IRSpec
import qualified ScheduleSpec
import           Test.Hspec   (hspec)

main :: IO ()
main = hspec $ do
    IRSpec.spec
    ScheduleSpec.spec
    CodegenSpec.spec
