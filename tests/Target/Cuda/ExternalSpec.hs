module Target.Cuda.ExternalSpec (spec) where

import           Aoewif.Target.Cuda.DSL      (Expr, call, int, var)
import           Aoewif.Target.Cuda.External (external)
import           Test.Hspec                  (Spec, describe, it, shouldBe)

spec :: Spec
spec = describe "external functions" $ do
    it "builds a call without arguments" $ do
        let finish :: Expr
            finish = external "finish"
        finish `shouldBe` call (var "finish") []

    it "builds a call with one argument" $ do
        let convert :: Expr -> Expr
            convert = external "convert"
        convert (int 1) `shouldBe` call (var "convert") [int 1]

    it "preserves argument order in a call with two arguments" $ do
        let combine :: Expr -> Expr -> Expr
            combine = external "combine"
        combine (int 1) (int 2) `shouldBe` call (var "combine") [int 1, int 2]

    it "supports partial application and reusing a partially applied function" $ do
        let combine :: Expr -> Expr -> Expr
            combine = external "combine"
            combineFirst :: Expr -> Expr
            combineFirst = combine (int 1)
        combineFirst (int 2) `shouldBe` call (var "combine") [int 1, int 2]
        combineFirst (int 3) `shouldBe` call (var "combine") [int 1, int 3]
