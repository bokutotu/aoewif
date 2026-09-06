module Target.Cuda.DSLSpec (spec) where

import qualified Aoewif.Target.Cuda.Codegen as Codegen
import           Aoewif.Target.Cuda.DSL
import           Test.Hspec                 (Spec, describe, it, shouldBe)

spec :: Spec
spec =
    describe "CUDA DSL" $ do
        it "renders shared declarations with natural, 16-byte, and 128-byte alignment" $ do
            let generated = Codegen.generate $ kernel "shared_alignments" $ body $ do
                    _ <- shared NaturalAlignment U32 "natural" (int 4)
                    _ <- shared Align16 U32 "aligned16" (int 8)
                    _ <- shared Align128 U32 "aligned128" (int 32)
                    pure ()
                expected =
                    unlines
                        [ "#include <stdint.h>"
                        , ""
                        , "extern \"C\" __global__ void shared_alignments() {"
                        , "    __shared__ uint32_t natural[4];"
                        , "    __shared__ __align__(16) uint32_t aligned16[8];"
                        , "    __shared__ __align__(128) uint32_t aligned128[32];"
                        , "}"
                        ]
            generated `shouldBe` expected

        it "renders thread indices, block indices, and launch dimensions" $ do
            let generated =
                    fmap
                        Codegen.renderExpr
                        [ threadIdxX
                        , threadIdxY
                        , threadIdxZ
                        , blockIdxX
                        , blockIdxY
                        , blockIdxZ
                        , blockDimX
                        , blockDimY
                        , blockDimZ
                        , gridDimX
                        , gridDimY
                        , gridDimZ
                        ]
                expected =
                    [ "threadIdx.x"
                    , "threadIdx.y"
                    , "threadIdx.z"
                    , "blockIdx.x"
                    , "blockIdx.y"
                    , "blockIdx.z"
                    , "blockDim.x"
                    , "blockDim.y"
                    , "blockDim.z"
                    , "gridDim.x"
                    , "gridDim.y"
                    , "gridDim.z"
                    ]
            generated `shouldBe` expected

        it "renders floating-point literals, casts, and composed bitcasts" $ do
            let generated =
                    fmap
                        Codegen.renderExpr
                        [ float 1.25
                        , cast USize blockDimX
                        , bitcast F32 (var "bits")
                        , bitcast F32 (var "bits") .+ float 1
                        ]
                expected =
                    [ "1.25f"
                    , "static_cast<size_t>(blockDim.x)"
                    , "(*reinterpret_cast<float*>(&bits))"
                    , "((*reinterpret_cast<float*>(&bits)) + 1.0f)"
                    ]
            generated `shouldBe` expected

        it "renders arithmetic, comparisons, and bitwise and logical operators with Haskell fixities" $ do
            let generated =
                    fmap
                        Codegen.renderExpr
                        [ var "lhs" ./ var "rhs"
                        , var "lhs" .% var "rhs"
                        , var "lhs" .- var "rhs"
                        , var "lhs" .== var "rhs"
                        , var "lhs" ./= var "rhs"
                        , var "lhs" .<= var "rhs"
                        , var "lhs" .>= var "rhs"
                        , var "lhs" .&& var "rhs"
                        , var "lhs" .|| var "rhs"
                        , not_ (var "condition")
                        , complement (var "value")
                        , shiftL (var "value") (var "amount")
                        , var "lhs" .|. var "rhs"
                        , ifElse (var "condition") (var "trueValue") (var "falseValue")
                        , var "first" .|. var "second" .&. var "third"
                        , var "lhs" .== var "rhs" .&& var "value" ./= int 0 .|| var "fallback"
                        , var "lhs" .- var "rhs" ./ int 2 .% int 3
                        ]
                expected =
                    [ "(lhs / rhs)"
                    , "(lhs % rhs)"
                    , "(lhs - rhs)"
                    , "(lhs == rhs)"
                    , "(lhs != rhs)"
                    , "(lhs <= rhs)"
                    , "(lhs >= rhs)"
                    , "(lhs && rhs)"
                    , "(lhs || rhs)"
                    , "(!condition)"
                    , "(~value)"
                    , "(value << amount)"
                    , "(lhs | rhs)"
                    , "(condition ? trueValue : falseValue)"
                    , "(first | (second & third))"
                    , "(((lhs == rhs) && (value != 0)) || fallback)"
                    , "(lhs - ((rhs / 2) % 3))"
                    ]
            generated `shouldBe` expected

        it "renders calls as expressions and statements" $ do
            let generated = Codegen.generate $ kernel "calls" $ body $ do
                    value <- define F32 "value" (call (var "sqrtf") [float 4])
                    call_ (var "consume") [value, int 2]
                    call_ (var "finish") []
                expected =
                    unlines
                        [ "#include <stdint.h>"
                        , ""
                        , "extern \"C\" __global__ void calls() {"
                        , "    float value = sqrtf(4.0f);"
                        , "    consume(value, 2);"
                        , "    finish();"
                        , "}"
                        ]
            generated `shouldBe` expected
