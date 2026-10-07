module Target.Cuda.RenderSpec (spec) where

import           Aoewif.Target.Cuda.Alignment (Alignment (..))
import           Aoewif.Target.Cuda.Expr      (Axis (..), BinaryOp (..),
                                               Expr (..))
import           Aoewif.Target.Cuda.Name      (Name (..))
import           Aoewif.Target.Cuda.Parameter (Parameter (..))
import           Aoewif.Target.Cuda.Render    (Render (render), indent)
import           Aoewif.Target.Cuda.Type      (Type (..))
import           Test.Hspec                   (Spec, describe, it, shouldBe)

spec :: Spec
spec =
    describe "CUDA rendering" $ do
        it "renders scalar types and nested qualifiers" $ do
            let generated =
                    map
                        render
                        [ Bool
                        , U32
                        , USize
                        , F16
                        , BF16
                        , F32
                        , Const U32
                        , Pointer (Const F16)
                        , Const (Pointer F32)
                        , Pointer (Pointer BF16)
                        ]
                expected =
                    [ "bool"
                    , "uint32_t"
                    , "size_t"
                    , "__half"
                    , "__nv_bfloat16"
                    , "float"
                    , "uint32_t const"
                    , "__half const*"
                    , "float* const"
                    , "__nv_bfloat16**"
                    ]
            generated `shouldBe` expected

        it "renders names and parameters" $ do
            let generated =
                    [ render (Name "matrix_a")
                    , render (Parameter (Pointer (Const F16)) (Name "matrix_a"))
                    , render (Parameter (Pointer F32) (Name "matrix_d"))
                    ]
                expected =
                    [ "matrix_a"
                    , "__half const* matrix_a"
                    , "float* matrix_d"
                    ]
            generated `shouldBe` expected

        it "renders optional alignment prefixes" $ do
            let generated = map render [NaturalAlignment, Align16, Align128]
                expected = ["", "__align__(16) ", "__align__(128) "]
            generated `shouldBe` expected

        it "renders axes" $ do
            let generated = map render [X, Y, Z]
                expected = ["x", "y", "z"]
            generated `shouldBe` expected

        it "renders builtin expressions for all axes" $ do
            let generated =
                    [ map render [ThreadIdx X, ThreadIdx Y, ThreadIdx Z]
                    , map render [BlockIdx X, BlockIdx Y, BlockIdx Z]
                    , map render [BlockDim X, BlockDim Y, BlockDim Z]
                    , map render [GridDim X, GridDim Y, GridDim Z]
                    ]
                expected =
                    [ ["threadIdx.x", "threadIdx.y", "threadIdx.z"]
                    , ["blockIdx.x", "blockIdx.y", "blockIdx.z"]
                    , ["blockDim.x", "blockDim.y", "blockDim.z"]
                    , ["gridDim.x", "gridDim.y", "gridDim.z"]
                    ]
            generated `shouldBe` expected

        it "renders binary operator tokens" $ do
            let generated =
                    map
                        render
                        [ Assign
                        , Add
                        , Subtract
                        , Multiply
                        , Divide
                        , Modulo
                        , Equal
                        , NotEqual
                        , LessThan
                        , LessThanOrEqual
                        , GreaterThanOrEqual
                        , LogicalAnd
                        , LogicalOr
                        , ShiftLeft
                        , ShiftRight
                        , BitAnd
                        , BitXor
                        , BitOr
                        ]
                expected =
                    [ "="
                    , "+"
                    , "-"
                    , "*"
                    , "/"
                    , "%"
                    , "=="
                    , "!="
                    , "<"
                    , "<="
                    , ">="
                    , "&&"
                    , "||"
                    , "<<"
                    , ">>"
                    , "&"
                    , "^"
                    , "|"
                    ]
            generated `shouldBe` expected

        it "renders non-finite floating-point literals" $ do
            let generated =
                    map
                        render
                        [ FloatLit (0 / 0)
                        , FloatLit (1 / 0)
                        , FloatLit (-(1 / 0))
                        ]
                expected = ["NAN", "INFINITY", "-INFINITY"]
            generated `shouldBe` expected

        it "indents each line of a rendered block and leaves empty blocks empty" $ do
            let generated =
                    map
                        indent
                        [ ""
                        , "first();\nsecond();\n"
                        , "if (condition) {\n    inner();\n}\n"
                        ]
                expected =
                    [ ""
                    , "    first();\n    second();\n"
                    , "    if (condition) {\n        inner();\n    }\n"
                    ]
            generated `shouldBe` expected
