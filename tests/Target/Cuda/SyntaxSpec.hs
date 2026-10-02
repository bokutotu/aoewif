module Target.Cuda.SyntaxSpec (spec) where

import qualified Aoewif.Target.Cuda.Codegen   as Codegen
import           Aoewif.Target.Cuda.Expr      (BinaryOp (..), Expr (..))
import           Aoewif.Target.Cuda.Kernel    (Kernel (..))
import           Aoewif.Target.Cuda.Name      (Name (..))
import           Aoewif.Target.Cuda.Parameter (Parameter (..))
import           Aoewif.Target.Cuda.Render    (Render (render))
import           Aoewif.Target.Cuda.Stmt      (Stmt (..))
import           Aoewif.Target.Cuda.Type      (Type (..))
import           Test.Hspec                   (Spec, describe, it, shouldBe)

spec :: Spec
spec =
    describe "CUDA syntax" $ do
        it "renders a kernel with relative indentation for nested loops and alternatives" $ do
            let index = Var (Name "index")
                output = Var (Name "output")
                generated =
                    render $
                        Kernel
                            (Name "nested")
                            [Parameter (Pointer F32) (Name "output")]
                            [ For
                                (Just (VarDecl U32 (Name "index") (Just (IntLit 0))))
                                (Binary LessThan index (IntLit 4))
                                (Just (Binary Assign index (Binary Add index (IntLit 1))))
                                [ If
                                    (Binary Equal index (IntLit 0))
                                    [ SyncThreads
                                    , NamedBarrierSync (IntLit 1) (IntLit 32)
                                    ]
                                    ( Just
                                        [ For
                                            Nothing
                                            (BoolLit False)
                                            Nothing
                                            [NamedBarrierArrive (IntLit 2) (IntLit 32)]
                                        ]
                                    )
                                , ExprStmt (Binary Assign (Subscript output index) (FloatLit 1.25))
                                ]
                            ]
                expected =
                    unlines
                        [ "extern \"C\" __global__ void nested(float* output) {"
                        , "    for (uint32_t index = 0; (index < 4); (index = (index + 1))) {"
                        , "        if ((index == 0)) {"
                        , "            __syncthreads();"
                        , "            asm volatile(\"barrier.sync %0, %1;\" :: \"r\"(1), \"r\"(32) : \"memory\");"
                        , "        } else {"
                        , "            for (; false; ) {"
                        , "                asm volatile(\"barrier.arrive %0, %1;\" :: \"r\"(2), \"r\"(32) : \"memory\");"
                        , "            }"
                        , "        }"
                        , "        (output[index] = 1.25f);"
                        , "    }"
                        , "}"
                        ]
            generated `shouldBe` expected

        it "renders declarations and expressions in for headers without statement terminators" $ do
            let index = Var (Name "index")
                generated =
                    map
                        render
                        [ VarDecl U32 (Name "index") Nothing
                        , For
                            (Just (VarDecl U32 (Name "index") Nothing))
                            (BoolLit False)
                            Nothing
                            []
                        , For
                            (Just (ExprStmt (Binary Assign index (IntLit 0))))
                            (Binary LessThan index (IntLit 4))
                            (Just (Binary Assign index (Binary Add index (IntLit 1))))
                            []
                        ]
                expected =
                    [ "uint32_t index;\n"
                    , "for (uint32_t index; false; ) {\n}\n"
                    , "for ((index = 0); (index < 4); (index = (index + 1))) {\n}\n"
                    ]
            generated `shouldBe` expected

        it "renders empty blocks without introducing blank lines" $ do
            let generated =
                    map
                        render
                        [ If (BoolLit True) [] Nothing
                        , If (BoolLit False) [] (Just [])
                        , For Nothing (BoolLit True) Nothing []
                        ]
                expected =
                    [ "if (true) {\n}\n"
                    , "if (false) {\n} else {\n}\n"
                    , "for (; true; ) {\n}\n"
                    ]
            generated `shouldBe` expected

        it "prepends configured includes without changing kernel rendering" $ do
            let generated =
                    Codegen.generateWith
                        (Codegen.Config [Codegen.CudaFp16Header, Codegen.CudaBf16Header])
                        ( Kernel
                            (Name "parameters")
                            [ Parameter (Pointer (Const F16)) (Name "source")
                            , Parameter (Pointer BF16) (Name "destination")
                            , Parameter (Const USize) (Name "count")
                            ]
                            []
                        )
                expected =
                    unlines
                        [ "#include <stdint.h>"
                        , "#include <cuda/ptx>"
                        , "#include <cuda_fp16.h>"
                        , "#include <cuda_bf16.h>"
                        , ""
                        , "extern \"C\" __global__ void parameters(__half const* source, __nv_bfloat16* destination, size_t const count) {"
                        , "}"
                        ]
            generated `shouldBe` expected
