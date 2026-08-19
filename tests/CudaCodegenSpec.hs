module CudaCodegenSpec (spec) where

import qualified Aoewif.Target.Cuda.Codegen as Codegen
import qualified Aoewif.Target.Cuda.Syntax  as Syntax
import           Test.Hspec                 (Spec, describe, it, shouldBe)

spec :: Spec
spec =
    describe "generate" $ do
        it "renders a CUDA kernel" $ do
            Codegen.generate
                ( Syntax.Kernel
                    (Syntax.Name "add")
                    [ Syntax.Parameter
                        (Syntax.Pointer (Syntax.Const Syntax.F32))
                        (Syntax.Name "source")
                    , Syntax.Parameter
                        (Syntax.Pointer Syntax.F32)
                        (Syntax.Name "result")
                    , Syntax.Parameter Syntax.USize (Syntax.Name "size")
                    ]
                    [ Syntax.VarDecl
                        Syntax.USize
                        (Syntax.Name "index")
                        ( Just
                            ( Syntax.Binary
                                Syntax.Add
                                ( Syntax.Binary
                                    Syntax.Multiply
                                    ( Syntax.Unary
                                        (Syntax.StaticCast Syntax.USize)
                                        (Syntax.BlockIdx Syntax.BlockIdxX)
                                    )
                                    ( Syntax.Unary
                                        (Syntax.StaticCast Syntax.USize)
                                        (Syntax.BlockDim Syntax.BlockDimX)
                                    )
                                )
                                ( Syntax.Unary
                                    (Syntax.StaticCast Syntax.USize)
                                    (Syntax.ThreadIdx Syntax.ThreadIdxX)
                                )
                            )
                        )
                    , Syntax.If
                        ( Syntax.Binary
                            Syntax.LessThan
                            (Syntax.Var (Syntax.Name "index"))
                            (Syntax.Var (Syntax.Name "size"))
                        )
                        [ Syntax.ExprStmt
                            ( Syntax.Binary
                                Syntax.Assign
                                ( Syntax.Subscript
                                    (Syntax.Var (Syntax.Name "result"))
                                    (Syntax.Var (Syntax.Name "index"))
                                )
                                ( Syntax.Binary
                                    Syntax.Add
                                    ( Syntax.Subscript
                                        (Syntax.Var (Syntax.Name "result"))
                                        (Syntax.Var (Syntax.Name "index"))
                                    )
                                    ( Syntax.Subscript
                                        (Syntax.Var (Syntax.Name "source"))
                                        (Syntax.Var (Syntax.Name "index"))
                                    )
                                )
                            )
                        ]
                        Nothing
                    ]
                )
                `shouldBe` unlines
                    [ "extern \"C\" __global__ void add(float const* source, float* result, size_t size) {"
                    , "    size_t index = ((static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x)) + static_cast<size_t>(threadIdx.x));"
                    , "    if ((index < size)) {"
                    , "        (result[index] = (result[index] + source[index]));"
                    , "    }"
                    , "}"
                    ]

        it "renders the remaining syntax forms" $ do
            Codegen.generate
                ( Syntax.Kernel
                    (Syntax.Name "syntax")
                    [ Syntax.Parameter Syntax.Bool (Syntax.Name "condition")
                    , Syntax.Parameter Syntax.U32 (Syntax.Name "count")
                    , Syntax.Parameter
                        (Syntax.Const (Syntax.Pointer Syntax.F32))
                        (Syntax.Name "pointer")
                    ]
                    [ Syntax.VarDecl Syntax.F32 (Syntax.Name "value") Nothing
                    , Syntax.ExprStmt
                        ( Syntax.Call
                            (Syntax.Var (Syntax.Name "function"))
                            [ Syntax.FloatLit 1.25
                            , Syntax.IntLit 2
                            ]
                        )
                    , Syntax.If
                        (Syntax.Var (Syntax.Name "condition"))
                        [ Syntax.ExprStmt (Syntax.ThreadIdx Syntax.ThreadIdxY)
                        , Syntax.ExprStmt (Syntax.ThreadIdx Syntax.ThreadIdxZ)
                        , Syntax.ExprStmt (Syntax.BlockIdx Syntax.BlockIdxY)
                        , Syntax.ExprStmt (Syntax.BlockIdx Syntax.BlockIdxZ)
                        , Syntax.ExprStmt (Syntax.BlockDim Syntax.BlockDimY)
                        , Syntax.ExprStmt (Syntax.BlockDim Syntax.BlockDimZ)
                        , Syntax.ExprStmt (Syntax.GridDim Syntax.GridDimX)
                        , Syntax.ExprStmt (Syntax.GridDim Syntax.GridDimY)
                        , Syntax.ExprStmt (Syntax.GridDim Syntax.GridDimZ)
                        ]
                        (Just [Syntax.ExprStmt (Syntax.IntLit (-1))])
                    ]
                )
                `shouldBe` unlines
                    [ "extern \"C\" __global__ void syntax(bool condition, uint32_t count, float* const pointer) {"
                    , "    float value;"
                    , "    function(1.25f, 2);"
                    , "    if (condition) {"
                    , "        threadIdx.y;"
                    , "        threadIdx.z;"
                    , "        blockIdx.y;"
                    , "        blockIdx.z;"
                    , "        blockDim.y;"
                    , "        blockDim.z;"
                    , "        gridDim.x;"
                    , "        gridDim.y;"
                    , "        gridDim.z;"
                    , "    } else {"
                    , "        -1;"
                    , "    }"
                    , "}"
                    ]
