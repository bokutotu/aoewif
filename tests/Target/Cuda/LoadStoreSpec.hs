module Target.Cuda.LoadStoreSpec (spec) where

import           Aoewif.Target.Cuda.AsmOperand
import           Aoewif.Target.Cuda.DSL.Builder   (body, kernel)
import qualified Aoewif.Target.Cuda.DSL.LoadStore as LS
import           Aoewif.Target.Cuda.Expr          (Expr (..))
import           Aoewif.Target.Cuda.Kernel        (Kernel (..))
import           Aoewif.Target.Cuda.LoadStore
import           Aoewif.Target.Cuda.Name          (Name (..))
import           Aoewif.Target.Cuda.Render        (Render (render))
import           Aoewif.Target.Cuda.Stmt          (Stmt (..))
import           Test.Hspec                       (Spec, describe, it, shouldBe)

spec :: Spec
spec = describe "vector load/store" $ do
    it "renders shared/global v2/v4 loads" $ do
        let a = writeOnly RegU32 (Var (Name "a"))
            b = writeOnly RegU32 (Var (Name "b"))
            c = writeOnly RegU32 (Var (Name "c"))
            d = writeOnly RegU32 (Var (Name "d"))
            address = Var (Name "address")
            generated =
                map
                    (render . VectorLoadExpr)
                    [ VectorLoad Shared V2 U32 [a, b] (input RegU32 address)
                    , VectorLoad Shared V4 U32 [a, b, c, d] (input RegU32 address)
                    , VectorLoad Global V2 U32 [a, b] (input RegU64 address)
                    , VectorLoad Global V4 U32 [a, b, c, d] (input RegU64 address)
                    ]
            expected =
                [ "asm volatile(\n    \"ld.shared.v2.u32 {%0, %1}, [%2];\\n\"\n    : \"=r\"(a), \"=r\"(b)\n    : \"r\"(address)\n    : \"memory\"\n)"
                , "asm volatile(\n    \"ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\\n\"\n    : \"=r\"(a), \"=r\"(b), \"=r\"(c), \"=r\"(d)\n    : \"r\"(address)\n    : \"memory\"\n)"
                , "asm volatile(\n    \"ld.global.v2.u32 {%0, %1}, [%2];\\n\"\n    : \"=r\"(a), \"=r\"(b)\n    : \"l\"(address)\n    : \"memory\"\n)"
                , "asm volatile(\n    \"ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\\n\"\n    : \"=r\"(a), \"=r\"(b), \"=r\"(c), \"=r\"(d)\n    : \"l\"(address)\n    : \"memory\"\n)"
                ]
        generated `shouldBe` expected

    it "renders shared/global v2/v4 stores" $ do
        let a = input RegU32 (Var (Name "a"))
            b = input RegU32 (Var (Name "b"))
            c = input RegU32 (Var (Name "c"))
            d = input RegU32 (Var (Name "d"))
            address = Var (Name "address")
            generated =
                map
                    (render . VectorStoreExpr)
                    [ VectorStore Shared V2 U32 (input RegU32 address) [a, b]
                    , VectorStore Shared V4 U32 (input RegU32 address) [a, b, c, d]
                    , VectorStore Global V2 U32 (input RegU64 address) [a, b]
                    , VectorStore Global V4 U32 (input RegU64 address) [a, b, c, d]
                    ]
            expected =
                [ "asm volatile(\n    \"st.shared.v2.u32 [%0], {%1, %2};\\n\"\n    :\n    : \"r\"(address), \"r\"(a), \"r\"(b)\n    : \"memory\"\n)"
                , "asm volatile(\n    \"st.shared.v4.u32 [%0], {%1, %2, %3, %4};\\n\"\n    :\n    : \"r\"(address), \"r\"(a), \"r\"(b), \"r\"(c), \"r\"(d)\n    : \"memory\"\n)"
                , "asm volatile(\n    \"st.global.v2.u32 [%0], {%1, %2};\\n\"\n    :\n    : \"l\"(address), \"r\"(a), \"r\"(b)\n    : \"memory\"\n)"
                , "asm volatile(\n    \"st.global.v4.u32 [%0], {%1, %2, %3, %4};\\n\"\n    :\n    : \"l\"(address), \"r\"(a), \"r\"(b), \"r\"(c), \"r\"(d)\n    : \"memory\"\n)"
                ]
        generated `shouldBe` expected

    it "renders b32 and f32 operands" $ do
        let a = Var (Name "a")
            b = Var (Name "b")
            address = input RegU64 (Var (Name "address"))
            generated =
                [ render (VectorLoad Global V2 B32 [writeOnly RegU32 a, writeOnly RegU32 b] address)
                , render (VectorStore Global V2 B32 address [input RegU32 a, input RegU32 b])
                , render (VectorLoad Global V2 F32 [writeOnly RegF32 a, writeOnly RegF32 b] address)
                , render (VectorStore Global V2 F32 address [input RegF32 a, input RegF32 b])
                ]
            expected =
                [ "asm volatile(\n    \"ld.global.v2.b32 {%0, %1}, [%2];\\n\"\n    : \"=r\"(a), \"=r\"(b)\n    : \"l\"(address)\n    : \"memory\"\n)"
                , "asm volatile(\n    \"st.global.v2.b32 [%0], {%1, %2};\\n\"\n    :\n    : \"l\"(address), \"r\"(a), \"r\"(b)\n    : \"memory\"\n)"
                , "asm volatile(\n    \"ld.global.v2.f32 {%0, %1}, [%2];\\n\"\n    : \"=f\"(a), \"=f\"(b)\n    : \"l\"(address)\n    : \"memory\"\n)"
                , "asm volatile(\n    \"st.global.v2.f32 [%0], {%1, %2};\\n\"\n    :\n    : \"l\"(address), \"f\"(a), \"f\"(b)\n    : \"memory\"\n)"
                ]
        generated `shouldBe` expected

    it "appends vector load/store expression statements through the DSL" $ do
        let a = Var (Name "a")
            b = Var (Name "b")
            address = input RegU32 (Var (Name "address"))
            outputs = [writeOnly RegU32 a, writeOnly RegU32 b]
            inputs = [input RegU32 a, input RegU32 b]
            generated = kernel "transfer" $ body $ do
                LS.vectorLoad LS.Shared LS.V2 LS.U32 outputs address
                LS.vectorStore LS.Shared LS.V2 LS.U32 address inputs
            expected =
                [ ExprStmt (VectorLoadExpr (VectorLoad Shared V2 U32 outputs address))
                , ExprStmt (VectorStoreExpr (VectorStore Shared V2 U32 address inputs))
                ]
        map render (kernelBody generated) `shouldBe` map render expected
