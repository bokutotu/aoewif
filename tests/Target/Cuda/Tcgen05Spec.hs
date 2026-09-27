module Target.Cuda.Tcgen05Spec (spec) where

import qualified Aoewif.Target.Cuda.Codegen     as Codegen
import           Aoewif.Target.Cuda.DSL
import           Aoewif.Target.Cuda.Tcgen05.DSL
import           Test.Hspec                     (Spec, describe, it, shouldBe)

spec :: Spec
spec =
    describe "Tcgen05 instructions" $
        it "renders the TMEM allocation from hoge.cu" $ do
            let generated = Codegen.generate $ kernel "allocate_tmem" $ body $ do
                    tmem <- shared NaturalAlignment U32 "tmem" (int 1)
                    tid <- define U32 "tid" threadIdxX
                    if_ (tid .< int 32) $
                        tcgen05Alloc tmem (int 32)
                expected =
                    unlines
                        [ "#include <stdint.h>"
                        , "#include <cuda/ptx>"
                        , ""
                        , "extern \"C\" __global__ void allocate_tmem() {"
                        , "    __shared__ uint32_t tmem[1];"
                        , "    uint32_t tid = threadIdx.x;"
                        , "    if ((tid < 32)) {"
                        , "        cuda::ptx::tcgen05_alloc(cuda::ptx::cta_group_1, tmem, 32);"
                        , "    }"
                        , "}"
                        ]
            generated `shouldBe` expected
