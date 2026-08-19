module Target.Cuda.CodegenSpec (spec) where

import qualified Aoewif.Target.Cuda.Codegen as Codegen
import           Aoewif.Target.Cuda.DSL
import           Test.Hspec                 (Spec, describe, it, shouldBe)

spec :: Spec
spec =
    describe "generate" $ do
        it "renders a CUDA kernel" $ do
            Codegen.generate
                ( kernel "add" $ do
                    source <- parameter (Pointer (Const F32)) "source"
                    result <- parameter (Pointer F32) "result"
                    size <- parameter USize "size"
                    body $ do
                        index <-
                            define USize "index" $
                                cast USize blockIdxX
                                    .* cast USize blockDimX
                                    .+ cast USize threadIdxX
                        if_ (index .< size) $
                            result ! index .= result ! index .+ source ! index
                )
                `shouldBe` unlines
                    [ "extern \"C\" __global__ void add(float const* source, float* result, size_t size) {"
                    , "    size_t index = ((static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x)) + static_cast<size_t>(threadIdx.x));"
                    , "    if ((index < size)) {"
                    , "        (result[index] = (result[index] + source[index]));"
                    , "    }"
                    , "}"
                    ]

        it "renders static shared memory" $ do
            Codegen.generate
                ( kernel "shared_stage" $ do
                    source <- parameter (Pointer (Const F32)) "source"
                    result <- parameter (Pointer F32) "result"
                    body $ do
                        tile <- shared F32 "tile" (int 256)
                        tile ! threadIdxX .= source ! threadIdxX
                        syncThreads
                        result ! threadIdxX .= tile ! threadIdxX
                )
                `shouldBe` unlines
                    [ "extern \"C\" __global__ void shared_stage(float const* source, float* result) {"
                    , "    __shared__ float tile[256];"
                    , "    (tile[threadIdx.x] = source[threadIdx.x]);"
                    , "    __syncthreads();"
                    , "    (result[threadIdx.x] = tile[threadIdx.x]);"
                    , "}"
                    ]

        it "renders the remaining syntax forms" $ do
            Codegen.generate
                ( kernel "syntax" $ do
                    condition <- parameter Bool "condition"
                    _ <- parameter U32 "count"
                    _ <- parameter (Const (Pointer F32)) "pointer"
                    body $ do
                        _ <- declare F32 "value"
                        call_ (var "function") [float 1.25, int 2]
                        ifElse_
                            condition
                            ( do
                                expr_ threadIdxY
                                expr_ threadIdxZ
                                expr_ blockIdxY
                                expr_ blockIdxZ
                                expr_ blockDimY
                                expr_ blockDimZ
                                expr_ gridDimX
                                expr_ gridDimY
                                expr_ gridDimZ
                            )
                            (expr_ (int (-1)))
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
