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
                `shouldBe` """
                           extern "C" __global__ void add(float const* source, float* result, size_t size) {
                               size_t index = ((static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x)) + static_cast<size_t>(threadIdx.x));
                               if ((index < size)) {
                                   (result[index] = (result[index] + source[index]));
                               }
                           }

                           """

        it "renders a for loop" $ do
            Codegen.generate
                ( kernel "loop" $ do
                    result <- parameter (Pointer F32) "result"
                    k <- parameter U32 "k"
                    body $ do
                        for_ (define U32 "kk" (int 0)) (.< k) (\kk -> kk .+ int 16) $ \kk -> do
                            result ! kk .= float 0
                )
                `shouldBe` """
                           extern "C" __global__ void loop(float* result, uint32_t k) {
                               for (uint32_t kk = 0; (kk < k); (kk = (kk + 16))) {
                                   (result[kk] = 0.0f);
                               }
                           }

                           """

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
                `shouldBe` """
                           extern "C" __global__ void shared_stage(float const* source, float* result) {
                               __shared__ float tile[256];
                               (tile[threadIdx.x] = source[threadIdx.x]);
                               __syncthreads();
                               (result[threadIdx.x] = tile[threadIdx.x]);
                           }

                           """

        it "renders XOR-swizzled shared addresses" $ do
            Codegen.generate
                ( kernel "swizzled" $ body $ do
                    tile <- shared F32 "tile" (int 128)
                    tile
                        ! ( (threadIdxX .+ int 16)
                                .^ (((threadIdxX .+ int 16) .>> int 3) .& int 7)
                          )
                        .= int 0
                )
                `shouldBe` """
                           extern "C" __global__ void swizzled() {
                               __shared__ float tile[128];
                               (tile[((threadIdx.x + 16) ^ (((threadIdx.x + 16) >> 3) & 7))] = 0);
                           }

                           """

        it "renders configured half-precision types and headers" $ do
            Codegen.generateWith
                ( Codegen.Config
                    [ Codegen.CudaFp16Header
                    , Codegen.CudaBf16Header
                    ]
                )
                ( kernel "half_types" $ do
                    source <- parameter (Pointer (Const F16)) "source"
                    result <- parameter (Pointer BF16) "result"
                    body $ do
                        value <- define F16 "value" (cast F16 (float 1.25))
                        tile <- shared BF16 "tile" (int 32)
                        tile
                            ! threadIdxX
                            .= cast BF16 (source ! threadIdxX .+ value)
                        result ! threadIdxX .= tile ! threadIdxX
                )
                `shouldBe` """
                           #include <cuda_fp16.h>
                           #include <cuda_bf16.h>

                           extern "C" __global__ void half_types(__half const* source, __nv_bfloat16* result) {
                               __half value = static_cast<__half>(1.25f);
                               __shared__ __nv_bfloat16 tile[32];
                               (tile[threadIdx.x] = static_cast<__nv_bfloat16>((source[threadIdx.x] + value)));
                               (result[threadIdx.x] = tile[threadIdx.x]);
                           }

                           """

        it "renders configured TensorFloat-32 types and headers" $ do
            Codegen.generateWith
                (Codegen.Config [Codegen.CudaTf32Header])
                ( kernel "tf32_types" $ do
                    source <- parameter (Pointer (Const TF32)) "source"
                    result <- parameter (Pointer TF32) "result"
                    body $
                        result ! threadIdxX .= source ! threadIdxX
                )
                `shouldBe` """
                           #include <cuda_tf32.h>

                           extern "C" __global__ void tf32_types(__nv_tf32 const* source, __nv_tf32* result) {
                               (result[threadIdx.x] = source[threadIdx.x]);
                           }

                           """

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
                `shouldBe` """
                           extern "C" __global__ void syntax(bool condition, uint32_t count, float* const pointer) {
                               float value;
                               function(1.25f, 2);
                               if (condition) {
                                   threadIdx.y;
                                   threadIdx.z;
                                   blockIdx.y;
                                   blockIdx.z;
                                   blockDim.y;
                                   blockDim.z;
                                   gridDim.x;
                                   gridDim.y;
                                   gridDim.z;
                               } else {
                                   -1;
                               }
                           }

                           """

        it "renders subtraction, modulo, bitcast, and a 128B swizzle expression" $ do
            -- The 128B swizzle for f16 elements stored in 16B rows: permutes
            -- the 16B granules of each 128B group while keeping the element
            -- offset within a granule, so cp.async/ldmatrix addresses stay 16B
            -- aligned. Composed from the raw operators, not a DSL primitive.
            let swizzled index =
                    ((index .>> int 3) .^ ((index .>> int 6) .& int 7)) .* int 8 .+ (index .& int 7)
            Codegen.generate
                ( kernel "ops" $ do
                    index <- parameter U32 "index"
                    value <- parameter U32 "value"
                    result <- parameter (Pointer F32) "result"
                    body $ do
                        tile <- shared F16 "tile" (int 128)
                        expr_ (index .- int 16)
                        expr_ (index .% int 4)
                        expr_ (swizzled (threadIdxX .+ index))
                        tile ! swizzled index .= cast F16 (float 1)
                        result ! index .= bitcast F32 value
                )
                `shouldBe` """
                           extern "C" __global__ void ops(uint32_t index, uint32_t value, float* result) {
                               __shared__ __half tile[128];
                               (index - 16);
                               (index % 4);
                               (((((threadIdx.x + index) >> 3) ^ (((threadIdx.x + index) >> 6) & 7)) * 8) + ((threadIdx.x + index) & 7));
                               (tile[((((index >> 3) ^ ((index >> 6) & 7)) * 8) + (index & 7))] = static_cast<__half>(1.0f));
                               (result[index] = *reinterpret_cast<float*>(&value));
                           }

                           """
