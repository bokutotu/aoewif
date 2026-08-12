module CodegenSpec (spec) where

import Aoewif
import GHC.Float (castWord32ToFloat)
import Prelude hiding (compare)
import Test.Hspec hiding (parallel)

data ElementwiseFixture = ElementwiseFixture
  { elementwiseFunction :: VerifiedComputeFunction,
    elementwiseOperation :: ComputeOpId,
    elementwiseColumn :: IteratorId
  }

elementwiseWithIndexSelect :: Either IrError ElementwiseFixture
elementwiseWithIndexSelect = do
  function <- buildComputeFunction "choose_by_column" $ do
    columns <- symbol "columns"
    left <- input "left" (tensorTypeF32 [StaticDim 2, SymbolDim columns])
    right <- input "right" (tensorTypeF32 [StaticDim 2, SymbolDim columns])
    output <- compute "output" $ do
      row <- parallel "row" (StaticDim 2)
      column <- parallel "column" (SymbolDim columns)
      leftValue <- readTensor left [IteratorIndex row, IteratorIndex column]
      rightValue <- readTensor right [IteratorIndex row, IteratorIndex column]
      columnIndex <- index column
      threshold <- constant (IndexLiteral 4)
      condition <- compare GreaterEqual columnIndex threshold
      selectScalar condition leftValue rightValue
    markOutput output
  let operation = onlyOperation function
      column = computeIterators operation !! 1
  pure
    ElementwiseFixture
      { elementwiseFunction = function,
        elementwiseOperation = computeOpId operation,
        elementwiseColumn = iteratorId column
      }

gemm :: Either IrError (VerifiedComputeFunction, ComputeOpId)
gemm = do
  function <- buildComputeFunction "gemm" $ do
    rows <- symbol "rows"
    columns <- symbol "columns"
    reductionSize <- symbol "reduction"
    left <- input "left" (tensorTypeF32 [SymbolDim rows, SymbolDim reductionSize])
    right <- input "right" (tensorTypeF32 [SymbolDim reductionSize, SymbolDim columns])
    output <- compute "output" $ do
      row <- parallel "row" (SymbolDim rows)
      column <- parallel "column" (SymbolDim columns)
      inner <- reduction "inner" (SymbolDim reductionSize)
      leftValue <- readTensor left [IteratorIndex row, IteratorIndex inner]
      rightValue <- readTensor right [IteratorIndex inner, IteratorIndex column]
      accumulator <- reductionInit (F32Literal 0.0)
      fma leftValue rightValue accumulator
    markOutput output
  let operation = onlyOperation function
  pure (function, computeOpId operation)

data CudaFixture = CudaFixture
  { cudaFunction :: VerifiedComputeFunction,
    cudaOperation :: ComputeOpId,
    cudaRow :: IteratorId,
    cudaColumn :: IteratorId
  }

cudaElementwise :: Either IrError CudaFixture
cudaElementwise = do
  function <- buildComputeFunction "cuda_add" $ do
    left <- input "left" (tensorTypeF32 [StaticDim 5, StaticDim 10])
    right <- input "right" (tensorTypeF32 [StaticDim 5, StaticDim 10])
    output <- compute "output" $ do
      row <- parallel "row" (StaticDim 5)
      column <- parallel "column" (StaticDim 10)
      leftValue <- readTensor left [IteratorIndex row, IteratorIndex column]
      rightValue <- readTensor right [IteratorIndex row, IteratorIndex column]
      add leftValue rightValue
    markOutput output
  let operation = onlyOperation function
      row = iteratorAt 0 operation
      column = iteratorAt 1 operation
  pure
    CudaFixture
      { cudaFunction = function,
        cudaOperation = computeOpId operation,
        cudaRow = iteratorId row,
        cudaColumn = iteratorId column
      }

twoOperationFunction :: Either IrError (VerifiedComputeFunction, ComputeOpId)
twoOperationFunction = do
  function <- buildComputeFunction "two_operations" $ do
    inputValue <- input "input" (tensorTypeF32 [StaticDim 4])
    intermediate <- compute "intermediate" $ do
      element <- parallel "element" (StaticDim 4)
      readTensor inputValue [IteratorIndex element]
    output <- compute "output" $ do
      element <- parallel "element" (StaticDim 4)
      readTensor intermediate [IteratorIndex element]
    markOutput output
  let operation = functionOperations (verifiedFunction function) !! 1
  pure (function, computeOpId operation)

spec :: Spec
spec = describe "source generation" $ do
  it "generates complete CPU elementwise source with split tail and ABI" $ do
    fixture <- rightOrFail elementwiseWithIndexSelect
    initialSchedule <- rightOrFail (newCpuSchedule (elementwiseFunction fixture) (elementwiseOperation fixture))
    column <- maybe (expectationFailure "missing column loop" >> fail "missing column loop") pure (cpuLoopFor (elementwiseColumn fixture) initialSchedule)
    (_, _, splitSchedule) <- rightOrFail (splitCpuSchedule column 4 initialSchedule)
    schedule <- rightOrFail (verifyCpuSchedule splitSchedule)
    actual <- rightOrFail (generateC schedule)
    (cSourceText actual, cFunctionName actual)
      `shouldBe` (expectedCpuElementwise, "choose_by_column")

  it "generates complete CPU strict GEMM source" $ do
    (function, operation) <- rightOrFail gemm
    initialSchedule <- rightOrFail (newCpuSchedule function operation)
    schedule <- rightOrFail (verifyCpuSchedule initialSchedule)
    actual <- rightOrFail (generateC schedule)
    (cSourceText actual, cFunctionName actual)
      `shouldBe` (expectedCpuGemm, "gemm")

  it "generates complete CUDA source with block and thread bindings" $ do
    fixture <- rightOrFail cudaElementwise
    initialSchedule <- rightOrFail (newCudaSchedule (cudaFunction fixture) (cudaOperation fixture) defaultCudaTarget)
    row <- maybe (expectationFailure "missing row loop" >> fail "missing row loop") pure (cudaLoopFor (cudaRow fixture) initialSchedule)
    column <- maybe (expectationFailure "missing column loop" >> fail "missing column loop") pure (cudaLoopFor (cudaColumn fixture) initialSchedule)
    (rowOuter, rowInner, rowSplitSchedule) <- rightOrFail (splitCudaSchedule row 2 initialSchedule)
    (columnOuter, columnInner, splitSchedule) <- rightOrFail (splitCudaSchedule column 4 rowSplitSchedule)
    reorderedSchedule <- rightOrFail (reorderCudaSchedule [rowOuter, columnOuter, rowInner, columnInner] splitSchedule)
    blockYSchedule <- rightOrFail (bindCudaSchedule rowOuter BlockY reorderedSchedule)
    blockXSchedule <- rightOrFail (bindCudaSchedule columnOuter BlockX blockYSchedule)
    threadYSchedule <- rightOrFail (bindCudaSchedule rowInner ThreadY blockXSchedule)
    threadXSchedule <- rightOrFail (bindCudaSchedule columnInner ThreadX threadYSchedule)
    schedule <- rightOrFail (verifyCudaSchedule threadXSchedule)
    actual <- rightOrFail (generateCuda schedule)
    (cudaSourceText actual, cudaKernelName actual)
      `shouldBe` (expectedCudaElementwise, "cuda_add")

  it "rejects multiple operations for both backends" $ do
    (function, operation) <- rightOrFail twoOperationFunction
    cpuSchedule <- rightOrFail (newCpuSchedule function operation >>= verifyCpuSchedule)
    cudaSchedule <- rightOrFail (newCudaSchedule function operation defaultCudaTarget >>= verifyCudaSchedule)
    generateC cpuSchedule `shouldBe` Left (ExactlyOneOperationRequired 2)
    generateCuda cudaSchedule `shouldBe` Left (ExactlyOneOperationRequired 2)

  it "renders f32 literals exactly like Rust Debug" $ do
    actual <-
      mapM
        generateConstantC
        [ castWord32ToFloat 0x4c4fde72,
          castWord32ToFloat 0x00000001,
          1.0e-4,
          1.0e-5,
          1.0e15,
          1.0e16,
          castWord32ToFloat 0x7f7fffff,
          castWord32ToFloat 0x80000000,
          castWord32ToFloat 0x7fc00000,
          castWord32ToFloat 0x7f800000,
          castWord32ToFloat 0xff800000
        ]
    map cSourceText actual
      `shouldBe` map
        expectedConstantC
        [ "54491590.0f",
          "1e-45f",
          "0.0001f",
          "1e-5f",
          "1000000000000000.0f",
          "1e16f",
          "3.4028235e38f",
          "-0.0f",
          "NAN",
          "INFINITY",
          "-INFINITY"
        ]
    map cFunctionName actual `shouldBe` replicate 11 "constant_value"

rightOrFail :: Show error => Either error value -> IO value
rightOrFail result = case result of
  Right value -> pure value
  Left failure -> expectationFailure (show failure) >> fail "unexpected Left"

onlyOperation :: VerifiedComputeFunction -> ComputeOp
onlyOperation function = case functionOperations (verifiedFunction function) of
  [operation] -> operation
  _ -> error "fixture must contain exactly one operation"

iteratorAt :: Int -> ComputeOp -> Iterator
iteratorAt iteratorIndex operation = case drop iteratorIndex (computeIterators operation) of
  iterator : _ -> iterator
  [] -> error "fixture is missing an iterator"

generateConstantC :: Float -> IO CSource
generateConstantC value = do
  function <- rightOrFail $ buildComputeFunction "constant_value" $ do
    output <- compute "output" (constant (F32Literal value))
    markOutput output
  let operation = computeOpId (onlyOperation function)
  schedule <- rightOrFail (newCpuSchedule function operation >>= verifyCpuSchedule)
  rightOrFail (generateC schedule)

expectedConstantC :: String -> String
expectedConstantC literal =
  unlines
    [ "#include <math.h>",
      "#include <stdbool.h>",
      "#include <stddef.h>",
      "",
      "#pragma STDC FP_CONTRACT OFF",
      "",
      "void constant_value(float* output) {",
      "    float scalar0 = " ++ literal ++ ";",
      "    output[0u] = scalar0;",
      "}"
    ]

expectedCpuElementwise :: String
expectedCpuElementwise =
  unlines
    [ "#include <math.h>",
      "#include <stdbool.h>",
      "#include <stddef.h>",
      "",
      "#pragma STDC FP_CONTRACT OFF",
      "",
      "void choose_by_column(const float* input0, const float* input1, float* output, size_t symbol0) {",
      "    for (size_t loop0 = 0; loop0 < 2; ++loop0) {",
      "        for (size_t loop2 = 0; loop2 < ((symbol0) + 3u) / 4u; ++loop2) {",
      "            for (size_t loop3 = 0; loop3 < 4; ++loop3) {",
      "                if (((loop2 * 4u) + loop3) < symbol0) {",
      "                    float scalar0 = input0[((loop0) * symbol0 + ((loop2 * 4u) + loop3))];",
      "                    float scalar1 = input1[((loop0) * symbol0 + ((loop2 * 4u) + loop3))];",
      "                    size_t scalar2 = ((loop2 * 4u) + loop3);",
      "                    size_t scalar3 = 4u;",
      "                    bool scalar4 = (scalar2 >= scalar3);",
      "                    float scalar5 = (scalar4 ? scalar0 : scalar1);",
      "                    output[((loop0) * symbol0 + ((loop2 * 4u) + loop3))] = scalar5;",
      "                }",
      "            }",
      "        }",
      "    }",
      "}"
    ]

expectedCpuGemm :: String
expectedCpuGemm =
  unlines
    [ "#include <math.h>",
      "#include <stdbool.h>",
      "#include <stddef.h>",
      "",
      "#pragma STDC FP_CONTRACT OFF",
      "",
      "void gemm(const float* input0, const float* input1, float* output, size_t symbol0, size_t symbol1, size_t symbol2) {",
      "    for (size_t loop0 = 0; loop0 < symbol0; ++loop0) {",
      "        for (size_t loop1 = 0; loop1 < symbol1; ++loop1) {",
      "            float accumulator = 0.0f;",
      "            for (size_t loop2 = 0; loop2 < symbol2; ++loop2) {",
      "                float scalar0 = input0[((loop0) * symbol2 + loop2)];",
      "                float scalar1 = input1[((loop2) * symbol1 + loop1)];",
      "                float scalar3 = fmaf(scalar0, scalar1, accumulator);",
      "                accumulator = scalar3;",
      "            }",
      "            output[((loop0) * symbol1 + loop1)] = accumulator;",
      "        }",
      "    }",
      "}"
    ]

expectedCudaElementwise :: String
expectedCudaElementwise =
  unlines
    [ "#include <cuda_runtime.h>",
      "#include <math.h>",
      "#include <stdbool.h>",
      "#include <stddef.h>",
      "",
      "__global__ void cuda_add(const float* input0, const float* input1, float* output) {",
      "    if (((((size_t)blockIdx.x) * 4u) + ((size_t)threadIdx.x)) < 10 && ((((size_t)blockIdx.y) * 2u) + ((size_t)threadIdx.y)) < 5) {",
      "        float scalar0 = input0[((((((size_t)blockIdx.y) * 2u) + ((size_t)threadIdx.y))) * 10 + ((((size_t)blockIdx.x) * 4u) + ((size_t)threadIdx.x)))];",
      "        float scalar1 = input1[((((((size_t)blockIdx.y) * 2u) + ((size_t)threadIdx.y))) * 10 + ((((size_t)blockIdx.x) * 4u) + ((size_t)threadIdx.x)))];",
      "        float scalar2 = __fadd_rn(scalar0, scalar1);",
      "        output[((((((size_t)blockIdx.y) * 2u) + ((size_t)threadIdx.y))) * 10 + ((((size_t)blockIdx.x) * 4u) + ((size_t)threadIdx.x)))] = scalar2;",
      "    }",
      "}"
    ]
