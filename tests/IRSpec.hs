module IRSpec (spec) where

import Aoewif.IR
import Prelude hiding (compare)
import Test.Hspec hiding (parallel)

data FunctionView = FunctionView
  { viewName :: String
  , viewSymbols :: [Symbol]
  , viewTensors :: [TensorValue]
  , viewOperations :: [ComputeOp]
  , viewOutputs :: [TensorValueId]
  }
  deriving stock (Eq, Show)

spec :: Spec
spec = describe "compute IR" $ do
  it "builds elementwise add" $ do
    let (identifiers, verified) = mustSucceed elementwiseAdd
        (rows, columns, left, right, row, column, leftElement, rightElement, sumValue, output) = identifiers
        function = verifiedFunction verified
        operationId = computeOpId (onlyOperation verified)
        shape = [SymbolDim rows, SymbolDim columns]
        expectedOperation =
          ComputeOp
            operationId
            "sum"
            [ Iterator row "row" (SymbolDim rows) Parallel
            , Iterator column "column" (SymbolDim columns) Parallel
            ]
            [ TensorAccess left [IteratorIndex row, IteratorIndex column] leftElement
            , TensorAccess right [IteratorIndex row, IteratorIndex column] rightElement
            ]
            Nothing
            Strict
            ( ScalarRegion
                [ ScalarArgument leftElement F32 (InputElement 0)
                , ScalarArgument rightElement F32 (InputElement 1)
                ]
                [ScalarOperation sumValue F32 (AddOperation leftElement rightElement)]
                sumValue
            )
            output
        expected =
          FunctionView
            "elementwise_add"
            [Symbol rows "rows", Symbol columns "columns"]
            [ TensorValue left "left" (tensorTypeF32 shape) (InputTensor 0)
            , TensorValue right "right" (tensorTypeF32 shape) (InputTensor 1)
            , TensorValue output "sum" (tensorTypeF32 shape) (ComputeResult operationId)
            ]
            [expectedOperation]
            [output]
    functionView function `shouldBe` expected

  it "builds vector broadcast" $ do
    let (identifiers, verified) = mustSucceed vectorBroadcast
        (rows, columns, matrix, bias, row, column, matrixElement, biasElement, biased, output) = identifiers
        operationId = computeOpId (onlyOperation verified)
        matrixShape = [SymbolDim rows, SymbolDim columns]
        expected =
          FunctionView
            "broadcast_add"
            [Symbol rows "rows", Symbol columns "columns"]
            [ TensorValue matrix "matrix" (tensorTypeF32 matrixShape) (InputTensor 0)
            , TensorValue bias "bias" (tensorTypeF32 [SymbolDim columns]) (InputTensor 1)
            , TensorValue output "biased" (tensorTypeF32 matrixShape) (ComputeResult operationId)
            ]
            [ ComputeOp
                operationId
                "biased"
                [ Iterator row "row" (SymbolDim rows) Parallel
                , Iterator column "column" (SymbolDim columns) Parallel
                ]
                [ TensorAccess matrix [IteratorIndex row, IteratorIndex column] matrixElement
                , TensorAccess bias [IteratorIndex column] biasElement
                ]
                Nothing
                Strict
                ( ScalarRegion
                    [ ScalarArgument matrixElement F32 (InputElement 0)
                    , ScalarArgument biasElement F32 (InputElement 1)
                    ]
                    [ScalarOperation biased F32 (AddOperation matrixElement biasElement)]
                    biased
                )
                output
            ]
            [output]
    functionView (verifiedFunction verified) `shouldBe` expected

  it "builds transpose projection" $ do
    let (identifiers, verified) = mustSucceed transposeProjection
        (rows, columns, inputTensor, row, column, element, output) = identifiers
        operationId = computeOpId (onlyOperation verified)
        expected =
          FunctionView
            "transpose"
            [Symbol rows "rows", Symbol columns "columns"]
            [ TensorValue
                inputTensor
                "input"
                (tensorTypeF32 [SymbolDim columns, SymbolDim rows])
                (InputTensor 0)
            , TensorValue
                output
                "output"
                (tensorTypeF32 [SymbolDim rows, SymbolDim columns])
                (ComputeResult operationId)
            ]
            [ ComputeOp
                operationId
                "output"
                [ Iterator row "row" (SymbolDim rows) Parallel
                , Iterator column "column" (SymbolDim columns) Parallel
                ]
                [TensorAccess inputTensor [IteratorIndex column, IteratorIndex row] element]
                Nothing
                Strict
                (ScalarRegion [ScalarArgument element F32 (InputElement 0)] [] element)
                output
            ]
            [output]
    functionView (verifiedFunction verified) `shouldBe` expected

  it "builds a strict reduction" $ do
    let (identifiers, verified) = mustSucceed strictReduction
        (rows, columns, inputTensor, row, column, element, accumulator, sumValue, output) = identifiers
        operationId = computeOpId (onlyOperation verified)
        expected =
          FunctionView
            "row_sum"
            [Symbol rows "rows", Symbol columns "columns"]
            [ TensorValue
                inputTensor
                "input"
                (tensorTypeF32 [SymbolDim rows, SymbolDim columns])
                (InputTensor 0)
            , TensorValue output "output" (tensorTypeF32 [SymbolDim rows]) (ComputeResult operationId)
            ]
            [ ComputeOp
                operationId
                "output"
                [ Iterator row "row" (SymbolDim rows) Parallel
                , Iterator column "column" (SymbolDim columns) Reduction
                ]
                [TensorAccess inputTensor [IteratorIndex row, IteratorIndex column] element]
                (Just (F32Literal 0.0))
                Strict
                ( ScalarRegion
                    [ ScalarArgument element F32 (InputElement 0)
                    , ScalarArgument accumulator F32 Accumulator
                    ]
                    [ScalarOperation sumValue F32 (AddOperation accumulator element)]
                    sumValue
                )
                output
            ]
            [output]
    functionView (verifiedFunction verified) `shouldBe` expected
    hasReduction (onlyOperation verified) `shouldBe` True

  it "builds GEMM" $ do
    let (identifiers, verified) = mustSucceed gemm
        (rows, columns, reductionExtent, left, right, row, column, inner, leftElement, rightElement, accumulator, result, output) = identifiers
        operationId = computeOpId (onlyOperation verified)
        expectedOperation =
          ComputeOp
            operationId
            "output"
            [ Iterator row "row" (SymbolDim rows) Parallel
            , Iterator column "column" (SymbolDim columns) Parallel
            , Iterator inner "inner" (SymbolDim reductionExtent) Reduction
            ]
            [ TensorAccess left [IteratorIndex row, IteratorIndex inner] leftElement
            , TensorAccess right [IteratorIndex inner, IteratorIndex column] rightElement
            ]
            (Just (F32Literal 0.0))
            Strict
            ( ScalarRegion
                [ ScalarArgument leftElement F32 (InputElement 0)
                , ScalarArgument rightElement F32 (InputElement 1)
                , ScalarArgument accumulator F32 Accumulator
                ]
                [ScalarOperation result F32 (FmaOperation leftElement rightElement accumulator)]
                result
            )
            output
        expected =
          FunctionView
            "gemm"
            [ Symbol rows "rows"
            , Symbol columns "columns"
            , Symbol reductionExtent "reduction"
            ]
            [ TensorValue
                left
                "left"
                (tensorTypeF32 [SymbolDim rows, SymbolDim reductionExtent])
                (InputTensor 0)
            , TensorValue
                right
                "right"
                (tensorTypeF32 [SymbolDim reductionExtent, SymbolDim columns])
                (InputTensor 1)
            , TensorValue
                output
                "output"
                (tensorTypeF32 [SymbolDim rows, SymbolDim columns])
                (ComputeResult operationId)
            ]
            [expectedOperation]
            [output]
    functionView (verifiedFunction verified) `shouldBe` expected

  it "builds batched GEMM" $ do
    let (identifiers, verified) = mustSucceed batchedGemm
        (batches, rows, columns, reductionExtent, left, right, batch, row, column, inner, leftElement, rightElement, accumulator, result, output) = identifiers
        operationId = computeOpId (onlyOperation verified)
        expectedOperation =
          ComputeOp
            operationId
            "output"
            [ Iterator batch "batch" (SymbolDim batches) Parallel
            , Iterator row "row" (SymbolDim rows) Parallel
            , Iterator column "column" (SymbolDim columns) Parallel
            , Iterator inner "inner" (SymbolDim reductionExtent) Reduction
            ]
            [ TensorAccess left [IteratorIndex batch, IteratorIndex row, IteratorIndex inner] leftElement
            , TensorAccess right [IteratorIndex batch, IteratorIndex inner, IteratorIndex column] rightElement
            ]
            (Just (F32Literal 0.0))
            Strict
            ( ScalarRegion
                [ ScalarArgument leftElement F32 (InputElement 0)
                , ScalarArgument rightElement F32 (InputElement 1)
                , ScalarArgument accumulator F32 Accumulator
                ]
                [ScalarOperation result F32 (FmaOperation leftElement rightElement accumulator)]
                result
            )
            output
        expectedTensors =
          [ TensorValue
              left
              "left"
              (tensorTypeF32 [SymbolDim batches, SymbolDim rows, SymbolDim reductionExtent])
              (InputTensor 0)
          , TensorValue
              right
              "right"
              (tensorTypeF32 [SymbolDim batches, SymbolDim reductionExtent, SymbolDim columns])
              (InputTensor 1)
          , TensorValue
              output
              "output"
              (tensorTypeF32 [SymbolDim batches, SymbolDim rows, SymbolDim columns])
              (ComputeResult operationId)
          ]
        expected =
          FunctionView
            "batched_gemm"
            [ Symbol batches "batches"
            , Symbol rows "rows"
            , Symbol columns "columns"
            , Symbol reductionExtent "reduction"
            ]
            expectedTensors
            [expectedOperation]
            [output]
    functionView (verifiedFunction verified) `shouldBe` expected

  it "builds index, comparison, and selection operations" $ do
    let (identifiers, verified) = mustSucceed maskedValues
        (lengthSymbol, inputTensor, iteratorIdentifier, element, indexValue, limit, condition, zero, selected, output) = identifiers
        operationId = computeOpId (onlyOperation verified)
        expected =
          FunctionView
            "masked_values"
            [Symbol lengthSymbol "length"]
            [ TensorValue inputTensor "input" (tensorTypeF32 [SymbolDim lengthSymbol]) (InputTensor 0)
            , TensorValue output "output" (tensorTypeF32 [SymbolDim lengthSymbol]) (ComputeResult operationId)
            ]
            [ ComputeOp
                operationId
                "output"
                [Iterator iteratorIdentifier "index" (SymbolDim lengthSymbol) Parallel]
                [TensorAccess inputTensor [IteratorIndex iteratorIdentifier] element]
                Nothing
                Strict
                ( ScalarRegion
                    [ScalarArgument element F32 (InputElement 0)]
                    [ ScalarOperation indexValue Index (IndexOperation iteratorIdentifier)
                    , ScalarOperation limit Index (ConstantOperation (IndexLiteral 4))
                    , ScalarOperation condition Bool (CompareOperation Less indexValue limit)
                    , ScalarOperation zero F32 (ConstantOperation (F32Literal 0.0))
                    , ScalarOperation selected F32 (SelectOperation condition element zero)
                    ]
                    selected
                )
                output
            ]
            [output]
    functionView (verifiedFunction verified) `shouldBe` expected

  it "chains compute results through tensor SSA" $ do
    let (identifiers, verified) = mustSucceed chainedComputes
        (lengthSymbol, inputTensor, squareIterator, squareElement, squareResult, squared, shiftIterator, shiftElement, one, shiftResult, shifted) = identifiers
        function = verifiedFunction verified
        (squareOperationId, shiftOperationId) =
          exactlyTwo "operations" (map computeOpId (functionOperations function))
        expected =
          FunctionView
            "chain"
            [Symbol lengthSymbol "length"]
            [ TensorValue inputTensor "input" (tensorTypeF32 [SymbolDim lengthSymbol]) (InputTensor 0)
            , TensorValue squared "squared" (tensorTypeF32 [SymbolDim lengthSymbol]) (ComputeResult squareOperationId)
            , TensorValue shifted "shifted" (tensorTypeF32 [SymbolDim lengthSymbol]) (ComputeResult shiftOperationId)
            ]
            [ ComputeOp
                squareOperationId
                "squared"
                [Iterator squareIterator "index" (SymbolDim lengthSymbol) Parallel]
                [TensorAccess inputTensor [IteratorIndex squareIterator] squareElement]
                Nothing
                Strict
                ( ScalarRegion
                    [ScalarArgument squareElement F32 (InputElement 0)]
                    [ScalarOperation squareResult F32 (MulOperation squareElement squareElement)]
                    squareResult
                )
                squared
            , ComputeOp
                shiftOperationId
                "shifted"
                [Iterator shiftIterator "index" (SymbolDim lengthSymbol) Parallel]
                [TensorAccess squared [IteratorIndex shiftIterator] shiftElement]
                Nothing
                Strict
                ( ScalarRegion
                    [ScalarArgument shiftElement F32 (InputElement 0)]
                    [ ScalarOperation one F32 (ConstantOperation (F32Literal 1.0))
                    , ScalarOperation shiftResult F32 (AddOperation shiftElement one)
                    ]
                    shiftResult
                )
                shifted
            ]
            [shifted]
    functionView function `shouldBe` expected
    map tensorValueId (inputTensors function) `shouldBe` [inputTensor]

  it "rejects IDs owned by another function" $ do
    let foreignVerified = mustSucceed foreignFixture
        foreignFunction = verifiedFunction foreignVerified
        foreignSymbol = exactlyOne "symbol" (functionSymbols foreignFunction)
        foreignTensor = exactlyOne "input tensor" (inputTensors foreignFunction)
        foreignOperation = onlyOperation foreignVerified
        foreignIterator = exactlyOne "iterator" (computeIterators foreignOperation)
        foreignScalarOperation =
          exactlyOne "scalar operation" (scalarOperations (computeBody foreignOperation))
        foreignSymbolId = symbolId foreignSymbol
        foreignTensorId = tensorValueId foreignTensor
        foreignIteratorId = iteratorId foreignIterator
        foreignScalarId = scalarOperationResult foreignScalarOperation
    foreignSymbolInput foreignSymbolId `shouldBe` Left (ForeignId "symbol")
    foreignOutput foreignTensorId `shouldBe` Left (ForeignId "tensor")
    foreignTensorRead foreignTensorId `shouldBe` Left (ForeignId "tensor")
    foreignIteratorRead foreignIteratorId `shouldBe` Left (ForeignId "iterator")
    foreignIteratorIndex foreignIteratorId `shouldBe` Left (ForeignId "iterator")
    foreignScalarAdd foreignScalarId `shouldBe` Left (ForeignId "scalar")

  it "rejects access rank mismatch" $
    badRank `shouldBe` Left (TensorRankMismatch 2 1)

  it "rejects access dimension mismatch" $
    badDimension `shouldBe` Left DimensionMismatch

  it "rejects invalid constant indices" $ do
    dynamicConstantIndex `shouldBe` Left ConstantIndexRequiresStaticDimension
    outOfBoundsConstantIndex `shouldBe` Left (ConstantIndexOutOfBounds 4 4)

  it "rejects unsupported and mismatched scalar types" $ do
    unsupportedTensorType Bool `shouldBe` Left (TensorElementTypeUnsupported Bool)
    unsupportedTensorType Index `shouldBe` Left (TensorElementTypeUnsupported Index)
    addTypeMismatch `shouldBe` Left (ScalarTypeMismatch F32 Index)
    compareTypeMismatch `shouldBe` Left ScalarOperandsMustMatch
    selectConditionMismatch `shouldBe` Left (ScalarTypeMismatch Bool F32)
    selectValueMismatch `shouldBe` Left ScalarOperandsMustMatch
    nonF32Result `shouldBe` Left (TensorElementTypeUnsupported Bool)

  it "rejects invalid reduction contracts" $ do
    missingReductionInit `shouldBe` Left ReductionInitRequired
    initWithoutReduction `shouldBe` Left ReductionInitWithoutReduction
    duplicateReductionInit `shouldBe` Left DuplicateAccumulator
    reductionResultTypeMismatch `shouldBe` Left ScalarResultMustMatchInit

  it "rejects missing, duplicate, and foreign outputs" $ do
    missingOutput `shouldBe` Left OutputRequired
    duplicateOutput `shouldBe` Left (OutputAlreadyMarked 0)
    let foreignFunctionValue = mustSucceed foreignFixture
        foreignOutputTensor =
          exactlyOne "output" (functionOutputs (verifiedFunction foreignFunctionValue))
    markForeignOutput foreignOutputTensor `shouldBe` Left (ForeignId "tensor")

elementwiseAdd
  :: Either
      IrError
      ( ( SymbolId
        , SymbolId
        , TensorValueId
        , TensorValueId
        , IteratorId
        , IteratorId
        , ScalarValueId
        , ScalarValueId
        , ScalarValueId
        , TensorValueId
        )
      , VerifiedComputeFunction
      )
elementwiseAdd = buildComputeFunctionWith "elementwise_add" $ do
  rows <- symbol "rows"
  columns <- symbol "columns"
  let shape = [SymbolDim rows, SymbolDim columns]
  left <- input "left" (tensorTypeF32 shape)
  right <- input "right" (tensorTypeF32 shape)
  ((row, column, leftElement, rightElement, sumValue), output) <- computeWith "sum" $ do
    row <- parallel "row" (SymbolDim rows)
    column <- parallel "column" (SymbolDim columns)
    leftElement <- readTensor left [IteratorIndex row, IteratorIndex column]
    rightElement <- readTensor right [IteratorIndex row, IteratorIndex column]
    sumValue <- add leftElement rightElement
    pure ((row, column, leftElement, rightElement, sumValue), sumValue)
  markOutput output
  pure (rows, columns, left, right, row, column, leftElement, rightElement, sumValue, output)

vectorBroadcast
  :: Either
      IrError
      ( ( SymbolId
        , SymbolId
        , TensorValueId
        , TensorValueId
        , IteratorId
        , IteratorId
        , ScalarValueId
        , ScalarValueId
        , ScalarValueId
        , TensorValueId
        )
      , VerifiedComputeFunction
      )
vectorBroadcast = buildComputeFunctionWith "broadcast_add" $ do
  rows <- symbol "rows"
  columns <- symbol "columns"
  matrix <- input "matrix" (tensorTypeF32 [SymbolDim rows, SymbolDim columns])
  bias <- input "bias" (tensorTypeF32 [SymbolDim columns])
  ((row, column, matrixElement, biasElement, biased), output) <- computeWith "biased" $ do
    row <- parallel "row" (SymbolDim rows)
    column <- parallel "column" (SymbolDim columns)
    matrixElement <- readTensor matrix [IteratorIndex row, IteratorIndex column]
    biasElement <- readTensor bias [IteratorIndex column]
    biased <- add matrixElement biasElement
    pure ((row, column, matrixElement, biasElement, biased), biased)
  markOutput output
  pure (rows, columns, matrix, bias, row, column, matrixElement, biasElement, biased, output)

transposeProjection
  :: Either
      IrError
      ( (SymbolId, SymbolId, TensorValueId, IteratorId, IteratorId, ScalarValueId, TensorValueId)
      , VerifiedComputeFunction
      )
transposeProjection = buildComputeFunctionWith "transpose" $ do
  rows <- symbol "rows"
  columns <- symbol "columns"
  inputTensor <- input "input" (tensorTypeF32 [SymbolDim columns, SymbolDim rows])
  ((row, column, element), output) <- computeWith "output" $ do
    row <- parallel "row" (SymbolDim rows)
    column <- parallel "column" (SymbolDim columns)
    element <- readTensor inputTensor [IteratorIndex column, IteratorIndex row]
    pure ((row, column, element), element)
  markOutput output
  pure (rows, columns, inputTensor, row, column, element, output)

strictReduction
  :: Either
      IrError
      ( ( SymbolId
        , SymbolId
        , TensorValueId
        , IteratorId
        , IteratorId
        , ScalarValueId
        , ScalarValueId
        , ScalarValueId
        , TensorValueId
        )
      , VerifiedComputeFunction
      )
strictReduction = buildComputeFunctionWith "row_sum" $ do
  rows <- symbol "rows"
  columns <- symbol "columns"
  inputTensor <- input "input" (tensorTypeF32 [SymbolDim rows, SymbolDim columns])
  ((row, column, element, accumulator, sumValue), output) <- computeWith "output" $ do
    row <- parallel "row" (SymbolDim rows)
    column <- reduction "column" (SymbolDim columns)
    element <- readTensor inputTensor [IteratorIndex row, IteratorIndex column]
    accumulator <- reductionInit (F32Literal 0.0)
    sumValue <- add accumulator element
    pure ((row, column, element, accumulator, sumValue), sumValue)
  markOutput output
  pure (rows, columns, inputTensor, row, column, element, accumulator, sumValue, output)

gemm
  :: Either
      IrError
      ( ( SymbolId
        , SymbolId
        , SymbolId
        , TensorValueId
        , TensorValueId
        , IteratorId
        , IteratorId
        , IteratorId
        , ScalarValueId
        , ScalarValueId
        , ScalarValueId
        , ScalarValueId
        , TensorValueId
        )
      , VerifiedComputeFunction
      )
gemm = buildComputeFunctionWith "gemm" $ do
  rows <- symbol "rows"
  columns <- symbol "columns"
  reductionExtent <- symbol "reduction"
  left <- input "left" (tensorTypeF32 [SymbolDim rows, SymbolDim reductionExtent])
  right <- input "right" (tensorTypeF32 [SymbolDim reductionExtent, SymbolDim columns])
  ((row, column, inner, leftElement, rightElement, accumulator, result), output) <- computeWith "output" $ do
    row <- parallel "row" (SymbolDim rows)
    column <- parallel "column" (SymbolDim columns)
    inner <- reduction "inner" (SymbolDim reductionExtent)
    leftElement <- readTensor left [IteratorIndex row, IteratorIndex inner]
    rightElement <- readTensor right [IteratorIndex inner, IteratorIndex column]
    accumulator <- reductionInit (F32Literal 0.0)
    result <- fma leftElement rightElement accumulator
    pure ((row, column, inner, leftElement, rightElement, accumulator, result), result)
  markOutput output
  pure
    ( rows
    , columns
    , reductionExtent
    , left
    , right
    , row
    , column
    , inner
    , leftElement
    , rightElement
    , accumulator
    , result
    , output
    )

batchedGemm
  :: Either
      IrError
      ( ( SymbolId
        , SymbolId
        , SymbolId
        , SymbolId
        , TensorValueId
        , TensorValueId
        , IteratorId
        , IteratorId
        , IteratorId
        , IteratorId
        , ScalarValueId
        , ScalarValueId
        , ScalarValueId
        , ScalarValueId
        , TensorValueId
        )
      , VerifiedComputeFunction
      )
batchedGemm = buildComputeFunctionWith "batched_gemm" $ do
  batches <- symbol "batches"
  rows <- symbol "rows"
  columns <- symbol "columns"
  reductionExtent <- symbol "reduction"
  left <- input "left" (tensorTypeF32 [SymbolDim batches, SymbolDim rows, SymbolDim reductionExtent])
  right <- input "right" (tensorTypeF32 [SymbolDim batches, SymbolDim reductionExtent, SymbolDim columns])
  ((batch, row, column, inner, leftElement, rightElement, accumulator, result), output) <- computeWith "output" $ do
    batch <- parallel "batch" (SymbolDim batches)
    row <- parallel "row" (SymbolDim rows)
    column <- parallel "column" (SymbolDim columns)
    inner <- reduction "inner" (SymbolDim reductionExtent)
    leftElement <- readTensor left [IteratorIndex batch, IteratorIndex row, IteratorIndex inner]
    rightElement <- readTensor right [IteratorIndex batch, IteratorIndex inner, IteratorIndex column]
    accumulator <- reductionInit (F32Literal 0.0)
    result <- fma leftElement rightElement accumulator
    pure ((batch, row, column, inner, leftElement, rightElement, accumulator, result), result)
  markOutput output
  pure
    ( batches
    , rows
    , columns
    , reductionExtent
    , left
    , right
    , batch
    , row
    , column
    , inner
    , leftElement
    , rightElement
    , accumulator
    , result
    , output
    )

maskedValues
  :: Either
      IrError
      ( ( SymbolId
        , TensorValueId
        , IteratorId
        , ScalarValueId
        , ScalarValueId
        , ScalarValueId
        , ScalarValueId
        , ScalarValueId
        , ScalarValueId
        , TensorValueId
        )
      , VerifiedComputeFunction
      )
maskedValues = buildComputeFunctionWith "masked_values" $ do
  lengthSymbol <- symbol "length"
  inputTensor <- input "input" (tensorTypeF32 [SymbolDim lengthSymbol])
  ((iteratorIdentifier, element, indexValue, limit, condition, zero, selected), output) <- computeWith "output" $ do
    iteratorIdentifier <- parallel "index" (SymbolDim lengthSymbol)
    element <- readTensor inputTensor [IteratorIndex iteratorIdentifier]
    indexValue <- index iteratorIdentifier
    limit <- constant (IndexLiteral 4)
    condition <- compare Less indexValue limit
    zero <- constant (F32Literal 0.0)
    selected <- selectScalar condition element zero
    pure ((iteratorIdentifier, element, indexValue, limit, condition, zero, selected), selected)
  markOutput output
  pure
    ( lengthSymbol
    , inputTensor
    , iteratorIdentifier
    , element
    , indexValue
    , limit
    , condition
    , zero
    , selected
    , output
    )

chainedComputes
  :: Either
      IrError
      ( ( SymbolId
        , TensorValueId
        , IteratorId
        , ScalarValueId
        , ScalarValueId
        , TensorValueId
        , IteratorId
        , ScalarValueId
        , ScalarValueId
        , ScalarValueId
        , TensorValueId
        )
      , VerifiedComputeFunction
      )
chainedComputes = buildComputeFunctionWith "chain" $ do
  lengthSymbol <- symbol "length"
  inputTensor <- input "input" (tensorTypeF32 [SymbolDim lengthSymbol])
  ((squareIterator, squareElement, squareResult), squared) <- computeWith "squared" $ do
    squareIterator <- parallel "index" (SymbolDim lengthSymbol)
    squareElement <- readTensor inputTensor [IteratorIndex squareIterator]
    squareResult <- mul squareElement squareElement
    pure ((squareIterator, squareElement, squareResult), squareResult)
  ((shiftIterator, shiftElement, one, shiftResult), shifted) <- computeWith "shifted" $ do
    shiftIterator <- parallel "index" (SymbolDim lengthSymbol)
    shiftElement <- readTensor squared [IteratorIndex shiftIterator]
    one <- constant (F32Literal 1.0)
    shiftResult <- add shiftElement one
    pure ((shiftIterator, shiftElement, one, shiftResult), shiftResult)
  markOutput shifted
  pure
    ( lengthSymbol
    , inputTensor
    , squareIterator
    , squareElement
    , squareResult
    , squared
    , shiftIterator
    , shiftElement
    , one
    , shiftResult
    , shifted
    )

foreignFixture :: Either IrError VerifiedComputeFunction
foreignFixture = buildComputeFunction "foreign" $ do
  _ <- symbol "foreign_extent"
  _ <- input "foreign_input" (tensorTypeF32 [StaticDim 4])
  output <- compute "foreign_operation" $ do
    _ <- parallel "index" (StaticDim 4)
    constant (F32Literal 1.0)
  markOutput output

foreignSymbolInput :: SymbolId -> Either IrError VerifiedComputeFunction
foreignSymbolInput foreignSymbol = buildComputeFunction "local_symbol" $ do
  output <- input "foreign_shaped" (tensorTypeF32 [SymbolDim foreignSymbol])
  markOutput output

foreignOutput :: TensorValueId -> Either IrError VerifiedComputeFunction
foreignOutput foreignTensor = buildComputeFunction "local_output" (markOutput foreignTensor)

foreignTensorRead :: TensorValueId -> Either IrError VerifiedComputeFunction
foreignTensorRead foreignTensor = buildComputeFunction "local_tensor_read" $ do
  localTensor <- input "local_input" (tensorTypeF32 [StaticDim 4])
  output <- compute "local_operation" $ do
    localIterator <- parallel "index" (StaticDim 4)
    _ <- readTensor localTensor [IteratorIndex localIterator]
    readTensor foreignTensor [IteratorIndex localIterator]
  markOutput output

foreignIteratorRead :: IteratorId -> Either IrError VerifiedComputeFunction
foreignIteratorRead foreignIterator = buildComputeFunction "local_iterator_read" $ do
  localTensor <- input "local_input" (tensorTypeF32 [StaticDim 4])
  output <- compute "local_operation" $
    readTensor localTensor [IteratorIndex foreignIterator]
  markOutput output

foreignIteratorIndex :: IteratorId -> Either IrError VerifiedComputeFunction
foreignIteratorIndex foreignIterator = buildComputeFunction "local_iterator_index" $ do
  output <- compute "local_operation" $ do
    _ <- parallel "index" (StaticDim 4)
    index foreignIterator
  markOutput output

foreignScalarAdd :: ScalarValueId -> Either IrError VerifiedComputeFunction
foreignScalarAdd foreignScalar = buildComputeFunction "local_scalar_add" $ do
  output <- compute "local_operation" $ do
    _ <- parallel "index" (StaticDim 4)
    localScalar <- constant (F32Literal 2.0)
    add localScalar foreignScalar
  markOutput output

badRank :: Either IrError VerifiedComputeFunction
badRank = buildComputeFunction "bad_rank" $ do
  inputTensor <- input "input" (tensorTypeF32 [StaticDim 4, StaticDim 8])
  output <- compute "output" $ do
    row <- parallel "row" (StaticDim 4)
    readTensor inputTensor [IteratorIndex row]
  markOutput output

badDimension :: Either IrError VerifiedComputeFunction
badDimension = buildComputeFunction "bad_dimension" $ do
  inputTensor <- input "input" (tensorTypeF32 [StaticDim 4])
  output <- compute "output" $ do
    iteratorIdentifier <- parallel "index" (StaticDim 8)
    readTensor inputTensor [IteratorIndex iteratorIdentifier]
  markOutput output

dynamicConstantIndex :: Either IrError VerifiedComputeFunction
dynamicConstantIndex = buildComputeFunction "dynamic_constant" $ do
  lengthSymbol <- symbol "length"
  inputTensor <- input "input" (tensorTypeF32 [SymbolDim lengthSymbol])
  output <- compute "output" $ readTensor inputTensor [ConstantIndex 0]
  markOutput output

outOfBoundsConstantIndex :: Either IrError VerifiedComputeFunction
outOfBoundsConstantIndex = buildComputeFunction "static_constant" $ do
  inputTensor <- input "input" (tensorTypeF32 [StaticDim 4])
  output <- compute "output" $ readTensor inputTensor [ConstantIndex 4]
  markOutput output

unsupportedTensorType :: ScalarType -> Either IrError VerifiedComputeFunction
unsupportedTensorType scalarType = buildComputeFunction "unsupported_tensor" $ do
  output <- input "values" (TensorType [StaticDim 4] scalarType)
  markOutput output

addTypeMismatch :: Either IrError VerifiedComputeFunction
addTypeMismatch = buildComputeFunction "add_type_mismatch" $ do
  inputTensor <- input "input" (tensorTypeF32 [StaticDim 4])
  output <- compute "output" $ do
    iteratorIdentifier <- parallel "index" (StaticDim 4)
    element <- readTensor inputTensor [IteratorIndex iteratorIdentifier]
    indexValue <- index iteratorIdentifier
    add indexValue element
  markOutput output

compareTypeMismatch :: Either IrError VerifiedComputeFunction
compareTypeMismatch = buildComputeFunction "compare_type_mismatch" $ do
  inputTensor <- input "input" (tensorTypeF32 [StaticDim 4])
  output <- compute "output" $ do
    iteratorIdentifier <- parallel "index" (StaticDim 4)
    element <- readTensor inputTensor [IteratorIndex iteratorIdentifier]
    indexValue <- index iteratorIdentifier
    compare Equal indexValue element
  markOutput output

selectConditionMismatch :: Either IrError VerifiedComputeFunction
selectConditionMismatch = buildComputeFunction "select_condition_mismatch" $ do
  inputTensor <- input "input" (tensorTypeF32 [StaticDim 4])
  output <- compute "output" $ do
    iteratorIdentifier <- parallel "index" (StaticDim 4)
    element <- readTensor inputTensor [IteratorIndex iteratorIdentifier]
    selectScalar element element element
  markOutput output

selectValueMismatch :: Either IrError VerifiedComputeFunction
selectValueMismatch = buildComputeFunction "select_value_mismatch" $ do
  inputTensor <- input "input" (tensorTypeF32 [StaticDim 4])
  output <- compute "output" $ do
    iteratorIdentifier <- parallel "index" (StaticDim 4)
    element <- readTensor inputTensor [IteratorIndex iteratorIdentifier]
    indexValue <- index iteratorIdentifier
    otherIndex <- constant (IndexLiteral 0)
    condition <- compare Equal indexValue otherIndex
    selectScalar condition element indexValue
  markOutput output

nonF32Result :: Either IrError VerifiedComputeFunction
nonF32Result = buildComputeFunction "non_f32_result" $ do
  output <- compute "output" $ do
    iteratorIdentifier <- parallel "index" (StaticDim 4)
    indexValue <- index iteratorIdentifier
    limit <- constant (IndexLiteral 4)
    compare Less indexValue limit
  markOutput output

missingReductionInit :: Either IrError VerifiedComputeFunction
missingReductionInit = buildComputeFunction "missing_reduction_init" $ do
  output <- compute "output" $ do
    _ <- reduction "index" (StaticDim 4)
    constant (F32Literal 1.0)
  markOutput output

initWithoutReduction :: Either IrError VerifiedComputeFunction
initWithoutReduction = buildComputeFunction "init_without_reduction" $ do
  output <- compute "output" $ do
    _ <- parallel "index" (StaticDim 4)
    reductionInit (F32Literal 0.0)
  markOutput output

duplicateReductionInit :: Either IrError VerifiedComputeFunction
duplicateReductionInit = buildComputeFunction "duplicate_reduction_init" $ do
  output <- compute "output" $ do
    _ <- reduction "index" (StaticDim 4)
    _ <- reductionInit (F32Literal 0.0)
    reductionInit (F32Literal 1.0)
  markOutput output

reductionResultTypeMismatch :: Either IrError VerifiedComputeFunction
reductionResultTypeMismatch = buildComputeFunction "reduction_type_mismatch" $ do
  output <- compute "output" $ do
    _ <- reduction "index" (StaticDim 4)
    _ <- reductionInit (IndexLiteral 0)
    constant (F32Literal 0.0)
  markOutput output

missingOutput :: Either IrError VerifiedComputeFunction
missingOutput = buildComputeFunction "missing_output" $ do
  inputTensor <- input "input" (tensorTypeF32 [StaticDim 4])
  _ <- compute "computed" $ do
    iteratorIdentifier <- parallel "index" (StaticDim 4)
    readTensor inputTensor [IteratorIndex iteratorIdentifier]
  pure ()

duplicateOutput :: Either IrError VerifiedComputeFunction
duplicateOutput = buildComputeFunction "duplicate_output" $ do
  output <- input "output" (tensorTypeF32 [StaticDim 4])
  markOutput output
  markOutput output

markForeignOutput :: TensorValueId -> Either IrError VerifiedComputeFunction
markForeignOutput foreignOutputTensor = buildComputeFunction "foreign_output" $ do
  localOutput <- input "output" (tensorTypeF32 [StaticDim 4])
  markOutput localOutput
  markOutput foreignOutputTensor

functionView :: ComputeFunction -> FunctionView
functionView function =
  FunctionView
    (functionName function)
    (functionSymbols function)
    (functionTensors function)
    (functionOperations function)
    (functionOutputs function)

onlyOperation :: VerifiedComputeFunction -> ComputeOp
onlyOperation = exactlyOne "operation" . functionOperations . verifiedFunction

exactlyOne :: String -> [value] -> value
exactlyOne _ [value] = value
exactlyOne name values = error ("expected one " <> name <> ", got " <> show (length values))

exactlyTwo :: String -> [value] -> (value, value)
exactlyTwo _ [first, second] = (first, second)
exactlyTwo name values = error ("expected two " <> name <> ", got " <> show (length values))

mustSucceed :: Show error => Either error value -> value
mustSucceed = either (error . show) id
