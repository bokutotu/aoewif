module Aoewif.Codegen
  ( CSource,
    cSourceText,
    cFunctionName,
    CudaSource,
    cudaSourceText,
    cudaKernelName,
    CodegenError (..),
    displayCodegenError,
    generateC,
    generateCuda,
  )
where

import qualified Aoewif.IR as IR
import qualified Aoewif.Schedule as Schedule
import Control.Exception (Exception (displayException))
import Control.Monad (foldM)
import Data.List (nub, sort)
import Data.Ratio ((%), denominator, numerator)
import Data.Word (Word32)
import GHC.Float (castFloatToWord32, castWord32ToFloat)

data CSource = CSource String String
  deriving (Eq, Show)

data CudaSource = CudaSource String String
  deriving (Eq, Show)

cSourceText :: CSource -> String
cSourceText (CSource sourceText _) = sourceText

cFunctionName :: CSource -> String
cFunctionName (CSource _ functionName) = functionName

cudaSourceText :: CudaSource -> String
cudaSourceText (CudaSource sourceText _) = sourceText

cudaKernelName :: CudaSource -> String
cudaKernelName (CudaSource _ kernelName) = kernelName

data CodegenError
  = ExactlyOneOperationRequired Int
  | ExactlyOneOutputRequired Int
  | ScheduledOperationIsNotOutput
  | ScheduledOperationMismatch
  | NonInputTensorAccess Int
  | UnsupportedTensorElement
  | UnsupportedScalarType
  | MissingScalarValue Int
  | MissingLogicalIndex Int
  | InvalidLoopPlan
  deriving (Eq, Show)

displayCodegenError :: CodegenError -> String
displayCodegenError codegenError = case codegenError of
  ExactlyOneOperationRequired actual ->
    "code generation requires exactly one compute operation, got " ++ show actual
  ExactlyOneOutputRequired actual ->
    "code generation requires exactly one output, got " ++ show actual
  ScheduledOperationIsNotOutput ->
    "the scheduled operation result is not the function output"
  ScheduledOperationMismatch ->
    "the schedule does not refer to the function operation"
  NonInputTensorAccess tensorIndex ->
    "tensor value " ++ show tensorIndex ++ " is not a function input"
  UnsupportedTensorElement -> "only f32 tensors are supported"
  UnsupportedScalarType -> "unsupported scalar type"
  MissingScalarValue scalarIndex -> "missing scalar value " ++ show scalarIndex
  MissingLogicalIndex iteratorIndex ->
    "missing logical index for iterator " ++ show iteratorIndex
  InvalidLoopPlan -> "invalid scheduled loop plan"

instance Exception CodegenError where
  displayException = displayCodegenError

generateC :: Schedule.VerifiedCpuSchedule -> Either CodegenError CSource
generateC schedule = do
  let verifiedFunction = Schedule.verifiedCpuFunction schedule
      operation = Schedule.verifiedCpuOperation schedule
      plan = Schedule.verifiedCpuPlan schedule
  generated <- generateSource CBackend verifiedFunction operation plan
  pure (CSource (generatedText generated) (generatedName generated))

generateCuda :: Schedule.VerifiedCudaSchedule -> Either CodegenError CudaSource
generateCuda schedule = do
  let verifiedFunction = Schedule.verifiedCudaFunction schedule
      operation = Schedule.verifiedCudaOperation schedule
      plan = Schedule.verifiedCudaPlan schedule
  generated <- generateSource CudaBackend verifiedFunction operation plan
  pure (CudaSource (generatedText generated) (generatedName generated))

data Backend = CBackend | CudaBackend
  deriving (Eq)

data GeneratedSource = GeneratedSource
  { generatedText :: String,
    generatedName :: String
  }

data SourceNode
  = SourceLine String
  | BlankLine
  | SourceScope String [SourceNode] String

generateSource :: Backend -> IR.VerifiedComputeFunction -> IR.ComputeOp -> Schedule.LoopPlan -> Either CodegenError GeneratedSource
generateSource backend verified operation plan = do
  let function = IR.verifiedFunction verified
  validateCodegenFunction function operation
  loopValue <- loopValueFunction backend plan
  let parallelLoops = filter ((== IR.Parallel) . Schedule.loopKind) (Schedule.planLoops plan)
      reductionLoops = filter ((== IR.Reduction) . Schedule.loopKind) (Schedule.planLoops plan)
      emittedParallelLoops = case backend of
        CBackend -> parallelLoops
        CudaBackend -> filter ((== Nothing) . Schedule.loopBinding) parallelLoops
  tailPredicate <- tailCondition plan loopValue
  (scalarNodes, scalarResult) <-
    emitScalarBody backend function operation plan loopValue
  outputIndex <- outputIndexExpression function operation plan loopValue
  resultNodes <- case IR.computeInit operation of
    Just initialValue -> do
      reductionNodes <-
        wrapLoops
          reductionLoops
          loopValue
          (scalarNodes ++ [SourceLine ("accumulator = " ++ scalarResult ++ ";")])
      pure $
        [SourceLine ("float accumulator = " ++ scalarLiteral initialValue ++ ";")]
          ++ reductionNodes
          ++ [SourceLine ("output[" ++ outputIndex ++ "] = accumulator;")]
    Nothing ->
      pure (scalarNodes ++ [SourceLine ("output[" ++ outputIndex ++ "] = " ++ scalarResult ++ ";")])
  let guardedNodes = maybe resultNodes (\condition -> [SourceScope ("if (" ++ condition ++ ") {") resultNodes "}"]) tailPredicate
  functionBody <- wrapLoops emittedParallelLoops loopValue guardedNodes
  let sourceNodes =
        backendPreamble backend
          ++ [ SourceScope
                 (functionDeclaration backend function ++ " {")
                 functionBody
                 "}"
             ]
  pure
    GeneratedSource
      { generatedText = renderSource sourceNodes,
        generatedName = IR.functionName function
      }

backendPreamble :: Backend -> [SourceNode]
backendPreamble backend = case backend of
  CBackend ->
    [ SourceLine "#include <math.h>",
      SourceLine "#include <stdbool.h>",
      SourceLine "#include <stddef.h>",
      BlankLine,
      SourceLine "#pragma STDC FP_CONTRACT OFF",
      BlankLine
    ]
  CudaBackend ->
    [ SourceLine "#include <cuda_runtime.h>",
      SourceLine "#include <math.h>",
      SourceLine "#include <stdbool.h>",
      SourceLine "#include <stddef.h>",
      BlankLine
    ]

functionDeclaration :: Backend -> IR.ComputeFunction -> String
functionDeclaration backend function =
  prefix ++ IR.functionName function ++ "(" ++ commaSeparated parameters ++ ")"
  where
    prefix = case backend of
      CBackend -> "void "
      CudaBackend -> "__global__ void "
    inputParameters = map inputParameter (IR.inputTensors function)
    parameters =
      inputParameters
        ++ ["float* output"]
        ++ map symbolParameter (IR.functionSymbols function)
    inputParameter tensor = case IR.tensorDefinition tensor of
      IR.InputTensor inputIndex -> "const float* input" ++ show inputIndex
      IR.ComputeResult _ -> error "inputTensors returned a compute result"
    symbolParameter symbol =
      "size_t symbol" ++ show (IR.symbolIdIndex (IR.symbolId symbol))

loopValueFunction :: Backend -> Schedule.LoopPlan -> Either CodegenError (Schedule.LoopId -> Either CodegenError String)
loopValueFunction backend plan =
  pure $ \loopId -> case Schedule.lookupLoopAxis loopId plan of
    Nothing -> Left InvalidLoopPlan
    Just loopAxis -> case (backend, Schedule.loopBinding loopAxis) of
      (CudaBackend, Just binding) -> Right (cudaBindingExpression binding)
      _ -> Right ("loop" ++ show (Schedule.loopIdIndex loopId))

wrapLoops :: [Schedule.LoopAxis] -> (Schedule.LoopId -> Either CodegenError String) -> [SourceNode] -> Either CodegenError [SourceNode]
wrapLoops loopAxes loopValue body = do
  loopNames <- mapM (loopValue . Schedule.loopAxisId) loopAxes
  pure (foldr wrap body (zip loopAxes loopNames))
  where
    wrap (loopAxis, loopName) inner =
      [ SourceScope
          ( "for (size_t "
              ++ loopName
              ++ " = 0; "
              ++ loopName
              ++ " < "
              ++ loopExtentExpression (Schedule.loopExtent loopAxis)
              ++ "; ++"
              ++ loopName
              ++ ") {"
          )
          inner
          "}"
      ]

emitScalarBody :: Backend -> IR.ComputeFunction -> IR.ComputeOp -> Schedule.LoopPlan -> (Schedule.LoopId -> Either CodegenError String) -> Either CodegenError ([SourceNode], String)
emitScalarBody backend function operation plan loopValue = do
  (argumentNodes, argumentValues) <-
    foldM emitArgument ([], []) (IR.scalarArguments (IR.computeBody operation))
  (operationNodes, values) <-
    foldM emitOperation ([], argumentValues) (IR.scalarOperations (IR.computeBody operation))
  result <- scalarValue (IR.scalarResult (IR.computeBody operation)) values
  pure (argumentNodes ++ operationNodes, result)
  where
    emitArgument (nodes, values) argument = case IR.scalarArgumentKind argument of
      IR.InputElement accessIndex -> do
        let access = IR.computeAccesses operation !! accessIndex
            tensorId = IR.accessTensor access
        tensor <- maybe (Left (NonInputTensorAccess (IR.tensorValueIdIndex tensorId))) Right (IR.lookupTensor tensorId function)
        indices <- mapM (accessIndexExpression plan loopValue) (IR.accessIndices access)
        name <- pure (scalarName (IR.scalarArgumentValue argument))
        input <- inputName function tensorId
        let line =
              "float "
                ++ name
                ++ " = "
                ++ input
                ++ "["
                ++ flattenedIndex (IR.tensorValueType tensor) indices
                ++ "];"
        pure (nodes ++ [SourceLine line], (IR.scalarArgumentValue argument, name) : values)
      IR.Accumulator ->
        pure (nodes, (IR.scalarArgumentValue argument, "accumulator") : values)

    emitOperation (nodes, values) scalarOperation = do
      expression <-
        scalarOperationExpression
          backend
          plan
          loopValue
          values
          (IR.scalarOperationKind scalarOperation)
      let name = scalarName (IR.scalarOperationResult scalarOperation)
          line =
            scalarTypeName (IR.scalarOperationResultType scalarOperation)
              ++ " "
              ++ name
              ++ " = "
              ++ expression
              ++ ";"
      pure (nodes ++ [SourceLine line], (IR.scalarOperationResult scalarOperation, name) : values)

scalarOperationExpression :: Backend -> Schedule.LoopPlan -> (Schedule.LoopId -> Either CodegenError String) -> [(IR.ScalarValueId, String)] -> IR.ScalarOperationKind -> Either CodegenError String
scalarOperationExpression backend plan loopValue values operationKind = case operationKind of
  IR.ConstantOperation literal -> Right (scalarLiteral literal)
  IR.IndexOperation iterator -> logicalIndexExpression plan iterator loopValue
  IR.AddOperation lhs rhs -> binary " + " "__fadd_rn" lhs rhs
  IR.SubOperation lhs rhs -> binary " - " "__fsub_rn" lhs rhs
  IR.MulOperation lhs rhs -> binary " * " "__fmul_rn" lhs rhs
  IR.DivOperation lhs rhs -> binary " / " "__fdiv_rn" lhs rhs
  IR.FmaOperation lhs rhs accumulator -> do
    lhsValue <- value lhs
    rhsValue <- value rhs
    accumulatorValue <- value accumulator
    let function = case backend of
          CBackend -> "fmaf"
          CudaBackend -> "__fmaf_rn"
    pure (function ++ "(" ++ commaSeparated [lhsValue, rhsValue, accumulatorValue] ++ ")")
  IR.MinOperation lhs rhs -> function2 "fminf" lhs rhs
  IR.MaxOperation lhs rhs -> function2 "fmaxf" lhs rhs
  IR.ExpOperation input -> function1 "expf" input
  IR.LogOperation input -> function1 "logf" input
  IR.CompareOperation predicate lhs rhs -> do
    lhsValue <- value lhs
    rhsValue <- value rhs
    pure ("(" ++ lhsValue ++ " " ++ compareOperator predicate ++ " " ++ rhsValue ++ ")")
  IR.SelectOperation condition trueValue falseValue -> do
    conditionExpression <- value condition
    trueExpression <- value trueValue
    falseExpression <- value falseValue
    pure ("(" ++ conditionExpression ++ " ? " ++ trueExpression ++ " : " ++ falseExpression ++ ")")
  where
    value scalarId = scalarValue scalarId values
    binary operator intrinsic lhs rhs = do
      lhsValue <- value lhs
      rhsValue <- value rhs
      pure $ case backend of
        CBackend -> "(" ++ lhsValue ++ operator ++ rhsValue ++ ")"
        CudaBackend -> intrinsic ++ "(" ++ commaSeparated [lhsValue, rhsValue] ++ ")"
    function1 function input = do
      inputValue <- value input
      pure (function ++ "(" ++ inputValue ++ ")")
    function2 function lhs rhs = do
      lhsValue <- value lhs
      rhsValue <- value rhs
      pure (function ++ "(" ++ commaSeparated [lhsValue, rhsValue] ++ ")")

scalarValue :: IR.ScalarValueId -> [(IR.ScalarValueId, String)] -> Either CodegenError String
scalarValue scalarId values =
  maybe
    (Left (MissingScalarValue (IR.scalarValueIdIndex scalarId)))
    Right
    (lookup scalarId values)

accessIndexExpression :: Schedule.LoopPlan -> (Schedule.LoopId -> Either CodegenError String) -> IR.IndexExpr -> Either CodegenError String
accessIndexExpression plan loopValue indexExpression = case indexExpression of
  IR.IteratorIndex iterator -> logicalIndexExpression plan iterator loopValue
  IR.ConstantIndex index -> Right (show index ++ "u")

outputIndexExpression :: IR.ComputeFunction -> IR.ComputeOp -> Schedule.LoopPlan -> (Schedule.LoopId -> Either CodegenError String) -> Either CodegenError String
outputIndexExpression function operation plan loopValue = do
  indices <-
    mapM
      (\iterator -> logicalIndexExpression plan (IR.iteratorId iterator) loopValue)
      (filter ((== IR.Parallel) . IR.iteratorKind) (IR.computeIterators operation))
  outputTensor <-
    maybe
      (Left ScheduledOperationMismatch)
      Right
      (IR.lookupTensor (IR.computeResult operation) function)
  pure (flattenedIndex (IR.tensorValueType outputTensor) indices)

logicalIndexExpression :: Schedule.LoopPlan -> IR.IteratorId -> (Schedule.LoopId -> Either CodegenError String) -> Either CodegenError String
logicalIndexExpression plan iterator loopValue = do
  logicalIndex <-
    maybe
      (Left (MissingLogicalIndex (IR.iteratorIdIndex iterator)))
      Right
      (Schedule.lookupLogicalIndex iterator plan)
  loopIndexExpression (Schedule.logicalExpression logicalIndex) loopValue

loopIndexExpression :: Schedule.LoopIndexExpr -> (Schedule.LoopId -> Either CodegenError String) -> Either CodegenError String
loopIndexExpression expression loopValue = case expression of
  Schedule.LoopIndex loopId -> loopValue loopId
  Schedule.LoopConstant value -> Right (show value ++ "u")
  Schedule.AddIndex lhs rhs -> binary "+" lhs rhs
  Schedule.MulIndex lhs rhs -> binary "*" lhs rhs
  where
    binary operator lhs rhs = do
      lhsExpression <- loopIndexExpression lhs loopValue
      rhsExpression <- loopIndexExpression rhs loopValue
      pure ("(" ++ lhsExpression ++ " " ++ operator ++ " " ++ rhsExpression ++ ")")

tailCondition :: Schedule.LoopPlan -> (Schedule.LoopId -> Either CodegenError String) -> Either CodegenError (Maybe String)
tailCondition plan loopValue = do
  predicates <-
    mapM
      (\predicate -> do
         index <- loopIndexExpression (Schedule.tailPredicateIndex predicate) loopValue
         pure (index ++ " < " ++ loopExtentExpression (Schedule.tailPredicateExtent predicate))
      )
      (concatMap Schedule.logicalTailPredicates (Schedule.planLogicalIndices plan))
  pure $ case nub (sort predicates) of
    [] -> Nothing
    conditions -> Just (joinWith " && " conditions)

flattenedIndex :: IR.TensorType -> [String] -> String
flattenedIndex _ [] = "0u"
flattenedIndex tensorType (firstIndex : remainingIndices) =
  foldl flatten firstIndex (zip (drop 1 (IR.tensorShape tensorType)) remainingIndices)
  where
    flatten expression (dimension, index) =
      "((" ++ expression ++ ") * " ++ dimExpression dimension ++ " + " ++ index ++ ")"

dimExpression :: IR.Dim -> String
dimExpression dimension = case dimension of
  IR.StaticDim extent -> show extent
  IR.SymbolDim symbol -> "symbol" ++ show (IR.symbolIdIndex symbol)

loopExtentExpression :: Schedule.LoopExtent -> String
loopExtentExpression extent = case extent of
  Schedule.StaticExtent value -> show value
  Schedule.SymbolExtent symbol -> "symbol" ++ show (IR.symbolIdIndex symbol)
  Schedule.CeilDivExtent dividend divisor ->
    "(("
      ++ loopExtentExpression dividend
      ++ ") + "
      ++ show (divisor - 1)
      ++ "u) / "
      ++ show divisor
      ++ "u"

scalarName :: IR.ScalarValueId -> String
scalarName value = "scalar" ++ show (IR.scalarValueIdIndex value)

scalarTypeName :: IR.ScalarType -> String
scalarTypeName scalarType = case scalarType of
  IR.F32 -> "float"
  IR.Bool -> "bool"
  IR.Index -> "size_t"

scalarLiteral :: IR.ScalarLiteral -> String
scalarLiteral literal = case literal of
  IR.F32Literal value
    | isNaN value -> "NAN"
    | isInfinite value && value > 0 -> "INFINITY"
    | isInfinite value -> "-INFINITY"
    | otherwise -> rustDebugFloat value ++ "f"
  IR.BoolLiteral value -> if value then "true" else "false"
  IR.IndexLiteral value -> show value ++ "u"

rustDebugFloat :: Float -> String
rustDebugFloat value
  | value == 0 = if isNegativeZero value then "-0.0" else "0.0"
  | decimalExponent < -4 || decimalExponent >= 16 = scientific
  | decimalPoint <= 0 = sign ++ "0." ++ replicate (negate decimalPoint) '0' ++ digits
  | decimalPoint < length digits =
      sign ++ take decimalPoint digits ++ "." ++ drop decimalPoint digits
  | otherwise =
      sign ++ digits ++ replicate (decimalPoint - length digits) '0' ++ ".0"
  where
    sign = if value < 0 then "-" else ""
    (coefficient, scaleExponent) = shortestFloatDecimal (abs value)
    digits = show coefficient
    decimalPoint = length digits + scaleExponent
    decimalExponent = decimalPoint - 1
    scientific = case digits of
      [] -> error "shortestFloatDecimal returned no digits"
      leadingDigit : remainingDigits ->
        sign
          ++ [leadingDigit]
          ++ (if null remainingDigits then "" else "." ++ remainingDigits)
          ++ "e"
          ++ show decimalExponent

shortestFloatDecimal :: Float -> (Integer, Int)
shortestFloatDecimal value = normalizeDecimal (findCoefficient 1)
  where
    bits = castFloatToWord32 value
    exact = floatRational value
    previous
      | bits == 1 = 0
      | otherwise = floatRational (castWord32ToFloat (bits - 1))
    following
      | bits == maxFiniteFloatBits = (2 ^ (128 :: Int)) % 1
      | otherwise = floatRational (castWord32ToFloat (bits + 1))
    lowerBoundary = (previous + exact) / 2
    upperBoundary = (exact + following) / 2
    inclusiveBoundary = even bits
    magnitudeExponent = floatDecimalMagnitude exact

    findCoefficient significantDigits
      | significantDigits > 9 = error "a finite f32 must have a nine-digit decimal representation"
      | lowerCoefficient <= upperCoefficient = (coefficient, scaleExponent)
      | otherwise = findCoefficient (significantDigits + 1)
      where
        scaleExponent = magnitudeExponent - significantDigits + 1
        scale = decimalPower scaleExponent
        lowerCoefficient =
          max
            (10 ^ (significantDigits - 1))
            (integerAbove inclusiveBoundary (lowerBoundary / scale))
        upperCoefficient =
          min
            (10 ^ significantDigits)
            (integerBelow inclusiveBoundary (upperBoundary / scale))
        nearestCoefficient = roundShortest (exact / scale)
        coefficient = max lowerCoefficient (min upperCoefficient nearestCoefficient)

normalizeDecimal :: (Integer, Int) -> (Integer, Int)
normalizeDecimal (coefficient, scaleExponent)
  | coefficient `rem` 10 == 0 = normalizeDecimal (coefficient `quot` 10, scaleExponent + 1)
  | otherwise = (coefficient, scaleExponent)

maxFiniteFloatBits :: Word32
maxFiniteFloatBits = 0x7f7fffff

floatRational :: Float -> Rational
floatRational value =
  let (floatSignificand, floatExponent) = decodeFloat value
   in (floatSignificand % 1) * binaryPower floatExponent

binaryPower :: Int -> Rational
binaryPower power
  | power >= 0 = (2 ^ power) % 1
  | otherwise = 1 % (2 ^ negate power)

decimalPower :: Int -> Rational
decimalPower power
  | power >= 0 = (10 ^ power) % 1
  | otherwise = 1 % (10 ^ negate power)

floatDecimalMagnitude :: Rational -> Int
floatDecimalMagnitude value = findMagnitude (-46)
  where
    findMagnitude magnitude
      | value < decimalPower (magnitude + 1) = magnitude
      | otherwise = findMagnitude (magnitude + 1)

integerAbove :: Bool -> Rational -> Integer
integerAbove inclusive value
  | inclusive = ceiling value
  | denominator value == 1 = numerator value + 1
  | otherwise = ceiling value

integerBelow :: Bool -> Rational -> Integer
integerBelow inclusive value
  | inclusive = floor value
  | denominator value == 1 = numerator value - 1
  | otherwise = floor value

roundShortest :: Rational -> Integer
roundShortest value
  | remainder * 2 < denominator value = quotient
  | otherwise = quotient + 1
  where
    (quotient, remainder) = quotRem (numerator value) (denominator value)

compareOperator :: IR.ComparePredicate -> String
compareOperator predicate = case predicate of
  IR.Equal -> "=="
  IR.NotEqual -> "!="
  IR.Less -> "<"
  IR.LessEqual -> "<="
  IR.Greater -> ">"
  IR.GreaterEqual -> ">="

cudaBindingExpression :: Schedule.CudaBinding -> String
cudaBindingExpression binding = case binding of
  Schedule.BlockX -> "((size_t)blockIdx.x)"
  Schedule.BlockY -> "((size_t)blockIdx.y)"
  Schedule.BlockZ -> "((size_t)blockIdx.z)"
  Schedule.ThreadX -> "((size_t)threadIdx.x)"
  Schedule.ThreadY -> "((size_t)threadIdx.y)"
  Schedule.ThreadZ -> "((size_t)threadIdx.z)"

validateCodegenFunction :: IR.ComputeFunction -> IR.ComputeOp -> Either CodegenError ()
validateCodegenFunction function operation = case operations of
  [functionOperation]
    | IR.computeOpId functionOperation /= IR.computeOpId operation -> Left ScheduledOperationMismatch
    | otherwise -> case outputs of
        [output]
          | output /= IR.computeResult operation -> Left ScheduledOperationIsNotOutput
          | unsupportedTensorElement -> Left UnsupportedTensorElement
          | otherwise -> Right ()
        _ -> Left (ExactlyOneOutputRequired (length outputs))
  _ -> Left (ExactlyOneOperationRequired (length operations))
  where
    operations = IR.functionOperations function
    outputs = IR.functionOutputs function
    unsupportedTensorElement =
      any
        ((/= IR.F32) . IR.tensorElementType . IR.tensorValueType)
        (IR.functionTensors function)

inputName :: IR.ComputeFunction -> IR.TensorValueId -> Either CodegenError String
inputName function tensorId = case IR.lookupTensor tensorId function of
  Just tensor -> case IR.tensorDefinition tensor of
    IR.InputTensor inputIndex -> Right ("input" ++ show inputIndex)
    IR.ComputeResult _ -> Left nonInputError
  Nothing -> Left nonInputError
  where
    nonInputError = NonInputTensorAccess (IR.tensorValueIdIndex tensorId)

renderSource :: [SourceNode] -> String
renderSource = concatMap (renderNode 0)
  where
    renderNode indentation node = case node of
      SourceLine line -> indent indentation ++ line ++ "\n"
      BlankLine -> "\n"
      SourceScope openLine body closeLine ->
        indent indentation
          ++ openLine
          ++ "\n"
          ++ concatMap (renderNode (indentation + 1)) body
          ++ indent indentation
          ++ closeLine
          ++ "\n"
    indent indentation = replicate (indentation * 4) ' '

commaSeparated :: [String] -> String
commaSeparated = joinWith ", "

joinWith :: String -> [String] -> String
joinWith _ [] = ""
joinWith _ [value] = value
joinWith separator (value : values) = value ++ separator ++ joinWith separator values
