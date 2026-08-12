{-# OPTIONS_GHC -fno-cse -fno-full-laziness #-}

module Aoewif.IR (
    SymbolId,
    symbolIdIndex,
    TensorValueId,
    tensorValueIdIndex,
    ComputeOpId,
    computeOpIdIndex,
    IteratorId,
    iteratorIdIndex,
    ScalarValueId,
    scalarValueIdIndex,
    Dim (..),
    ScalarType (..),
    TensorType (..),
    tensorTypeF32,
    tensorRank,
    Symbol (..),
    TensorDefinition (..),
    TensorValue (..),
    IteratorKind (..),
    Iterator (..),
    IndexExpr (..),
    TensorAccess (..),
    ScalarLiteral (..),
    scalarLiteralType,
    ComparePredicate (..),
    ScalarOperationKind (..),
    scalarOperationOperands,
    ScalarOperation (..),
    ScalarArgumentKind (..),
    ScalarArgument (..),
    ScalarRegion (..),
    ReductionPolicy (..),
    ComputeOp (..),
    hasReduction,
    ComputeFunction (..),
    lookupTensor,
    lookupOperation,
    inputTensors,
    VerifiedComputeFunction,
    verifiedFunction,
    IrError (..),
    FunctionBuilder,
    ComputeBuilder,
    buildComputeFunction,
    buildComputeFunctionWith,
    symbol,
    input,
    compute,
    computeWith,
    markOutput,
    parallel,
    reduction,
    readTensor,
    reductionInit,
    constant,
    index,
    add,
    sub,
    mul,
    divide,
    fma,
    minimum,
    maximum,
    expScalar,
    logScalar,
    compare,
    selectScalar,
) where

import           Control.Exception (Exception)
import           Data.Char         (isAsciiLower, isAsciiUpper, isDigit)
import           Data.IORef        (IORef, atomicModifyIORef', newIORef)
import           Data.List         (find)
import           Data.Word         (Word64)
import           Prelude           hiding (compare, maximum, minimum)
import           System.IO.Unsafe  (unsafePerformIO)

newtype Owner = Owner Word64
    deriving stock (Eq, Ord, Show)

data SymbolId = SymbolId !Owner !Int
    deriving stock (Eq, Ord)

data TensorValueId = TensorValueId !Owner !Int
    deriving stock (Eq, Ord)

data ComputeOpId = ComputeOpId !Owner !Int
    deriving stock (Eq, Ord)

data IteratorId = IteratorId !Owner !Int
    deriving stock (Eq, Ord)

data ScalarValueId = ScalarValueId !Owner !Int
    deriving stock (Eq, Ord)

instance Show SymbolId where
    show = showId "SymbolId" symbolIdIndex

instance Show TensorValueId where
    show = showId "TensorValueId" tensorValueIdIndex

instance Show ComputeOpId where
    show = showId "ComputeOpId" computeOpIdIndex

instance Show IteratorId where
    show = showId "IteratorId" iteratorIdIndex

instance Show ScalarValueId where
    show = showId "ScalarValueId" scalarValueIdIndex

showId :: String -> (identifier -> Int) -> identifier -> String
showId name getIndex identifier = name <> " " <> show (getIndex identifier)

symbolIdIndex :: SymbolId -> Int
symbolIdIndex (SymbolId _ identifierIndex) = identifierIndex

tensorValueIdIndex :: TensorValueId -> Int
tensorValueIdIndex (TensorValueId _ identifierIndex) = identifierIndex

computeOpIdIndex :: ComputeOpId -> Int
computeOpIdIndex (ComputeOpId _ identifierIndex) = identifierIndex

iteratorIdIndex :: IteratorId -> Int
iteratorIdIndex (IteratorId _ identifierIndex) = identifierIndex

scalarValueIdIndex :: ScalarValueId -> Int
scalarValueIdIndex (ScalarValueId _ identifierIndex) = identifierIndex

data Dim
    = StaticDim Word64
    | SymbolDim SymbolId
    deriving stock (Eq, Show)

data ScalarType
    = F32
    | Bool
    | Index
    deriving stock (Eq, Show)

data TensorType = TensorType
    { tensorShape       :: [Dim]
    , tensorElementType :: ScalarType
    }
    deriving stock (Eq, Show)

tensorTypeF32 :: [Dim] -> TensorType
tensorTypeF32 shape = TensorType shape F32

tensorRank :: TensorType -> Int
tensorRank = length . tensorShape

data Symbol = Symbol
    { symbolId   :: SymbolId
    , symbolName :: String
    }
    deriving stock (Eq, Show)

data TensorDefinition
    = InputTensor Int
    | ComputeResult ComputeOpId
    deriving stock (Eq, Show)

data TensorValue = TensorValue
    { tensorValueId    :: TensorValueId
    , tensorValueName  :: String
    , tensorValueType  :: TensorType
    , tensorDefinition :: TensorDefinition
    }
    deriving stock (Eq, Show)

data IteratorKind
    = Parallel
    | Reduction
    deriving stock (Eq, Show)

data Iterator = Iterator
    { iteratorId     :: IteratorId
    , iteratorName   :: String
    , iteratorExtent :: Dim
    , iteratorKind   :: IteratorKind
    }
    deriving stock (Eq, Show)

data IndexExpr
    = IteratorIndex IteratorId
    | ConstantIndex Word64
    deriving stock (Eq, Show)

data TensorAccess = TensorAccess
    { accessTensor  :: TensorValueId
    , accessIndices :: [IndexExpr]
    , accessScalar  :: ScalarValueId
    }
    deriving stock (Eq, Show)

data ScalarLiteral
    = F32Literal Float
    | BoolLiteral Bool
    | IndexLiteral Word64
    deriving stock (Eq, Show)

scalarLiteralType :: ScalarLiteral -> ScalarType
scalarLiteralType (F32Literal _)   = F32
scalarLiteralType (BoolLiteral _)  = Bool
scalarLiteralType (IndexLiteral _) = Index

data ComparePredicate
    = Equal
    | NotEqual
    | Less
    | LessEqual
    | Greater
    | GreaterEqual
    deriving stock (Eq, Show)

data ScalarOperationKind
    = ConstantOperation ScalarLiteral
    | IndexOperation IteratorId
    | AddOperation ScalarValueId ScalarValueId
    | SubOperation ScalarValueId ScalarValueId
    | MulOperation ScalarValueId ScalarValueId
    | DivOperation ScalarValueId ScalarValueId
    | FmaOperation ScalarValueId ScalarValueId ScalarValueId
    | MinOperation ScalarValueId ScalarValueId
    | MaxOperation ScalarValueId ScalarValueId
    | ExpOperation ScalarValueId
    | LogOperation ScalarValueId
    | CompareOperation ComparePredicate ScalarValueId ScalarValueId
    | SelectOperation ScalarValueId ScalarValueId ScalarValueId
    deriving stock (Eq, Show)

scalarOperationOperands :: ScalarOperationKind -> [ScalarValueId]
scalarOperationOperands (ConstantOperation _) = []
scalarOperationOperands (IndexOperation _) = []
scalarOperationOperands (AddOperation lhs rhs) = [lhs, rhs]
scalarOperationOperands (SubOperation lhs rhs) = [lhs, rhs]
scalarOperationOperands (MulOperation lhs rhs) = [lhs, rhs]
scalarOperationOperands (DivOperation lhs rhs) = [lhs, rhs]
scalarOperationOperands (FmaOperation lhs rhs accumulator) = [lhs, rhs, accumulator]
scalarOperationOperands (MinOperation lhs rhs) = [lhs, rhs]
scalarOperationOperands (MaxOperation lhs rhs) = [lhs, rhs]
scalarOperationOperands (ExpOperation value) = [value]
scalarOperationOperands (LogOperation value) = [value]
scalarOperationOperands (CompareOperation _ lhs rhs) = [lhs, rhs]
scalarOperationOperands (SelectOperation condition trueValue falseValue) =
    [condition, trueValue, falseValue]

data ScalarOperation = ScalarOperation
    { scalarOperationResult     :: ScalarValueId
    , scalarOperationResultType :: ScalarType
    , scalarOperationKind       :: ScalarOperationKind
    }
    deriving stock (Eq, Show)

data ScalarArgumentKind
    = InputElement Int
    | Accumulator
    deriving stock (Eq, Show)

data ScalarArgument = ScalarArgument
    { scalarArgumentValue :: ScalarValueId
    , scalarArgumentType  :: ScalarType
    , scalarArgumentKind  :: ScalarArgumentKind
    }
    deriving stock (Eq, Show)

data ScalarRegion = ScalarRegion
    { scalarArguments  :: [ScalarArgument]
    , scalarOperations :: [ScalarOperation]
    , scalarResult     :: ScalarValueId
    }
    deriving stock (Eq, Show)

data ReductionPolicy = Strict
    deriving stock (Eq, Show)

data ComputeOp = ComputeOp
    { computeOpId            :: ComputeOpId
    , computeOpName          :: String
    , computeIterators       :: [Iterator]
    , computeAccesses        :: [TensorAccess]
    , computeInit            :: Maybe ScalarLiteral
    , computeReductionPolicy :: ReductionPolicy
    , computeBody            :: ScalarRegion
    , computeResult          :: TensorValueId
    }
    deriving stock (Eq, Show)

hasReduction :: ComputeOp -> Bool
hasReduction = any ((== Reduction) . iteratorKind) . computeIterators

data ComputeFunction = ComputeFunction
    { functionOwner      :: Owner
    , functionName       :: String
    , functionSymbols    :: [Symbol]
    , functionTensors    :: [TensorValue]
    , functionOperations :: [ComputeOp]
    , functionOutputs    :: [TensorValueId]
    }
    deriving stock (Eq, Show)

lookupTensor :: TensorValueId -> ComputeFunction -> Maybe TensorValue
lookupTensor (TensorValueId owner identifierIndex) function
    | owner == functionOwner function = atMay identifierIndex (functionTensors function)
    | otherwise = Nothing

lookupOperation :: ComputeOpId -> ComputeFunction -> Maybe ComputeOp
lookupOperation (ComputeOpId owner identifierIndex) function
    | owner == functionOwner function = atMay identifierIndex (functionOperations function)
    | otherwise = Nothing

inputTensors :: ComputeFunction -> [TensorValue]
inputTensors = filter isInput . functionTensors
  where
    isInput tensor = case tensorDefinition tensor of
        InputTensor _   -> True
        ComputeResult _ -> False

newtype VerifiedComputeFunction = VerifiedComputeFunction ComputeFunction
    deriving stock (Eq, Show)

verifiedFunction :: VerifiedComputeFunction -> ComputeFunction
verifiedFunction (VerifiedComputeFunction function) = function

data IrError
    = InvalidIdentifier String
    | DuplicateName String
    | ForeignId String
    | UnknownTensor Int
    | UnknownIterator Int
    | UnknownScalar Int
    | TensorRankMismatch Int Int
    | TensorElementTypeUnsupported ScalarType
    | DimensionMismatch
    | ConstantIndexRequiresStaticDimension
    | ConstantIndexOutOfBounds Word64 Word64
    | ReductionInitRequired
    | ReductionInitWithoutReduction
    | ReductionAccumulatorRequired
    | AccumulatorWithoutReduction
    | DuplicateAccumulator
    | ScalarTypeMismatch ScalarType ScalarType
    | ScalarOperandsMustMatch
    | ScalarResultMustMatchInit
    | InvalidScalarResult
    | OutputRequired
    | OutputAlreadyMarked Int
    deriving stock (Eq)

instance Exception IrError

instance Show IrError where
    show (InvalidIdentifier value) = "invalid identifier `" <> value <> "`"
    show (DuplicateName value) = "duplicate name `" <> value <> "`"
    show (ForeignId kind) = "foreign " <> kind <> " ID"
    show (UnknownTensor identifierIndex) = "unknown tensor value " <> show identifierIndex
    show (UnknownIterator identifierIndex) = "unknown iterator " <> show identifierIndex
    show (UnknownScalar identifierIndex) = "unknown scalar value " <> show identifierIndex
    show (TensorRankMismatch expected actual) =
        "tensor rank mismatch: expected " <> show expected <> ", got " <> show actual
    show (TensorElementTypeUnsupported scalarType) =
        "unsupported tensor element type " <> show scalarType
    show DimensionMismatch = "iterator extent does not match tensor dimension"
    show ConstantIndexRequiresStaticDimension =
        "constant indices require a static tensor dimension"
    show (ConstantIndexOutOfBounds indexValue extent) =
        "constant index " <> show indexValue <> " is outside extent " <> show extent
    show ReductionInitRequired = "reduction requires an initial value"
    show ReductionInitWithoutReduction = "initial value is only valid for a reduction"
    show ReductionAccumulatorRequired = "reduction body requires an accumulator argument"
    show AccumulatorWithoutReduction = "accumulator is only valid for a reduction"
    show DuplicateAccumulator = "reduction already has an accumulator"
    show (ScalarTypeMismatch expected actual) =
        "scalar type mismatch: expected " <> show expected <> ", got " <> show actual
    show ScalarOperandsMustMatch = "scalar operand types must match"
    show ScalarResultMustMatchInit = "reduction result type must match its initial value"
    show InvalidScalarResult = "invalid scalar region result"
    show OutputRequired = "compute function requires at least one output"
    show (OutputAlreadyMarked identifierIndex) =
        "tensor value " <> show identifierIndex <> " is already marked as an output"

data FunctionState = FunctionState
    { stateFunction     :: ComputeFunction
    , stateNames        :: [String]
    , stateNextIterator :: Int
    , stateNextScalar   :: Int
    }

newtype FunctionBuilder value = FunctionBuilder
    { runFunctionBuilder :: FunctionState -> Either IrError (value, FunctionState)
    }

instance Functor FunctionBuilder where
    fmap transform (FunctionBuilder action) = FunctionBuilder $ \state -> do
        (value, nextState) <- action state
        pure (transform value, nextState)

instance Applicative FunctionBuilder where
    pure value = FunctionBuilder $ \state -> Right (value, state)
    FunctionBuilder functionAction <*> FunctionBuilder valueAction = FunctionBuilder $ \state -> do
        (function, functionState) <- functionAction state
        (value, valueState) <- valueAction functionState
        pure (function value, valueState)

instance Monad FunctionBuilder where
    FunctionBuilder action >>= next = FunctionBuilder $ \state -> do
        (value, nextState) <- action state
        runFunctionBuilder (next value) nextState

data ComputeState = ComputeState
    { computeFunctionState      :: FunctionState
    , computeBuilderName        :: String
    , computeBuilderIterators   :: [Iterator]
    , computeBuilderAccesses    :: [TensorAccess]
    , computeBuilderInit        :: Maybe ScalarLiteral
    , computeBuilderAccumulator :: Maybe ScalarValueId
    , computeBuilderArguments   :: [ScalarArgument]
    , computeBuilderOperations  :: [ScalarOperation]
    , computeBuilderScalarTypes :: [(ScalarValueId, ScalarType)]
    }

newtype ComputeBuilder value = ComputeBuilder
    { runComputeBuilder :: ComputeState -> Either IrError (value, ComputeState)
    }

instance Functor ComputeBuilder where
    fmap transform (ComputeBuilder action) = ComputeBuilder $ \state -> do
        (value, nextState) <- action state
        pure (transform value, nextState)

instance Applicative ComputeBuilder where
    pure value = ComputeBuilder $ \state -> Right (value, state)
    ComputeBuilder functionAction <*> ComputeBuilder valueAction = ComputeBuilder $ \state -> do
        (function, functionState) <- functionAction state
        (value, valueState) <- valueAction functionState
        pure (function value, valueState)

instance Monad ComputeBuilder where
    ComputeBuilder action >>= next = ComputeBuilder $ \state -> do
        (value, nextState) <- action state
        runComputeBuilder (next value) nextState

buildComputeFunction :: String -> FunctionBuilder value -> Either IrError VerifiedComputeFunction
buildComputeFunction name builder = snd <$> buildComputeFunctionWith name builder

buildComputeFunctionWith ::
    String ->
    FunctionBuilder value ->
    Either IrError (value, VerifiedComputeFunction)
buildComputeFunctionWith name builder = do
    validateIdentifier name
    let initialFunction =
            ComputeFunction
                { functionOwner = freshOwner name
                , functionName = name
                , functionSymbols = []
                , functionTensors = []
                , functionOperations = []
                , functionOutputs = []
                }
        initialState = FunctionState initialFunction [] 0 0
    (value, finalState) <- runFunctionBuilder builder initialState
    verifyFunction (stateFunction finalState)
    pure (value, VerifiedComputeFunction (stateFunction finalState))

symbol :: String -> FunctionBuilder SymbolId
symbol name = FunctionBuilder $ \state -> do
    reserved <- reserveFunctionName name state
    let function = stateFunction reserved
        identifier = SymbolId (functionOwner function) (length (functionSymbols function))
        newSymbol = Symbol identifier name
        nextFunction = function{functionSymbols = functionSymbols function <> [newSymbol]}
    pure (identifier, reserved{stateFunction = nextFunction})

input :: String -> TensorType -> FunctionBuilder TensorValueId
input name tensorType = FunctionBuilder $ \state -> do
    reserved <- reserveFunctionName name state
    verifyTensorType reserved tensorType
    let function = stateFunction reserved
        identifier = TensorValueId (functionOwner function) (length (functionTensors function))
        inputIndex = length (inputTensors function)
        tensor = TensorValue identifier name tensorType (InputTensor inputIndex)
        nextFunction = function{functionTensors = functionTensors function <> [tensor]}
    pure (identifier, reserved{stateFunction = nextFunction})

compute :: String -> ComputeBuilder ScalarValueId -> FunctionBuilder TensorValueId
compute name body = snd <$> computeWith name ((,) () <$> body)

computeWith ::
    String ->
    ComputeBuilder (value, ScalarValueId) ->
    FunctionBuilder (value, TensorValueId)
computeWith name body = FunctionBuilder $ \state -> do
    reserved <- reserveFunctionName name state
    let initialComputeState =
            ComputeState
                { computeFunctionState = reserved
                , computeBuilderName = name
                , computeBuilderIterators = []
                , computeBuilderAccesses = []
                , computeBuilderInit = Nothing
                , computeBuilderAccumulator = Nothing
                , computeBuilderArguments = []
                , computeBuilderOperations = []
                , computeBuilderScalarTypes = []
                }
    ((value, result), finalComputeState) <- runComputeBuilder body initialComputeState
    (tensorIdentifier, finalFunctionState) <- finishCompute result finalComputeState
    pure ((value, tensorIdentifier), finalFunctionState)

markOutput :: TensorValueId -> FunctionBuilder ()
markOutput identifier = FunctionBuilder $ \state -> do
    _ <- tensorById identifier state
    let function = stateFunction state
    if identifier `elem` functionOutputs function
        then Left (OutputAlreadyMarked (tensorValueIdIndex identifier))
        else
            let nextFunction = function{functionOutputs = functionOutputs function <> [identifier]}
             in pure ((), state{stateFunction = nextFunction})

parallel :: String -> Dim -> ComputeBuilder IteratorId
parallel name extent = iterator name extent Parallel

reduction :: String -> Dim -> ComputeBuilder IteratorId
reduction name extent = iterator name extent Reduction

readTensor :: TensorValueId -> [IndexExpr] -> ComputeBuilder ScalarValueId
readTensor tensorIdentifier indices = ComputeBuilder $ \state -> do
    tensor <- tensorById tensorIdentifier (computeFunctionState state)
    verifyAccess state (tensorValueType tensor) indices
    let (scalar, allocatedState) = allocateScalar state
        accessIndex = length (computeBuilderAccesses state)
        access = TensorAccess tensorIdentifier indices scalar
        argument = ScalarArgument scalar (tensorElementType (tensorValueType tensor)) (InputElement accessIndex)
        scalarType = tensorElementType (tensorValueType tensor)
        nextState =
            allocatedState
                { computeBuilderAccesses = computeBuilderAccesses state <> [access]
                , computeBuilderArguments = computeBuilderArguments state <> [argument]
                , computeBuilderScalarTypes = (scalar, scalarType) : computeBuilderScalarTypes state
                }
    pure (scalar, nextState)

reductionInit :: ScalarLiteral -> ComputeBuilder ScalarValueId
reductionInit literal = ComputeBuilder $ \state -> case computeBuilderAccumulator state of
    Just _ -> Left DuplicateAccumulator
    Nothing ->
        let (accumulator, allocatedState) = allocateScalar state
            argument = ScalarArgument accumulator (scalarLiteralType literal) Accumulator
            nextState =
                allocatedState
                    { computeBuilderInit = Just literal
                    , computeBuilderAccumulator = Just accumulator
                    , computeBuilderArguments = computeBuilderArguments state <> [argument]
                    , computeBuilderScalarTypes =
                        (accumulator, scalarLiteralType literal) : computeBuilderScalarTypes state
                    }
         in pure (accumulator, nextState)

constant :: ScalarLiteral -> ComputeBuilder ScalarValueId
constant literal = pushOperation (ConstantOperation literal) (scalarLiteralType literal)

index :: IteratorId -> ComputeBuilder ScalarValueId
index identifier = ComputeBuilder $ \state -> do
    _ <- iteratorById identifier state
    runComputeBuilder (pushOperation (IndexOperation identifier) Index) state

add :: ScalarValueId -> ScalarValueId -> ComputeBuilder ScalarValueId
add = binaryF32 AddOperation

sub :: ScalarValueId -> ScalarValueId -> ComputeBuilder ScalarValueId
sub = binaryF32 SubOperation

mul :: ScalarValueId -> ScalarValueId -> ComputeBuilder ScalarValueId
mul = binaryF32 MulOperation

divide :: ScalarValueId -> ScalarValueId -> ComputeBuilder ScalarValueId
divide = binaryF32 DivOperation

fma :: ScalarValueId -> ScalarValueId -> ScalarValueId -> ComputeBuilder ScalarValueId
fma lhs rhs accumulator = ComputeBuilder $ \state -> do
    expectType lhs F32 state
    expectType rhs F32 state
    expectType accumulator F32 state
    runComputeBuilder (pushOperation (FmaOperation lhs rhs accumulator) F32) state

minimum :: ScalarValueId -> ScalarValueId -> ComputeBuilder ScalarValueId
minimum = binaryF32 MinOperation

maximum :: ScalarValueId -> ScalarValueId -> ComputeBuilder ScalarValueId
maximum = binaryF32 MaxOperation

expScalar :: ScalarValueId -> ComputeBuilder ScalarValueId
expScalar value = unaryF32 ExpOperation value

logScalar :: ScalarValueId -> ComputeBuilder ScalarValueId
logScalar value = unaryF32 LogOperation value

compare :: ComparePredicate -> ScalarValueId -> ScalarValueId -> ComputeBuilder ScalarValueId
compare predicate lhs rhs = ComputeBuilder $ \state -> do
    lhsType <- scalarTypeOf lhs state
    rhsType <- scalarTypeOf rhs state
    if lhsType /= rhsType
        then Left ScalarOperandsMustMatch
        else runComputeBuilder (pushOperation (CompareOperation predicate lhs rhs) Bool) state

selectScalar ::
    ScalarValueId ->
    ScalarValueId ->
    ScalarValueId ->
    ComputeBuilder ScalarValueId
selectScalar condition trueValue falseValue = ComputeBuilder $ \state -> do
    expectType condition Bool state
    trueType <- scalarTypeOf trueValue state
    falseType <- scalarTypeOf falseValue state
    if trueType /= falseType
        then Left ScalarOperandsMustMatch
        else
            runComputeBuilder
                (pushOperation (SelectOperation condition trueValue falseValue) trueType)
                state

iterator :: String -> Dim -> IteratorKind -> ComputeBuilder IteratorId
iterator name extent kind = ComputeBuilder $ \state -> do
    validateIdentifier name
    if any ((== name) . iteratorName) (computeBuilderIterators state)
        then Left (DuplicateName name)
        else do
            verifyDim (computeFunctionState state) extent
            let functionState = computeFunctionState state
                function = stateFunction functionState
                identifier = IteratorId (functionOwner function) (stateNextIterator functionState)
                newIterator = Iterator identifier name extent kind
                nextFunctionState = functionState{stateNextIterator = stateNextIterator functionState + 1}
                nextState =
                    state
                        { computeFunctionState = nextFunctionState
                        , computeBuilderIterators = computeBuilderIterators state <> [newIterator]
                        }
            pure (identifier, nextState)

finishCompute ::
    ScalarValueId ->
    ComputeState ->
    Either IrError (TensorValueId, FunctionState)
finishCompute result state = do
    resultType <- scalarTypeOf result state
    let hasReductionIterator = any ((== Reduction) . iteratorKind) (computeBuilderIterators state)
    case (hasReductionIterator, computeBuilderInit state, computeBuilderAccumulator state) of
        (True, Nothing, _)       -> Left ReductionInitRequired
        (True, Just _, Nothing)  -> Left ReductionAccumulatorRequired
        (False, Just _, _)       -> Left ReductionInitWithoutReduction
        (False, Nothing, Just _) -> Left AccumulatorWithoutReduction
        _                        -> Right ()
    case computeBuilderInit state of
        Just initialValue
            | resultType /= scalarLiteralType initialValue -> Left ScalarResultMustMatchInit
        _ -> Right ()
    if resultType /= F32
        then Left (TensorElementTypeUnsupported resultType)
        else do
            let functionState = computeFunctionState state
                function = stateFunction functionState
                operationIdentifier = ComputeOpId (functionOwner function) (length (functionOperations function))
                tensorIdentifier = TensorValueId (functionOwner function) (length (functionTensors function))
                resultShape =
                    [ iteratorExtent computeIterator
                    | computeIterator <- computeBuilderIterators state
                    , iteratorKind computeIterator == Parallel
                    ]
                tensor =
                    TensorValue
                        tensorIdentifier
                        (computeBuilderName state)
                        (tensorTypeF32 resultShape)
                        (ComputeResult operationIdentifier)
                operation =
                    ComputeOp
                        { computeOpId = operationIdentifier
                        , computeOpName = computeBuilderName state
                        , computeIterators = computeBuilderIterators state
                        , computeAccesses = computeBuilderAccesses state
                        , computeInit = computeBuilderInit state
                        , computeReductionPolicy = Strict
                        , computeBody =
                            ScalarRegion
                                (computeBuilderArguments state)
                                (computeBuilderOperations state)
                                result
                        , computeResult = tensorIdentifier
                        }
                nextFunction =
                    function
                        { functionTensors = functionTensors function <> [tensor]
                        , functionOperations = functionOperations function <> [operation]
                        }
            pure (tensorIdentifier, functionState{stateFunction = nextFunction})

binaryF32 ::
    (ScalarValueId -> ScalarValueId -> ScalarOperationKind) ->
    ScalarValueId ->
    ScalarValueId ->
    ComputeBuilder ScalarValueId
binaryF32 constructor lhs rhs = ComputeBuilder $ \state -> do
    expectType lhs F32 state
    expectType rhs F32 state
    runComputeBuilder (pushOperation (constructor lhs rhs) F32) state

unaryF32 ::
    (ScalarValueId -> ScalarOperationKind) ->
    ScalarValueId ->
    ComputeBuilder ScalarValueId
unaryF32 constructor value = ComputeBuilder $ \state -> do
    expectType value F32 state
    runComputeBuilder (pushOperation (constructor value) F32) state

pushOperation :: ScalarOperationKind -> ScalarType -> ComputeBuilder ScalarValueId
pushOperation kind resultType = ComputeBuilder $ \state ->
    let (result, allocatedState) = allocateScalar state
        operation = ScalarOperation result resultType kind
        nextState =
            allocatedState
                { computeBuilderOperations = computeBuilderOperations state <> [operation]
                , computeBuilderScalarTypes = (result, resultType) : computeBuilderScalarTypes state
                }
     in pure (result, nextState)

allocateScalar :: ComputeState -> (ScalarValueId, ComputeState)
allocateScalar state =
    let functionState = computeFunctionState state
        function = stateFunction functionState
        identifier = ScalarValueId (functionOwner function) (stateNextScalar functionState)
        nextFunctionState = functionState{stateNextScalar = stateNextScalar functionState + 1}
     in (identifier, state{computeFunctionState = nextFunctionState})

reserveFunctionName :: String -> FunctionState -> Either IrError FunctionState
reserveFunctionName name state = do
    validateIdentifier name
    if name `elem` stateNames state
        then Left (DuplicateName name)
        else Right state{stateNames = stateNames state <> [name]}

verifyTensorType :: FunctionState -> TensorType -> Either IrError ()
verifyTensorType state tensorType = do
    if tensorElementType tensorType /= F32
        then Left (TensorElementTypeUnsupported (tensorElementType tensorType))
        else Right ()
    mapM_ (verifyDim state) (tensorShape tensorType)

verifyDim :: FunctionState -> Dim -> Either IrError ()
verifyDim _ (StaticDim _)              = Right ()
verifyDim state (SymbolDim identifier) = () <$ symbolById identifier state

symbolById :: SymbolId -> FunctionState -> Either IrError Symbol
symbolById (SymbolId owner identifierIndex) state
    | owner /= functionOwner function = Left (ForeignId "symbol")
    | otherwise = maybe (Left (ForeignId "symbol")) Right (atMay identifierIndex (functionSymbols function))
  where
    function = stateFunction state

tensorById :: TensorValueId -> FunctionState -> Either IrError TensorValue
tensorById (TensorValueId owner identifierIndex) state
    | owner /= functionOwner function = Left (ForeignId "tensor")
    | otherwise = maybe (Left (UnknownTensor identifierIndex)) Right (atMay identifierIndex (functionTensors function))
  where
    function = stateFunction state

iteratorById :: IteratorId -> ComputeState -> Either IrError Iterator
iteratorById identifier@(IteratorId owner identifierIndex) state
    | owner /= functionOwner function = Left (ForeignId "iterator")
    | otherwise =
        maybe
            (Left (UnknownIterator identifierIndex))
            Right
            (find ((== identifier) . iteratorId) (computeBuilderIterators state))
  where
    function = stateFunction (computeFunctionState state)

scalarTypeOf :: ScalarValueId -> ComputeState -> Either IrError ScalarType
scalarTypeOf identifier@(ScalarValueId owner identifierIndex) state
    | owner /= functionOwner function = Left (ForeignId "scalar")
    | otherwise = maybe (Left (UnknownScalar identifierIndex)) Right (lookup identifier (computeBuilderScalarTypes state))
  where
    function = stateFunction (computeFunctionState state)

expectType :: ScalarValueId -> ScalarType -> ComputeState -> Either IrError ()
expectType value expected state = do
    actual <- scalarTypeOf value state
    if actual == expected
        then Right ()
        else Left (ScalarTypeMismatch expected actual)

verifyAccess :: ComputeState -> TensorType -> [IndexExpr] -> Either IrError ()
verifyAccess state tensorType indices = do
    let expected = tensorRank tensorType
        actual = length indices
    if expected == actual
        then Right ()
        else Left (TensorRankMismatch expected actual)
    mapM_ verifyIndex (zip (tensorShape tensorType) indices)
  where
    verifyIndex (dimension, IteratorIndex identifier) = do
        computeIterator <- iteratorById identifier state
        if iteratorExtent computeIterator == dimension
            then Right ()
            else Left DimensionMismatch
    verifyIndex (StaticDim extent, ConstantIndex indexValue)
        | indexValue < extent = Right ()
        | otherwise = Left (ConstantIndexOutOfBounds indexValue extent)
    verifyIndex (SymbolDim _, ConstantIndex _) = Left ConstantIndexRequiresStaticDimension

validateIdentifier :: String -> Either IrError ()
validateIdentifier [] = Left (InvalidIdentifier [])
validateIdentifier value@(first : rest)
    | validFirst first && all validRest rest = Right ()
    | otherwise = Left (InvalidIdentifier value)
  where
    validFirst character = character == '_' || isAsciiLower character || isAsciiUpper character
    validRest character = validFirst character || isDigit character

verifyFunction :: ComputeFunction -> Either IrError ()
verifyFunction function = do
    if null (functionOutputs function)
        then Left OutputRequired
        else Right ()
    mapM_ verifySymbolOwner (functionSymbols function)
    mapM_ verifyTensor (functionTensors function)
    mapM_ (verifyOperation function) (functionOperations function)
    mapM_ verifyOutput (functionOutputs function)
  where
    owner = functionOwner function
    verifySymbolOwner functionSymbol = case symbolId functionSymbol of
        SymbolId symbolOwner _
            | symbolOwner == owner -> Right ()
            | otherwise -> Left (ForeignId "symbol")
    verifyTensor tensor = do
        case tensorValueId tensor of
            TensorValueId tensorOwner _
                | tensorOwner == owner -> Right ()
                | otherwise -> Left (ForeignId "tensor")
        mapM_ verifyTensorDim (tensorShape (tensorValueType tensor))
    verifyTensorDim (StaticDim _) = Right ()
    verifyTensorDim (SymbolDim identifier) = case identifier of
        SymbolId symbolOwner symbolIndex
            | symbolOwner == owner && atMay symbolIndex (functionSymbols function) /= Nothing -> Right ()
            | otherwise -> Left (ForeignId "symbol")
    verifyOutput identifier = case identifier of
        TensorValueId tensorOwner tensorIndex
            | tensorOwner == owner && atMay tensorIndex (functionTensors function) /= Nothing -> Right ()
            | otherwise -> Left (ForeignId "output tensor")

verifyOperation :: ComputeFunction -> ComputeOp -> Either IrError ()
verifyOperation function operation = do
    mapM_ verifyIteratorOwner iterators
    mapM_ verifyOperationAccess (computeAccesses operation)
    argumentTypes <- foldEither addArgument ([], 0 :: Int) (scalarArguments (computeBody operation))
    scalarTypes <- foldEither addOperation (fst argumentTypes) (scalarOperations (computeBody operation))
    resultType <-
        maybe
            (Left InvalidScalarResult)
            Right
            (lookup (scalarResult (computeBody operation)) scalarTypes)
    verifyReductionContract (snd argumentTypes) resultType
    resultTensor <-
        maybe
            (Left (UnknownTensor (tensorValueIdIndex (computeResult operation))))
            Right
            (lookupTensor (computeResult operation) function)
    let expectedType = tensorElementType (tensorValueType resultTensor)
    if expectedType == resultType
        then Right ()
        else Left (ScalarTypeMismatch expectedType resultType)
  where
    owner = functionOwner function
    iterators = computeIterators operation
    verifyIteratorOwner computeIterator = case iteratorId computeIterator of
        IteratorId iteratorOwner _
            | iteratorOwner == owner -> Right ()
            | otherwise -> Left (ForeignId "iterator")
    verifyOperationAccess access = do
        tensor <-
            maybe
                (Left (UnknownTensor (tensorValueIdIndex (accessTensor access))))
                Right
                (lookupTensor (accessTensor access) function)
        let expected = tensorRank (tensorValueType tensor)
            actual = length (accessIndices access)
        if expected == actual
            then Right ()
            else Left (TensorRankMismatch expected actual)
        mapM_ verifyAccessIterator (accessIndices access)
    verifyAccessIterator (ConstantIndex _) = Right ()
    verifyAccessIterator (IteratorIndex identifier)
        | any ((== identifier) . iteratorId) iterators = Right ()
        | otherwise = Left (UnknownIterator (iteratorIdIndex identifier))
    addArgument (types, accumulatorCount) argument = case scalarArgumentValue argument of
        value@(ScalarValueId scalarOwner _)
            | scalarOwner /= owner -> Left (ForeignId "scalar")
            | otherwise ->
                let count = case scalarArgumentKind argument of
                        InputElement _ -> accumulatorCount
                        Accumulator    -> accumulatorCount + 1
                 in Right ((value, scalarArgumentType argument) : types, count)
    addOperation types scalarOperation = do
        mapM_ (knownScalar types) (scalarOperationOperands (scalarOperationKind scalarOperation))
        pure ((scalarOperationResult scalarOperation, scalarOperationResultType scalarOperation) : types)
    knownScalar types identifier =
        maybe (Left (UnknownScalar (scalarValueIdIndex identifier))) (const (Right ())) (lookup identifier types)
    verifyReductionContract accumulatorCount resultType =
        case (hasReduction operation, computeInit operation, accumulatorCount) of
            (True, Nothing, _) -> Left ReductionInitRequired
            (True, Just _, 0) -> Left ReductionAccumulatorRequired
            (True, Just initialValue, 1)
                | scalarLiteralType initialValue /= resultType -> Left ScalarResultMustMatchInit
                | otherwise -> Right ()
            (True, Just _, _) -> Left DuplicateAccumulator
            (False, Just _, _) -> Left ReductionInitWithoutReduction
            (False, Nothing, 0) -> Right ()
            (False, Nothing, _) -> Left AccumulatorWithoutReduction

foldEither :: (state -> value -> Either error state) -> state -> [value] -> Either error state
foldEither _ state [] = Right state
foldEither step state (value : rest) = do
    nextState <- step state value
    foldEither step nextState rest

atMay :: Int -> [value] -> Maybe value
atMay identifierIndex values
    | identifierIndex < 0 = Nothing
    | otherwise = case drop identifierIndex values of
        value : _ -> Just value
        []        -> Nothing

ownerCounter :: IORef Word64
ownerCounter = unsafePerformIO (newIORef 1)
{-# NOINLINE ownerCounter #-}

freshOwner :: String -> Owner
freshOwner name = name `seq` unsafePerformIO (atomicModifyIORef' ownerCounter allocate)
  where
    allocate current = (current + 1, Owner current)
{-# NOINLINE freshOwner #-}
