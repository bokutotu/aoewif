module Aoewif.Internal.IR (
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
    TensorType,
    tensorTypeF32,
    tensorShape,
    tensorRank,
    Symbol (..),
    TensorDefinition (..),
    TensorValue (..),
    IteratorKind (..),
    Iterator (..),
    IndexExpr (..),
    TensorAccess,
    accessTensor,
    accessIndices,
    ScalarLiteral (..),
    scalarLiteralType,
    ComparePredicate (..),
    ScalarOperationKind (..),
    ScalarOperation (..),
    ScalarArgumentKind (..),
    ScalarArgument (..),
    ScalarRegion (..),
    ReductionPolicy (..),
    ComputeOp (..),
    ComputeFunction,
    functionName,
    functionSymbols,
    functionTensors,
    functionOperations,
    functionOutputs,
    lookupTensor,
    lookupOperation,
    inputTensors,
    ComputeError (..),
    FunctionBuilder,
    ComputeBuilder,
    buildComputeFunctionWith,
    symbol,
    input,
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

import           Data.List (find)
import           Data.Word (Word64)
import           Prelude   hiding (compare, maximum, minimum)

newtype SymbolId = SymbolId Int
    deriving stock (Eq, Ord)

newtype TensorValueId = TensorValueId Int
    deriving stock (Eq, Ord)

newtype ComputeOpId = ComputeOpId Int
    deriving stock (Eq, Ord)

newtype IteratorId = IteratorId Int
    deriving stock (Eq, Ord)

newtype ScalarValueId = ScalarValueId Int
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
symbolIdIndex (SymbolId identifierIndex) = identifierIndex

tensorValueIdIndex :: TensorValueId -> Int
tensorValueIdIndex (TensorValueId identifierIndex) = identifierIndex

computeOpIdIndex :: ComputeOpId -> Int
computeOpIdIndex (ComputeOpId identifierIndex) = identifierIndex

iteratorIdIndex :: IteratorId -> Int
iteratorIdIndex (IteratorId identifierIndex) = identifierIndex

scalarValueIdIndex :: ScalarValueId -> Int
scalarValueIdIndex (ScalarValueId identifierIndex) = identifierIndex

data Dim
    = StaticDim Word64
    | SymbolDim SymbolId
    deriving stock (Eq, Show)

data ScalarType
    = F32
    | Bool
    | Index
    deriving stock (Eq, Show)

newtype TensorType = TensorType [Dim]
    deriving stock (Eq, Show)

tensorTypeF32 :: [Dim] -> TensorType
tensorTypeF32 = TensorType

tensorShape :: TensorType -> [Dim]
tensorShape (TensorType shape) = shape

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

data TensorAccess = TensorAccess TensorValueId [IndexExpr] ScalarValueId
    deriving stock (Eq, Show)

accessTensor :: TensorAccess -> TensorValueId
accessTensor (TensorAccess tensor _ _) = tensor

accessIndices :: TensorAccess -> [IndexExpr]
accessIndices (TensorAccess _ indices _) = indices

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

data ComputeFunction = ComputeFunction
    { internalFunctionName       :: String
    , internalFunctionSymbols    :: [Symbol]
    , internalFunctionTensors    :: [TensorValue]
    , internalFunctionOperations :: [ComputeOp]
    , internalFunctionOutputs    :: [TensorValueId]
    }
    deriving stock (Eq, Show)

functionName :: ComputeFunction -> String
functionName = internalFunctionName

functionSymbols :: ComputeFunction -> [Symbol]
functionSymbols = internalFunctionSymbols

functionTensors :: ComputeFunction -> [TensorValue]
functionTensors = internalFunctionTensors

functionOperations :: ComputeFunction -> [ComputeOp]
functionOperations = internalFunctionOperations

functionOutputs :: ComputeFunction -> [TensorValueId]
functionOutputs = internalFunctionOutputs

lookupTensor :: TensorValueId -> ComputeFunction -> Maybe TensorValue
lookupTensor identifier function = atMay (tensorValueIdIndex identifier) (functionTensors function)

lookupOperation :: ComputeOpId -> ComputeFunction -> Maybe ComputeOp
lookupOperation identifier function = atMay (computeOpIdIndex identifier) (functionOperations function)

inputTensors :: ComputeFunction -> [TensorValue]
inputTensors = filter isInput . functionTensors
  where
    isInput tensor = case tensorDefinition tensor of
        InputTensor _   -> True
        ComputeResult _ -> False

data ComputeError
    = UnknownSymbol Int
    | UnknownTensor Int
    | UnknownIterator Int
    | UnknownScalar Int
    | TensorRankMismatch Int Int
    | DimensionMismatch
    | ConstantIndexRequiresStaticDimension
    | ConstantIndexOutOfBounds Word64 Word64
    | ReductionInitRequired
    | ReductionInitWithoutReduction
    | DuplicateAccumulator
    | ScalarTypeMismatch ScalarType ScalarType
    | ScalarOperandsMustMatch
    | ScalarResultMustMatchInit
    | OutputRequired
    | OutputAlreadyMarked Int
    deriving stock (Eq, Show)

data FunctionState = FunctionState
    { stateFunction     :: ComputeFunction
    , stateNextIterator :: Int
    , stateNextScalar   :: Int
    }

newtype FunctionBuilder value = FunctionBuilder
    { runFunctionBuilder :: FunctionState -> Either ComputeError (value, FunctionState)
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
    , computeBuilderArguments   :: [ScalarArgument]
    , computeBuilderOperations  :: [ScalarOperation]
    , computeBuilderScalarTypes :: [(ScalarValueId, ScalarType)]
    }

newtype ComputeBuilder value = ComputeBuilder
    { runComputeBuilder :: ComputeState -> Either ComputeError (value, ComputeState)
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

buildComputeFunctionWith ::
    String ->
    FunctionBuilder value ->
    Either ComputeError (value, ComputeFunction)
buildComputeFunctionWith name builder = do
    let initialFunction =
            ComputeFunction
                { internalFunctionName = name
                , internalFunctionSymbols = []
                , internalFunctionTensors = []
                , internalFunctionOperations = []
                , internalFunctionOutputs = []
                }
        initialState = FunctionState initialFunction 0 0
    (value, finalState) <- runFunctionBuilder builder initialState
    let function = stateFunction finalState
    if null (functionOutputs function)
        then Left OutputRequired
        else pure (value, function)

symbol :: String -> FunctionBuilder SymbolId
symbol name = FunctionBuilder $ \state -> do
    let function = stateFunction state
        identifier = SymbolId (length (functionSymbols function))
        newSymbol = Symbol identifier name
        nextFunction = function{internalFunctionSymbols = functionSymbols function <> [newSymbol]}
    pure (identifier, state{stateFunction = nextFunction})

input :: String -> TensorType -> FunctionBuilder TensorValueId
input name tensorType = FunctionBuilder $ \state -> do
    shape <- mapM (resolveDim state) (tensorShape tensorType)
    let resolvedType = tensorTypeF32 shape
        function = stateFunction state
        identifier = TensorValueId (length (functionTensors function))
        inputIndex = length (inputTensors function)
        tensor = TensorValue identifier name resolvedType (InputTensor inputIndex)
        nextFunction = function{internalFunctionTensors = functionTensors function <> [tensor]}
    pure (identifier, state{stateFunction = nextFunction})

computeWith ::
    String ->
    ComputeBuilder (value, ScalarValueId) ->
    FunctionBuilder (value, TensorValueId)
computeWith name body = FunctionBuilder $ \state -> do
    let initialComputeState =
            ComputeState
                { computeFunctionState = state
                , computeBuilderName = name
                , computeBuilderIterators = []
                , computeBuilderAccesses = []
                , computeBuilderInit = Nothing
                , computeBuilderArguments = []
                , computeBuilderOperations = []
                , computeBuilderScalarTypes = []
                }
    ((value, result), finalComputeState) <- runComputeBuilder body initialComputeState
    (tensorIdentifier, finalFunctionState) <- finishCompute result finalComputeState
    pure ((value, tensorIdentifier), finalFunctionState)

markOutput :: TensorValueId -> FunctionBuilder ()
markOutput identifier = FunctionBuilder $ \state -> do
    _ <- resolveTensor identifier state
    let function = stateFunction state
    if identifier `elem` functionOutputs function
        then Left (OutputAlreadyMarked (tensorValueIdIndex identifier))
        else
            let nextFunction = function{internalFunctionOutputs = functionOutputs function <> [identifier]}
             in pure ((), state{stateFunction = nextFunction})

parallel :: String -> Dim -> ComputeBuilder IteratorId
parallel name extent = iterator name extent Parallel

reduction :: String -> Dim -> ComputeBuilder IteratorId
reduction name extent = iterator name extent Reduction

readTensor :: TensorValueId -> [IndexExpr] -> ComputeBuilder ScalarValueId
readTensor tensorIdentifier indices = ComputeBuilder $ \state -> do
    tensor <- resolveTensor tensorIdentifier (computeFunctionState state)
    let (scalar, allocatedState) = allocateScalar state
    access <- makeTensorAccess state tensor indices scalar
    let
        accessIndex = length (computeBuilderAccesses state)
        argument = ScalarArgument scalar F32 (InputElement accessIndex)
        nextState =
            allocatedState
                { computeBuilderAccesses = computeBuilderAccesses state <> [access]
                , computeBuilderArguments = computeBuilderArguments state <> [argument]
                , computeBuilderScalarTypes = (scalar, F32) : computeBuilderScalarTypes state
                }
    pure (scalar, nextState)

reductionInit :: ScalarLiteral -> ComputeBuilder ScalarValueId
reductionInit literal = ComputeBuilder $ \state -> case computeBuilderInit state of
    Just _ -> Left DuplicateAccumulator
    Nothing ->
        let (accumulator, allocatedState) = allocateScalar state
            argument = ScalarArgument accumulator (scalarLiteralType literal) Accumulator
            nextState =
                allocatedState
                    { computeBuilderInit = Just literal
                    , computeBuilderArguments = computeBuilderArguments state <> [argument]
                    , computeBuilderScalarTypes =
                        (accumulator, scalarLiteralType literal) : computeBuilderScalarTypes state
                    }
         in pure (accumulator, nextState)

constant :: ScalarLiteral -> ComputeBuilder ScalarValueId
constant literal = pushOperation (ConstantOperation literal) (scalarLiteralType literal)

index :: IteratorId -> ComputeBuilder ScalarValueId
index identifier = ComputeBuilder $ \state -> do
    computeIterator <- resolveIterator identifier state
    runComputeBuilder (pushOperation (IndexOperation (iteratorId computeIterator)) Index) state

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
    resolvedLhs <- scalarOfType lhs F32 state
    resolvedRhs <- scalarOfType rhs F32 state
    resolvedAccumulator <- scalarOfType accumulator F32 state
    runComputeBuilder (pushOperation (FmaOperation resolvedLhs resolvedRhs resolvedAccumulator) F32) state

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
    lhsType <- resolveScalarType lhs state
    rhsType <- resolveScalarType rhs state
    if lhsType /= rhsType
        then Left ScalarOperandsMustMatch
        else runComputeBuilder (pushOperation (CompareOperation predicate lhs rhs) Bool) state

selectScalar ::
    ScalarValueId ->
    ScalarValueId ->
    ScalarValueId ->
    ComputeBuilder ScalarValueId
selectScalar condition trueValue falseValue = ComputeBuilder $ \state -> do
    resolvedCondition <- scalarOfType condition Bool state
    trueType <- resolveScalarType trueValue state
    falseType <- resolveScalarType falseValue state
    if trueType /= falseType
        then Left ScalarOperandsMustMatch
        else
            runComputeBuilder
                (pushOperation (SelectOperation resolvedCondition trueValue falseValue) trueType)
                state

iterator :: String -> Dim -> IteratorKind -> ComputeBuilder IteratorId
iterator name extent kind = ComputeBuilder $ \state -> do
    resolvedExtent <- resolveDim (computeFunctionState state) extent
    let functionState = computeFunctionState state
        identifier = IteratorId (stateNextIterator functionState)
        newIterator = Iterator identifier name resolvedExtent kind
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
    Either ComputeError (TensorValueId, FunctionState)
finishCompute result state = do
    resultType <- resolveScalarType result state
    let hasReductionIterator = any ((== Reduction) . iteratorKind) (computeBuilderIterators state)
    case (hasReductionIterator, computeBuilderInit state) of
        (True, Nothing)  -> Left ReductionInitRequired
        (False, Just _)  -> Left ReductionInitWithoutReduction
        (True, Just _)   -> Right ()
        (False, Nothing) -> Right ()
    case computeBuilderInit state of
        Just initialValue
            | resultType /= scalarLiteralType initialValue -> Left ScalarResultMustMatchInit
        _ -> Right ()
    if resultType /= F32
        then Left (ScalarTypeMismatch F32 resultType)
        else do
            let functionState = computeFunctionState state
                function = stateFunction functionState
                operationIdentifier = ComputeOpId (length (functionOperations function))
                tensorIdentifier = TensorValueId (length (functionTensors function))
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
                        { internalFunctionTensors = functionTensors function <> [tensor]
                        , internalFunctionOperations = functionOperations function <> [operation]
                        }
            pure (tensorIdentifier, functionState{stateFunction = nextFunction})

binaryF32 ::
    (ScalarValueId -> ScalarValueId -> ScalarOperationKind) ->
    ScalarValueId ->
    ScalarValueId ->
    ComputeBuilder ScalarValueId
binaryF32 constructor lhs rhs = ComputeBuilder $ \state -> do
    resolvedLhs <- scalarOfType lhs F32 state
    resolvedRhs <- scalarOfType rhs F32 state
    runComputeBuilder (pushOperation (constructor resolvedLhs resolvedRhs) F32) state

unaryF32 ::
    (ScalarValueId -> ScalarOperationKind) ->
    ScalarValueId ->
    ComputeBuilder ScalarValueId
unaryF32 constructor value = ComputeBuilder $ \state -> do
    resolvedValue <- scalarOfType value F32 state
    runComputeBuilder (pushOperation (constructor resolvedValue) F32) state

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
        identifier = ScalarValueId (stateNextScalar functionState)
        nextFunctionState = functionState{stateNextScalar = stateNextScalar functionState + 1}
     in (identifier, state{computeFunctionState = nextFunctionState})

resolveDim :: FunctionState -> Dim -> Either ComputeError Dim
resolveDim _ dimension@(StaticDim _) = Right dimension
resolveDim state (SymbolDim identifier) = SymbolDim . symbolId <$> resolveSymbol identifier state

resolveSymbol :: SymbolId -> FunctionState -> Either ComputeError Symbol
resolveSymbol (SymbolId identifierIndex) state =
    maybe (Left (UnknownSymbol identifierIndex)) Right (atMay identifierIndex (functionSymbols function))
  where
    function = stateFunction state

resolveTensor :: TensorValueId -> FunctionState -> Either ComputeError TensorValue
resolveTensor (TensorValueId identifierIndex) state =
    maybe (Left (UnknownTensor identifierIndex)) Right (atMay identifierIndex (functionTensors function))
  where
    function = stateFunction state

resolveIterator :: IteratorId -> ComputeState -> Either ComputeError Iterator
resolveIterator identifier@(IteratorId identifierIndex) state =
    maybe
        (Left (UnknownIterator identifierIndex))
        Right
        (find ((== identifier) . iteratorId) (computeBuilderIterators state))

resolveScalarType :: ScalarValueId -> ComputeState -> Either ComputeError ScalarType
resolveScalarType identifier@(ScalarValueId identifierIndex) state =
    maybe (Left (UnknownScalar identifierIndex)) Right (lookup identifier (computeBuilderScalarTypes state))

scalarOfType :: ScalarValueId -> ScalarType -> ComputeState -> Either ComputeError ScalarValueId
scalarOfType value expected state = do
    actual <- resolveScalarType value state
    if actual == expected
        then Right value
        else Left (ScalarTypeMismatch expected actual)

makeTensorAccess :: ComputeState -> TensorValue -> [IndexExpr] -> ScalarValueId -> Either ComputeError TensorAccess
makeTensorAccess state tensor indices scalar = do
    let expected = tensorRank (tensorValueType tensor)
        actual = length indices
    if expected == actual
        then Right ()
        else Left (TensorRankMismatch expected actual)
    resolvedIndices <- mapM resolveIndex (zip (tensorShape (tensorValueType tensor)) indices)
    pure (TensorAccess (tensorValueId tensor) resolvedIndices scalar)
  where
    resolveIndex (dimension, IteratorIndex identifier) = do
        computeIterator <- resolveIterator identifier state
        if iteratorExtent computeIterator == dimension
            then Right (IteratorIndex (iteratorId computeIterator))
            else Left DimensionMismatch
    resolveIndex (StaticDim extent, indexExpression@(ConstantIndex indexValue))
        | indexValue < extent = Right indexExpression
        | otherwise = Left (ConstantIndexOutOfBounds indexValue extent)
    resolveIndex (SymbolDim _, ConstantIndex _) = Left ConstantIndexRequiresStaticDimension

atMay :: Int -> [value] -> Maybe value
atMay identifierIndex values
    | identifierIndex < 0 = Nothing
    | otherwise = case drop identifierIndex values of
        value : _ -> Just value
        []        -> Nothing
