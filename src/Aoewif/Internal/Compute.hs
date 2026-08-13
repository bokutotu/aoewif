{-# LANGUAGE AllowAmbiguousTypes    #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RoleAnnotations        #-}
{-# LANGUAGE ScopedTypeVariables    #-}
{-# LANGUAGE TypeFamilies           #-}
{-# LANGUAGE TypeFamilyDependencies #-}

module Aoewif.Internal.Compute (
    F32,
    Boolean,
    Index,
    Spatial,
    Reduction,
    R1,
    R2,
    R3,
    Name,
    Dim,
    Tensor,
    Axis,
    Expr,
    Computation,
    Entry,
    Program,
    ProgramM,
    Axes,
    IsShape,
    IsIndices,
    Comparison (..),
    ComputeError (..),
    program,
    dim,
    staticDim,
    input,
    compute,
    entry,
    (!),
    f32,
    boolean,
    index,
    indexLiteral,
    (.+.),
    (.-.),
    (.*.),
    (./.),
    fma,
    minimum,
    maximum,
    exp,
    log,
    compare,
    select,
    foldOver,
    named,
    withProgram,
    axisIndexId,
) where

import qualified Aoewif.Internal.Compute.IR as IR
import           Data.Char                  (isAsciiLower, isAsciiUpper,
                                             isDigit)
import           Data.Kind                  (Type)
import           Data.Proxy                 (Proxy (..))
import           Data.Word                  (Word64)
import           GHC.OverloadedLabels       (IsLabel (..))
import           GHC.TypeLits               (KnownSymbol, symbolVal)
import           Prelude                    hiding (compare, exp, log, maximum,
                                             minimum)

data F32

data Boolean

data Index

data Spatial

data Reduction

data R1

data R2

data R3

newtype Name = Name String

instance (KnownSymbol label) => IsLabel label Name where
    fromLabel = Name (symbolVal (Proxy @label))

newtype Dim = Dim IR.Dim

newtype Tensor scope rank element = Tensor IR.Tensor

type role Tensor nominal nominal nominal

newtype Axis scope kind = Axis IR.IndexVar

type role Axis nominal nominal

data Expr scope element where
    F32Expression :: Float -> Expr scope F32
    BooleanExpression :: Bool -> Expr scope Boolean
    IndexExpression :: Axis scope kind -> Expr scope Index
    IndexLiteralExpression :: Word64 -> Expr scope Index
    LoadExpression :: IR.Tensor -> [IR.IndexExpression] -> Expr scope F32
    AddExpression :: Expr scope F32 -> Expr scope F32 -> Expr scope F32
    SubExpression :: Expr scope F32 -> Expr scope F32 -> Expr scope F32
    MulExpression :: Expr scope F32 -> Expr scope F32 -> Expr scope F32
    DivExpression :: Expr scope F32 -> Expr scope F32 -> Expr scope F32
    FmaExpression :: Expr scope F32 -> Expr scope F32 -> Expr scope F32 -> Expr scope F32
    MinExpression :: Expr scope F32 -> Expr scope F32 -> Expr scope F32
    MaxExpression :: Expr scope F32 -> Expr scope F32 -> Expr scope F32
    ExpExpression :: Expr scope F32 -> Expr scope F32
    LogExpression :: Expr scope F32 -> Expr scope F32
    CompareExpression :: Comparison -> Expr scope element -> Expr scope element -> Expr scope Boolean
    SelectExpression :: Expr scope Boolean -> Expr scope element -> Expr scope element -> Expr scope element
    FoldExpression :: Dim -> Float -> (Axis scope Reduction -> Expr scope F32 -> Expr scope F32) -> Expr scope F32
    AccumulatorExpression :: IR.AccumulatorId -> Expr scope F32
    NamedExpression :: String -> Expr scope element -> Expr scope element

type role Expr nominal nominal

data Computation scope (axes :: Type -> Type) rank element where
    Computation :: IR.Compute -> axes scope -> Computation scope axes rank element

type role Computation nominal nominal nominal nominal

data Entry scope (axes :: Type -> Type) where
    Entry :: IR.Compute -> axes scope -> Entry scope axes

type role Entry nominal nominal

data Program (axes :: Type -> Type) where
    Program :: IR.Program -> IR.Compute -> axes scope -> Program axes

data ProgramState = ProgramState
    { stateSymbols         :: [IR.Symbol]
    , stateInputs          :: [IR.Tensor]
    , stateComputes        :: [IR.Compute]
    , stateNextTensor      :: !Int
    , stateNextIndex       :: !Int
    , stateNextAccumulator :: !Int
    , stateNextNode        :: !Int
    }

newtype ProgramM scope value = ProgramM
    { runProgramM :: ProgramState -> Either ComputeError (value, ProgramState)
    }

type role ProgramM nominal representational

instance Functor (ProgramM scope) where
    fmap transform (ProgramM action) = ProgramM $ \state -> do
        (value, nextState) <- action state
        pure (transform value, nextState)

instance Applicative (ProgramM scope) where
    pure value = ProgramM $ \state -> Right (value, state)
    ProgramM functionAction <*> ProgramM valueAction = ProgramM $ \state -> do
        (function, functionState) <- functionAction state
        (value, valueState) <- valueAction functionState
        pure (function value, valueState)

instance Monad (ProgramM scope) where
    ProgramM action >>= next = ProgramM $ \state -> do
        (value, nextState) <- action state
        runProgramM (next value) nextState

type family Axes scope rank = result | result -> scope rank where
    Axes scope R1 = Axis scope Spatial
    Axes scope R2 = (Axis scope Spatial, Axis scope Spatial)
    Axes scope R3 = (Axis scope Spatial, Axis scope Spatial, Axis scope Spatial)

class IsShape shape rank | shape -> rank where
    shapeDimensions :: shape -> [Dim]
    createSpatialAxes :: forall scope. shape -> ProgramM scope (Axes scope rank)
    spatialIndices :: Axes scope rank -> [IR.IndexVar]

instance IsShape Dim R1 where
    shapeDimensions dimension = [dimension]
    createSpatialAxes (Dim extent) = Axis <$> freshIndex "axis0" extent
    spatialIndices (Axis axis) = [axis]

instance IsShape (Dim, Dim) R2 where
    shapeDimensions (first, second) = [first, second]
    createSpatialAxes (Dim first, Dim second) = do
        firstAxis <- Axis <$> freshIndex "axis0" first
        secondAxis <- Axis <$> freshIndex "axis1" second
        pure (firstAxis, secondAxis)
    spatialIndices (Axis first, Axis second) = [first, second]

instance IsShape (Dim, Dim, Dim) R3 where
    shapeDimensions (first, second, third) = [first, second, third]
    createSpatialAxes (Dim first, Dim second, Dim third) = do
        firstAxis <- Axis <$> freshIndex "axis0" first
        secondAxis <- Axis <$> freshIndex "axis1" second
        thirdAxis <- Axis <$> freshIndex "axis2" third
        pure (firstAxis, secondAxis, thirdAxis)
    spatialIndices (Axis first, Axis second, Axis third) = [first, second, third]

class IsIndex value (scope :: Type) | value -> scope where
    indexExpression :: value -> IR.IndexExpression

instance IsIndex (Axis scope kind) scope where
    indexExpression (Axis axis) = IR.VariableIndex axis

newtype ConstantIndex scope = ConstantIndex Word64

type role ConstantIndex nominal

instance IsIndex (ConstantIndex scope) scope where
    indexExpression (ConstantIndex value) = IR.ConstantIndex value

class IsIndices rank indices (scope :: Type) where
    indexExpressions :: indices -> [IR.IndexExpression]

instance (IsIndex indexValue scope) => IsIndices R1 indexValue scope where
    indexExpressions value = [indexExpression value]

instance (IsIndex first scope, IsIndex second scope) => IsIndices R2 (first, second) scope where
    indexExpressions (first, second) = [indexExpression first, indexExpression second]

instance
    (IsIndex first scope, IsIndex second scope, IsIndex third scope) =>
    IsIndices R3 (first, second, third) scope
    where
    indexExpressions (first, second, third) =
        [indexExpression first, indexExpression second, indexExpression third]

data Comparison
    = Equal
    | NotEqual
    | LessThan
    | LessEqual
    | GreaterThan
    | GreaterEqual
    deriving stock (Eq, Show)

data ComputeError
    = DimensionMismatch
    | ConstantIndexRequiresStaticDimension
    | ConstantIndexOutOfBounds Word64 Word64
    | InvalidFunctionIdentifier String
    deriving stock (Eq, Show)

program ::
    Name ->
    (forall scope. ProgramM scope (Entry scope axes)) ->
    Either ComputeError (Program axes)
program (Name name) action
    | not (validIdentifier name) = Left (InvalidFunctionIdentifier name)
    | otherwise = do
        (Entry output axes, finalState) <- runProgramM action initialState
        let computeProgram =
                IR.Program
                    { IR.programName = name
                    , IR.programSymbols = stateSymbols finalState
                    , IR.programInputs = stateInputs finalState
                    , IR.programComputes = stateComputes finalState
                    , IR.programOutput = IR.computeResult output
                    }
        pure (Program computeProgram output axes)
  where
    initialState =
        ProgramState
            { stateSymbols = []
            , stateInputs = []
            , stateComputes = []
            , stateNextTensor = 0
            , stateNextIndex = 0
            , stateNextAccumulator = 0
            , stateNextNode = 0
            }

dim :: Name -> ProgramM scope Dim
dim (Name name) = ProgramM $ \state ->
    let identifier = IR.SymbolId (length (stateSymbols state))
        value = IR.Symbol identifier name
     in Right (Dim (IR.SymbolDim identifier), state{stateSymbols = stateSymbols state ++ [value]})

staticDim :: Word64 -> Dim
staticDim = Dim . IR.StaticDim

input ::
    forall element shape scope rank.
    (element ~ F32, IsShape shape rank) =>
    Name ->
    shape ->
    ProgramM scope (Tensor scope rank element)
input (Name name) shape = ProgramM $ \state ->
    let identifier = IR.TensorId (stateNextTensor state)
        dimensions = map (\(Dim dimension) -> dimension) (shapeDimensions shape)
        tensor = IR.Tensor identifier name dimensions (IR.InputTensor (length (stateInputs state)))
        nextState =
            state
                { stateInputs = stateInputs state ++ [tensor]
                , stateNextTensor = stateNextTensor state + 1
                }
     in Right (Tensor tensor, nextState)

compute ::
    forall shape scope rank axes.
    (IsShape shape rank) =>
    Name ->
    shape ->
    (Axes scope rank -> (axes scope, Expr scope F32)) ->
    ProgramM scope (Computation scope axes rank F32)
compute (Name name) shape build = do
    axes <- createSpatialAxes shape
    let (exportedAxes, expression) = build axes
    body <- reifyExpression expression
    computation <- appendCompute name (spatialIndices @shape @rank axes) body
    pure (Computation computation exportedAxes)

entry :: Computation scope axes rank element -> ProgramM scope (Entry scope axes)
entry (Computation computation axes) = pure (Entry computation axes)

(!) ::
    forall rank indices scope.
    (IsIndices rank indices scope) =>
    Tensor scope rank F32 ->
    indices ->
    Expr scope F32
Tensor tensor ! indices = LoadExpression tensor (indexExpressions @rank @indices @scope indices)

infixl 9 !

f32 :: Float -> Expr scope F32
f32 = F32Expression

boolean :: Bool -> Expr scope Boolean
boolean = BooleanExpression

index :: Axis scope kind -> Expr scope Index
index = IndexExpression

indexLiteral :: Word64 -> Expr scope Index
indexLiteral = IndexLiteralExpression

(.+.) :: Expr scope F32 -> Expr scope F32 -> Expr scope F32
(.+.) = AddExpression

(.-.) :: Expr scope F32 -> Expr scope F32 -> Expr scope F32
(.-.) = SubExpression

(.*.) :: Expr scope F32 -> Expr scope F32 -> Expr scope F32
(.*.) = MulExpression

(./.) :: Expr scope F32 -> Expr scope F32 -> Expr scope F32
(./.) = DivExpression

infixl 6 .+., .-.
infixl 7 .*., ./.

fma :: Expr scope F32 -> Expr scope F32 -> Expr scope F32 -> Expr scope F32
fma = FmaExpression

minimum :: Expr scope F32 -> Expr scope F32 -> Expr scope F32
minimum = MinExpression

maximum :: Expr scope F32 -> Expr scope F32 -> Expr scope F32
maximum = MaxExpression

exp :: Expr scope F32 -> Expr scope F32
exp = ExpExpression

log :: Expr scope F32 -> Expr scope F32
log = LogExpression

compare :: Comparison -> Expr scope element -> Expr scope element -> Expr scope Boolean
compare = CompareExpression

select ::
    Expr scope Boolean ->
    Expr scope element ->
    Expr scope element ->
    Expr scope element
select = SelectExpression

foldOver ::
    Dim ->
    Float ->
    (Axis scope Reduction -> Expr scope F32 -> Expr scope F32) ->
    Expr scope F32
foldOver = FoldExpression

named :: Name -> Expr scope element -> Expr scope element
named (Name name) = NamedExpression name

withProgram ::
    Program axes ->
    (forall scope. IR.Program -> IR.Compute -> axes scope -> result) ->
    result
withProgram (Program computeProgram computation axes) consume = consume computeProgram computation axes

axisIndexId :: Axis scope kind -> IR.IndexId
axisIndexId (Axis axis) = IR.indexId axis

appendCompute :: String -> [IR.IndexVar] -> IR.Expression -> ProgramM scope IR.Compute
appendCompute name indices body = ProgramM $ \state ->
    let computationId = IR.ComputeId (length (stateComputes state))
        tensor =
            IR.Tensor
                { IR.tensorId = IR.TensorId (stateNextTensor state)
                , IR.tensorName = name
                , IR.tensorShape = map IR.indexExtent indices
                , IR.tensorKind = IR.ResultTensor computationId
                }
        computation =
            IR.Compute
                { IR.computeName = name
                , IR.computeIndices = indices
                , IR.computeResult = tensor
                , IR.computeBody = body
                }
        nextState =
            state
                { stateComputes = stateComputes state ++ [computation]
                , stateNextTensor = stateNextTensor state + 1
                }
     in Right (computation, nextState)

freshIndex :: String -> IR.Dim -> ProgramM scope IR.IndexVar
freshIndex name extent = ProgramM $ \state ->
    let value = IR.IndexVar (IR.IndexId (stateNextIndex state)) name extent
     in Right (value, state{stateNextIndex = stateNextIndex state + 1})

freshAccumulator :: ProgramM scope IR.AccumulatorId
freshAccumulator = ProgramM $ \state ->
    let value = IR.AccumulatorId (stateNextAccumulator state)
     in Right (value, state{stateNextAccumulator = stateNextAccumulator state + 1})

freshNode :: ProgramM scope IR.ComputeNodeId
freshNode = ProgramM $ \state ->
    let value = IR.ComputeNodeId (stateNextNode state)
     in Right (value, state{stateNextNode = stateNextNode state + 1})

reifyExpression :: Expr scope element -> ProgramM scope IR.Expression
reifyExpression expression = case expression of
    F32Expression value -> pure (IR.LiteralExpression (IR.F32Literal value))
    BooleanExpression value -> pure (IR.LiteralExpression (IR.BoolLiteral value))
    IndexExpression (Axis axis) -> pure (IR.IndexValueExpression axis)
    IndexLiteralExpression value -> pure (IR.LiteralExpression (IR.IndexLiteral value))
    LoadExpression tensor indices ->
        IR.ReadExpression tensor
            <$> mapM reifyAccessIndex (zip (IR.tensorShape tensor) indices)
    AddExpression lhs rhs -> reifyBinary IR.AddExpression lhs rhs
    SubExpression lhs rhs -> reifyBinary IR.SubExpression lhs rhs
    MulExpression lhs rhs -> reifyBinary IR.MulExpression lhs rhs
    DivExpression lhs rhs -> reifyBinary IR.DivExpression lhs rhs
    FmaExpression lhs rhs accumulator ->
        IR.FmaExpression <$> reifyExpression lhs <*> reifyExpression rhs <*> reifyExpression accumulator
    MinExpression lhs rhs -> reifyBinary IR.MinExpression lhs rhs
    MaxExpression lhs rhs -> reifyBinary IR.MaxExpression lhs rhs
    ExpExpression value -> IR.ExpExpression <$> reifyExpression value
    LogExpression value -> IR.LogExpression <$> reifyExpression value
    CompareExpression predicate lhs rhs ->
        IR.CompareExpression (reifyComparison predicate) <$> reifyExpression lhs <*> reifyExpression rhs
    SelectExpression condition trueValue falseValue ->
        IR.SelectExpression
            <$> reifyExpression condition
            <*> reifyExpression trueValue
            <*> reifyExpression falseValue
    FoldExpression (Dim extent) initialValue buildBody -> do
        reductionIndex <- freshIndex "reduction" extent
        accumulator <- freshAccumulator
        body <- reifyExpression (buildBody (Axis reductionIndex) (AccumulatorExpression accumulator))
        pure (IR.FoldExpression reductionIndex initialValue accumulator body)
    AccumulatorExpression identifier -> pure (IR.AccumulatorExpression identifier)
    NamedExpression name value -> do
        identifier <- freshNode
        IR.NamedExpression identifier name <$> reifyExpression value
  where
    reifyBinary constructor lhs rhs = constructor <$> reifyExpression lhs <*> reifyExpression rhs
    reifyAccessIndex (dimension, accessIndex@(IR.VariableIndex variable))
        | dimension == IR.indexExtent variable = pure accessIndex
        | otherwise = throwCompute DimensionMismatch
    reifyAccessIndex (IR.StaticDim extent, accessIndex@(IR.ConstantIndex value))
        | value < extent = pure accessIndex
        | otherwise = throwCompute (ConstantIndexOutOfBounds value extent)
    reifyAccessIndex (IR.SymbolDim _, IR.ConstantIndex _) = throwCompute ConstantIndexRequiresStaticDimension

throwCompute :: ComputeError -> ProgramM scope value
throwCompute computeError = ProgramM $ \_ -> Left computeError

reifyComparison :: Comparison -> IR.ComparePredicate
reifyComparison comparison = case comparison of
    Equal        -> IR.Equal
    NotEqual     -> IR.NotEqual
    LessThan     -> IR.Less
    LessEqual    -> IR.LessEqual
    GreaterThan  -> IR.Greater
    GreaterEqual -> IR.GreaterEqual

validIdentifier :: String -> Bool
validIdentifier (first : rest) = validFirst first && all validRest rest
  where
    validFirst character = character == '_' || isAsciiLower character || isAsciiUpper character
    validRest character = validFirst character || isDigit character
validIdentifier [] = False
