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
    Kernel,
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
    withProgram,
    axisIteratorId,
) where

import           Aoewif.Internal.IR   (ComputeError (..))
import qualified Aoewif.Internal.IR   as IR
import           Data.Kind            (Type)
import           Data.Proxy           (Proxy (..))
import           Data.Word            (Word64)
import           GHC.OverloadedLabels (IsLabel (..))
import           GHC.TypeLits         (KnownSymbol, symbolVal)
import           Prelude              hiding (compare, exp, log, maximum,
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

newtype Tensor scope rank element = Tensor IR.TensorValueId

type role Tensor nominal nominal nominal

newtype Axis scope kind = Axis IR.IteratorId

type role Axis nominal nominal

data Expr scope element where
    F32Expression :: Float -> Expr scope F32
    BooleanExpression :: Bool -> Expr scope Boolean
    IndexExpression :: Axis scope kind -> Expr scope Index
    IndexLiteralExpression :: Word64 -> Expr scope Index
    LoadExpression :: Tensor scope rank F32 -> [IR.IndexExpr] -> Expr scope F32
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
    ScalarExpression :: IR.ScalarValueId -> Expr scope element

type role Expr nominal nominal

data Kernel scope (axes :: Type -> Type) rank element where
    Kernel :: IR.TensorValueId -> axes scope -> Kernel scope axes rank element

type role Kernel nominal nominal nominal nominal

data Entry scope (axes :: Type -> Type) where
    Entry :: IR.TensorValueId -> axes scope -> Entry scope axes

type role Entry nominal nominal

data Program (axes :: Type -> Type) where
    Program :: IR.ComputeFunction -> IR.ComputeOpId -> axes scope -> Program axes

newtype ProgramM scope value = ProgramM
    { runProgramM :: IR.FunctionBuilder value
    }

type role ProgramM nominal representational

instance Functor (ProgramM scope) where
    fmap transform (ProgramM action) = ProgramM (transform <$> action)

instance Applicative (ProgramM scope) where
    pure = ProgramM . pure
    ProgramM functionAction <*> ProgramM valueAction = ProgramM (functionAction <*> valueAction)

instance Monad (ProgramM scope) where
    ProgramM action >>= next = ProgramM (action >>= runProgramM . next)

type family Axes scope rank = result | result -> scope rank where
    Axes scope R1 = Axis scope Spatial
    Axes scope R2 = (Axis scope Spatial, Axis scope Spatial)
    Axes scope R3 = (Axis scope Spatial, Axis scope Spatial, Axis scope Spatial)

class IsShape shape rank | shape -> rank where
    shapeDimensions :: shape -> [Dim]
    createSpatialAxes :: forall scope. shape -> IR.ComputeBuilder (Axes scope rank)

instance IsShape Dim R1 where
    shapeDimensions dimension = [dimension]
    createSpatialAxes (Dim extent) = Axis <$> IR.parallel "axis0" extent

instance IsShape (Dim, Dim) R2 where
    shapeDimensions (first, second) = [first, second]
    createSpatialAxes (Dim first, Dim second) = do
        firstAxis <- Axis <$> IR.parallel "axis0" first
        secondAxis <- Axis <$> IR.parallel "axis1" second
        pure (firstAxis, secondAxis)

instance IsShape (Dim, Dim, Dim) R3 where
    shapeDimensions (first, second, third) = [first, second, third]
    createSpatialAxes (Dim first, Dim second, Dim third) = do
        firstAxis <- Axis <$> IR.parallel "axis0" first
        secondAxis <- Axis <$> IR.parallel "axis1" second
        thirdAxis <- Axis <$> IR.parallel "axis2" third
        pure (firstAxis, secondAxis, thirdAxis)

class IsIndex value (scope :: Type) | value -> scope where
    indexExpression :: value -> IR.IndexExpr

instance IsIndex (Axis scope kind) scope where
    indexExpression (Axis identifier) = IR.IteratorIndex identifier

newtype ConstantIndex scope = ConstantIndex Word64

type role ConstantIndex nominal

instance IsIndex (ConstantIndex scope) scope where
    indexExpression (ConstantIndex value) = IR.ConstantIndex value

class IsIndices rank indices (scope :: Type) where
    indexExpressions :: indices -> [IR.IndexExpr]

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

program ::
    Name ->
    (forall scope. ProgramM scope (Entry scope axes)) ->
    Either ComputeError (Program axes)
program (Name name) action = do
    (Entry output axes, function) <- IR.buildComputeFunctionWith name (runProgramM action)
    tensor <- maybe (Left (IR.UnknownTensor (IR.tensorValueIdIndex output))) Right (IR.lookupTensor output function)
    operation <- case IR.tensorDefinition tensor of
        IR.ComputeResult identifier -> Right identifier
        IR.InputTensor _ -> Left (IR.UnknownTensor (IR.tensorValueIdIndex output))
    pure (Program function operation axes)

dim :: Name -> ProgramM scope Dim
dim (Name name) = ProgramM (Dim . IR.SymbolDim <$> IR.symbol name)

staticDim :: Word64 -> Dim
staticDim = Dim . IR.StaticDim

input ::
    forall element shape scope rank.
    (element ~ F32, IsShape shape rank) =>
    Name ->
    shape ->
    ProgramM scope (Tensor scope rank element)
input (Name name) shape = ProgramM $ do
    let dimensions = map (\(Dim dimension) -> dimension) (shapeDimensions shape)
    Tensor <$> IR.input name (IR.tensorTypeF32 dimensions)

compute ::
    forall shape scope rank axes.
    (IsShape shape rank) =>
    Name ->
    shape ->
    (Axes scope rank -> (axes scope, Expr scope F32)) ->
    ProgramM scope (Kernel scope axes rank F32)
compute (Name name) shape build = ProgramM $ do
    (axes, tensor) <- IR.computeWith name $ do
        spatialAxes <- createSpatialAxes shape
        let (exportedAxes, body) = build spatialAxes
        result <- lowerExpression body
        pure (exportedAxes, result)
    pure (Kernel tensor axes)

entry :: Kernel scope axes rank element -> ProgramM scope (Entry scope axes)
entry (Kernel tensor axes) = ProgramM $ do
    IR.markOutput tensor
    pure (Entry tensor axes)

(!) ::
    forall rank indices scope.
    (IsIndices rank indices scope) =>
    Tensor scope rank F32 ->
    indices ->
    Expr scope F32
Tensor tensor ! indices =
    LoadExpression (Tensor tensor) (indexExpressions @rank @indices @scope indices)

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

withProgram ::
    Program axes ->
    (forall scope. IR.ComputeFunction -> IR.ComputeOpId -> axes scope -> result) ->
    result
withProgram (Program function operation axes) consume = consume function operation axes

axisIteratorId :: Axis scope kind -> IR.IteratorId
axisIteratorId (Axis identifier) = identifier

lowerExpression :: Expr scope element -> IR.ComputeBuilder IR.ScalarValueId
lowerExpression expression = case expression of
    F32Expression value -> IR.constant (IR.F32Literal value)
    BooleanExpression value -> IR.constant (IR.BoolLiteral value)
    IndexExpression (Axis identifier) -> IR.index identifier
    IndexLiteralExpression value -> IR.constant (IR.IndexLiteral value)
    LoadExpression (Tensor tensor) indices -> IR.readTensor tensor indices
    AddExpression lhs rhs -> lowerBinary IR.add lhs rhs
    SubExpression lhs rhs -> lowerBinary IR.sub lhs rhs
    MulExpression lhs rhs -> lowerBinary IR.mul lhs rhs
    DivExpression lhs rhs -> lowerBinary IR.divide lhs rhs
    FmaExpression lhs rhs accumulator -> do
        loweredLhs <- lowerExpression lhs
        loweredRhs <- lowerExpression rhs
        loweredAccumulator <- lowerExpression accumulator
        IR.fma loweredLhs loweredRhs loweredAccumulator
    MinExpression lhs rhs -> lowerBinary IR.minimum lhs rhs
    MaxExpression lhs rhs -> lowerBinary IR.maximum lhs rhs
    ExpExpression value -> lowerExpression value >>= IR.expScalar
    LogExpression value -> lowerExpression value >>= IR.logScalar
    CompareExpression predicate lhs rhs -> do
        loweredLhs <- lowerExpression lhs
        loweredRhs <- lowerExpression rhs
        IR.compare (lowerComparison predicate) loweredLhs loweredRhs
    SelectExpression condition trueValue falseValue -> do
        loweredCondition <- lowerExpression condition
        loweredTrueValue <- lowerExpression trueValue
        loweredFalseValue <- lowerExpression falseValue
        IR.selectScalar loweredCondition loweredTrueValue loweredFalseValue
    FoldExpression (Dim extent) initialValue body -> do
        reductionAxis <- Axis <$> IR.reduction "reduction" extent
        accumulator <- IR.reductionInit (IR.F32Literal initialValue)
        lowerExpression (body reductionAxis (ScalarExpression accumulator))
    ScalarExpression identifier -> pure identifier
  where
    lowerBinary operation lhs rhs = do
        loweredLhs <- lowerExpression lhs
        loweredRhs <- lowerExpression rhs
        operation loweredLhs loweredRhs

lowerComparison :: Comparison -> IR.ComparePredicate
lowerComparison comparison = case comparison of
    Equal        -> IR.Equal
    NotEqual     -> IR.NotEqual
    LessThan     -> IR.Less
    LessEqual    -> IR.LessEqual
    GreaterThan  -> IR.Greater
    GreaterEqual -> IR.GreaterEqual
