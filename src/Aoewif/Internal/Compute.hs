{-# LANGUAGE RoleAnnotations #-}

module Aoewif.Internal.Compute (
    F32,
    Boolean,
    Index,
    Input,
    Output,
    Name (..),
    Dim,
    DTypeRep,
    f32,
    Tensor,
    Axis,
    Expr,
    Reducer,
    Compute,
    BlockM,
    ComputeIR (..),
    SymbolId (..),
    TensorId (..),
    BlockId (..),
    AxisId (..),
    ComputeNodeId (..),
    DimExpr (..),
    Symbol (..),
    DType (..),
    TensorKind (..),
    TensorDecl (..),
    AxisKind (..),
    AxisDecl (..),
    IndexExpr (..),
    ScalarLiteral (..),
    ComparePredicate (..),
    ScalarExpr (..),
    ReducerKind (..),
    Definition (..),
    ComputeBlock (..),
    tensorAt,
    blockAt,
    axisAt,
    program,
    dim,
    staticDim,
    input,
    output,
    block,
    spatial,
    reduction,
    define,
    reduce,
    sumOver,
    add,
    multiply,
    minimumReducer,
    maximumReducer,
    (!),
    boolean,
    index,
    indexLiteral,
    fma,
    min_,
    max_,
    exp_,
    log_,
    compare_,
    select,
    named,
    axisIdOf,
) where

import           Aoewif.Internal.Compute.IR (AxisDecl (..), AxisId (..),
                                             AxisKind (..), BlockId (..),
                                             ComparePredicate (..),
                                             ComputeBlock (..), ComputeIR (..),
                                             ComputeNodeId (..), DType (..),
                                             Definition (..), DimExpr (..),
                                             IndexExpr (..), Name (..),
                                             ReducerKind (..), ScalarExpr (..),
                                             ScalarLiteral (..), Symbol (..),
                                             SymbolId (..), TensorDecl (..),
                                             TensorId (..), TensorKind (..))
import qualified Aoewif.Internal.Compute.IR as IR
import           Data.Word                  (Word64)
import           Prelude                    hiding (exp, log, maximum, minimum)

type Dim = IR.DimExpr

data F32
data Boolean
data Index
data Input
data Output

data DTypeRep element where
    F32Rep :: DTypeRep F32

f32 :: DTypeRep F32
f32 = F32Rep

newtype Tensor access element = Tensor IR.TensorId

type role Tensor nominal nominal

newtype Axis scope = Axis IR.AxisId

type role Axis nominal

data Expr scope element where
    Expression :: IR.ScalarExpr -> Expr scope element
    ReductionExpression :: IR.ReducerKind -> IR.ScalarExpr -> [IR.AxisId] -> IR.ScalarExpr -> Expr scope F32

type role Expr nominal nominal

newtype Reducer element = Reducer IR.ReducerKind

data ComputeState = ComputeState
    { stateSymbols  :: [IR.Symbol]
    , stateTensors  :: [IR.TensorDecl]
    , stateBlocks   :: [IR.ComputeBlock]
    , stateNextAxis :: !Int
    , stateNextNode :: !Int
    }

newtype Compute value = Compute
    { runCompute :: ComputeState -> (value, ComputeState)
    }

instance Functor Compute where
    fmap transform (Compute action) = Compute $ \state ->
        let (value, nextState) = action state
         in (transform value, nextState)

instance Applicative Compute where
    pure value = Compute (value,)
    Compute functionAction <*> Compute valueAction = Compute $ \state ->
        let (function, functionState) = functionAction state
            (value, valueState) = valueAction functionState
         in (function value, valueState)

instance Monad Compute where
    Compute action >>= next = Compute $ \state ->
        let (value, nextState) = action state
         in runCompute (next value) nextState

data BlockState = BlockState
    { blockStateAxes        :: [IR.AxisDecl]
    , blockStateDefinitions :: [IR.Definition]
    , blockStateNextAxis    :: !Int
    , blockStateNextNode    :: !Int
    }

newtype BlockM scope value = BlockM
    { runBlockM :: BlockState -> (value, BlockState)
    }

type role BlockM nominal representational

instance Functor (BlockM scope) where
    fmap transform (BlockM action) = BlockM $ \state ->
        let (value, nextState) = action state
         in (transform value, nextState)

instance Applicative (BlockM scope) where
    pure value = BlockM (value,)
    BlockM functionAction <*> BlockM valueAction = BlockM $ \state ->
        let (function, functionState) = functionAction state
            (value, valueState) = valueAction functionState
         in (function value, valueState)

instance Monad (BlockM scope) where
    BlockM action >>= next = BlockM $ \state ->
        let (value, nextState) = action state
         in runBlockM (next value) nextState

program :: Name -> Compute () -> ComputeIR
program name action =
    let (_, finalState) = runCompute action initialState
     in IR.ComputeIR
            { IR.computeName = name
            , IR.computeSymbols = stateSymbols finalState
            , IR.computeTensors = stateTensors finalState
            , IR.computeBlocks = stateBlocks finalState
            }
  where
    initialState = ComputeState [] [] [] 0 0

dim :: Name -> Compute Dim
dim name = Compute $ \state ->
    let identifier = IR.SymbolId (length (stateSymbols state))
        symbol = IR.Symbol identifier name
     in (IR.SymbolDim identifier, state{stateSymbols = stateSymbols state ++ [symbol]})

staticDim :: Word64 -> Dim
staticDim = IR.StaticDim

input :: Name -> DTypeRep element -> [Dim] -> Compute (Tensor Input element)
input name dtype shape = declareTensor dtype name shape IR.InputTensor

output :: Name -> DTypeRep element -> [Dim] -> Compute (Tensor Output element)
output name dtype shape = declareTensor dtype name shape IR.OutputTensor

declareTensor :: DTypeRep element -> Name -> [Dim] -> (Int -> IR.TensorKind) -> Compute (Tensor access element)
declareTensor dtype name shape kind = Compute $ \state ->
    let identifier = IR.TensorId (length (stateTensors state))
        ordinal = length (filter (sameKind . IR.tensorKind) (stateTensors state))
        tensor = IR.TensorDecl identifier name (dtypeValue dtype) shape (kind ordinal)
     in (Tensor identifier, state{stateTensors = stateTensors state ++ [tensor]})
  where
    sameKind tensorKind = case (kind 0, tensorKind) of
        (IR.InputTensor _, IR.InputTensor _)   -> True
        (IR.OutputTensor _, IR.OutputTensor _) -> True
        _                                      -> False

dtypeValue :: DTypeRep element -> IR.DType
dtypeValue F32Rep = IR.F32Type

block :: Name -> (forall scope. BlockM scope ()) -> Compute ()
block name action = Compute $ \state ->
    let initialBlock = BlockState [] [] (stateNextAxis state) (stateNextNode state)
        (_, finalBlock) = runBlockM action initialBlock
        computeBlock =
            IR.ComputeBlock
                { IR.blockId = IR.BlockId (length (stateBlocks state))
                , IR.blockName = name
                , IR.blockAxes = blockStateAxes finalBlock
                , IR.blockDefinitions = blockStateDefinitions finalBlock
                }
        nextState =
            state
                { stateBlocks = stateBlocks state ++ [computeBlock]
                , stateNextAxis = blockStateNextAxis finalBlock
                , stateNextNode = blockStateNextNode finalBlock
                }
     in ((), nextState)

spatial :: Name -> Dim -> BlockM scope (Axis scope)
spatial = declareAxis IR.Spatial

reduction :: Name -> Dim -> BlockM scope (Axis scope)
reduction = declareAxis IR.Reduction

declareAxis :: IR.AxisKind -> Name -> Dim -> BlockM scope (Axis scope)
declareAxis kind name extent = BlockM $ \state ->
    let identifier = IR.AxisId (blockStateNextAxis state)
        axis = IR.AxisDecl identifier name kind (IR.StaticDim 0) extent
     in ( Axis identifier
        , state
            { blockStateAxes = blockStateAxes state ++ [axis]
            , blockStateNextAxis = blockStateNextAxis state + 1
            }
        )

define :: Tensor Output F32 -> [Axis scope] -> Expr scope F32 -> BlockM scope ()
define (Tensor target) indices expression = BlockM $ \state ->
    let targetIndices = map (IR.AxisIndex . axisIdOf) indices
        definition = case expression of
            Expression value -> IR.PointwiseDef target targetIndices value
            ReductionExpression reducer identity axes value ->
                IR.ReductionDef target targetIndices reducer identity axes value
     in ((), state{blockStateDefinitions = blockStateDefinitions state ++ [definition]})

reduce :: Reducer F32 -> Expr scope F32 -> [Axis scope] -> Expr scope F32 -> Expr scope F32
reduce (Reducer reducer) identity axes value =
    ReductionExpression reducer (scalarValue identity) (map axisIdOf axes) (scalarValue value)

sumOver :: [Axis scope] -> Expr scope F32 -> Expr scope F32
sumOver = reduce add 0

add :: Reducer F32
add = Reducer IR.AddReducer

multiply :: Reducer F32
multiply = Reducer IR.MulReducer

minimumReducer :: Reducer F32
minimumReducer = Reducer IR.MinReducer

maximumReducer :: Reducer F32
maximumReducer = Reducer IR.MaxReducer

(!) :: Tensor access F32 -> [Axis scope] -> Expr scope F32
Tensor tensor ! indices = Expression (IR.LoadExpr tensor (map (IR.AxisIndex . axisIdOf) indices))

infixl 9 !

boolean :: Bool -> Expr scope Boolean
boolean = Expression . IR.LiteralExpr . IR.BoolLiteral

index :: Axis scope -> Expr scope Index
index = Expression . IR.IndexValueExpr . axisIdOf

indexLiteral :: Word64 -> Expr scope Index
indexLiteral = Expression . IR.LiteralExpr . IR.IndexLiteral

fma :: Expr scope F32 -> Expr scope F32 -> Expr scope F32 -> Expr scope F32
fma lhs rhs accumulator =
    Expression (IR.FmaExpr (scalarValue lhs) (scalarValue rhs) (scalarValue accumulator))

min_ :: Expr scope F32 -> Expr scope F32 -> Expr scope F32
min_ lhs rhs = Expression (IR.MinExpr (scalarValue lhs) (scalarValue rhs))

max_ :: Expr scope F32 -> Expr scope F32 -> Expr scope F32
max_ lhs rhs = Expression (IR.MaxExpr (scalarValue lhs) (scalarValue rhs))

exp_ :: Expr scope F32 -> Expr scope F32
exp_ = Expression . IR.ExpExpr . scalarValue

log_ :: Expr scope F32 -> Expr scope F32
log_ = Expression . IR.LogExpr . scalarValue

compare_ :: ComparePredicate -> Expr scope element -> Expr scope element -> Expr scope Boolean
compare_ predicate lhs rhs = Expression (IR.CompareExpr predicate (scalarValue lhs) (scalarValue rhs))

select :: Expr scope Boolean -> Expr scope element -> Expr scope element -> Expr scope element
select condition trueValue falseValue =
    Expression (IR.SelectExpr (scalarValue condition) (scalarValue trueValue) (scalarValue falseValue))

named :: Name -> Expr scope element -> BlockM scope (Expr scope element)
named name expression = BlockM $ \state ->
    let identifier = IR.ComputeNodeId (blockStateNextNode state)
        namedValue = IR.NamedExpr identifier name (scalarValue expression)
     in (Expression namedValue, state{blockStateNextNode = blockStateNextNode state + 1})

axisIdOf :: Axis scope -> IR.AxisId
axisIdOf (Axis identifier) = identifier

scalarValue :: Expr scope element -> IR.ScalarExpr
scalarValue expression = case expression of
    Expression value -> value
    ReductionExpression{} -> error "a reduction must be the complete right-hand side of define"

instance Num (Expr scope F32) where
    lhs + rhs = Expression (IR.AddExpr (scalarValue lhs) (scalarValue rhs))
    lhs - rhs = Expression (IR.SubExpr (scalarValue lhs) (scalarValue rhs))
    lhs * rhs = Expression (IR.MulExpr (scalarValue lhs) (scalarValue rhs))
    negate value = 0 - value
    abs value = max_ value (negate value)
    signum value =
        select
            (compare_ IR.Greater value 0)
            1
            (select (compare_ IR.Less value 0) (-1) 0)
    fromInteger = Expression . IR.LiteralExpr . IR.F32Literal . fromInteger

instance Fractional (Expr scope F32) where
    lhs / rhs = Expression (IR.DivExpr (scalarValue lhs) (scalarValue rhs))
    fromRational = Expression . IR.LiteralExpr . IR.F32Literal . fromRational

tensorAt :: TensorId -> ComputeIR -> TensorDecl
tensorAt = IR.tensorAt

blockAt :: BlockId -> ComputeIR -> ComputeBlock
blockAt = IR.blockAt

axisAt :: AxisId -> ComputeBlock -> AxisDecl
axisAt = IR.axisAt
