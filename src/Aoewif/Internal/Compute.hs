{-# LANGUAGE RoleAnnotations #-}

module Aoewif.Internal.Compute (
    F32,
    Boolean,
    IndexValue,
    Input,
    Output,
    Name (..),
    Dim,
    DTypeRep,
    f32,
    Tensor,
    Index,
    Expr,
    Reducer,
    Compute,
    BlockM,
    SymbolId (..),
    TensorId (..),
    IndexId (..),
    LoopId (..),
    BlockId (..),
    DimExpr (..),
    Symbol (..),
    DType (..),
    ScalarLiteral (..),
    ComparePredicate (..),
    TensorKind (..),
    TensorDecl (..),
    IndexExpr (..),
    Predicate (..),
    IndexBinding (..),
    Loop (..),
    Block (..),
    Statement (..),
    LoopIR (..),
    IR (..),
    ComputeIndexExpr (..),
    ComputeValueId (..),
    ScalarExpr (..),
    ReducerKind (..),
    ComputeStatement (..),
    ComputeBlock (..),
    tensorAt,
    program,
    dim,
    staticDim,
    input,
    output,
    for,
    block,
    load,
    store,
    update,
    add,
    multiply,
    minimumReducer,
    maximumReducer,
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
) where

import           Aoewif.Internal.Compute.Operation (ComputeBlock (..),
                                                    ComputeIndexExpr (..),
                                                    ComputeStatement (..),
                                                    ComputeValueId (..),
                                                    ReducerKind (..),
                                                    ScalarExpr (..))
import           Aoewif.Internal.IR                (Block (..), BlockId (..),
                                                    ComparePredicate (..),
                                                    DType (..), DimExpr (..),
                                                    IR (..), IndexBinding (..),
                                                    IndexExpr (..),
                                                    IndexId (..), Loop (..),
                                                    LoopIR (..), LoopId (..),
                                                    Name (..), Predicate (..),
                                                    ScalarLiteral (..),
                                                    Statement (..), Symbol (..),
                                                    SymbolId (..),
                                                    TensorDecl (..),
                                                    TensorId (..),
                                                    TensorKind (..))
import qualified Aoewif.Internal.IR                as IR
import           Data.Word                         (Word64)
import           Prelude                           hiding (exp, log, maximum,
                                                    minimum)

type Dim = DimExpr

data F32
data Boolean
data IndexValue
data Input
data Output

data DTypeRep element where
    F32Rep :: DTypeRep F32

f32 :: DTypeRep F32
f32 = F32Rep

newtype Tensor access element = Tensor TensorId

type role Tensor nominal nominal

newtype Index scope = Index IndexId

type role Index nominal

newtype Expr scope element = Expression ScalarExpr

type role Expr representational nominal

newtype Reducer element = Reducer ReducerKind

type role Reducer nominal

data ComputeState = ComputeState
    { stateSymbols    :: [Symbol]
    , stateTensors    :: [TensorDecl]
    , stateStatements :: [Statement ComputeBlock]
    , stateBindings   :: [IndexBinding]
    , stateNextIndex  :: !Int
    , stateNextLoop   :: !Int
    , stateNextBlock  :: !Int
    }

newtype Compute scope value = Compute
    { runCompute :: ComputeState -> (value, ComputeState)
    }

type role Compute nominal representational

instance Functor (Compute scope) where
    fmap transform (Compute action) = Compute $ \state ->
        let (value, nextState) = action state
         in (transform value, nextState)

instance Applicative (Compute scope) where
    pure value = Compute (value,)
    Compute functionAction <*> Compute valueAction = Compute $ \state ->
        let (function, functionState) = functionAction state
            (value, valueState) = valueAction functionState
         in (function value, valueState)

instance Monad (Compute scope) where
    Compute action >>= next = Compute $ \state ->
        let (value, nextState) = action state
         in runCompute (next value) nextState

data BlockState = BlockState
    { blockStateStatements :: [ComputeStatement]
    , blockStateNextValue  :: !Int
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

program :: Name -> (forall scope. Compute scope ()) -> IR ComputeBlock
program name action =
    let (_, finalState) = runCompute action initialState
     in IR
            { irName = name
            , irSymbols = stateSymbols finalState
            , irTensors = stateTensors finalState
            , irBody = LoopIR (stateStatements finalState)
            }
  where
    initialState = ComputeState [] [] [] [] 0 0 0

dim :: Name -> Compute scope Dim
dim name = Compute $ \state ->
    let identifier = SymbolId (length (stateSymbols state))
        symbol = Symbol identifier name
     in ( SymbolDim identifier
        , state{stateSymbols = stateSymbols state ++ [symbol]}
        )

staticDim :: Word64 -> Dim
staticDim = StaticDim

input :: Name -> DTypeRep element -> [Dim] -> Compute scope (Tensor Input element)
input name dtype shape = declareTensor dtype name shape InputTensor

output :: Name -> DTypeRep element -> [Dim] -> Compute scope (Tensor Output element)
output name dtype shape = declareTensor dtype name shape OutputTensor

declareTensor :: DTypeRep element -> Name -> [Dim] -> (Int -> TensorKind) -> Compute scope (Tensor access element)
declareTensor dtype name shape kind = Compute $ \state ->
    let identifier = TensorId (length (stateTensors state))
        ordinal = length (filter (sameKind . tensorKind) (stateTensors state))
        tensor = TensorDecl identifier name (dtypeValue dtype) shape (kind ordinal)
     in ( Tensor identifier
        , state{stateTensors = stateTensors state ++ [tensor]}
        )
  where
    sameKind tensorKindValue = case (kind 0, tensorKindValue) of
        (InputTensor _, InputTensor _)   -> True
        (OutputTensor _, OutputTensor _) -> True
        _                                -> False

dtypeValue :: DTypeRep element -> DType
dtypeValue F32Rep = F32Type

for :: Name -> Dim -> (Index scope -> Compute scope ()) -> Compute scope ()
for name extent action = Compute $ \state ->
    let loopIdentifier = LoopId (stateNextLoop state)
        indexIdentifier = IndexId (stateNextIndex state)
        loop = Loop loopIdentifier name (StaticDim 0) extent
        outerStatements = stateStatements state
        outerBindings = stateBindings state
        binding = IndexBinding indexIdentifier (LoopIndex loopIdentifier)
        bodyInitialState =
            state
                { stateStatements = []
                , stateBindings = outerBindings ++ [binding]
                , stateNextIndex = stateNextIndex state + 1
                , stateNextLoop = stateNextLoop state + 1
                }
        (_, bodyFinalState) = runCompute (action (Index indexIdentifier)) bodyInitialState
        loopStatement = For loop (LoopIR (stateStatements bodyFinalState))
        finalState =
            bodyFinalState
                { stateStatements = outerStatements ++ [loopStatement]
                , stateBindings = outerBindings
                }
     in ((), finalState)

block :: Name -> BlockM scope () -> Compute scope ()
block name action = Compute $ \state ->
    let identifier = BlockId (stateNextBlock state)
        (_, finalBlockState) = runBlockM action (BlockState [] 0)
        operation = ComputeBlock (blockStateStatements finalBlockState)
        blockSite = Block identifier name (stateBindings state) operation
     in ( ()
        , state
            { stateStatements = stateStatements state ++ [Execute blockSite]
            , stateNextBlock = stateNextBlock state + 1
            }
        )

load :: Tensor access element -> [Index scope] -> BlockM scope (Expr scope element)
load (Tensor tensorIdentifier) indices = BlockM $ \state ->
    let identifier = ComputeValueId (blockStateNextValue state)
        statement =
            Load
                { statementResult = identifier
                , statementSource = tensorIdentifier
                , statementIndices = map indexExpression indices
                }
     in ( Expression (ValueExpr identifier)
        , state
            { blockStateStatements = blockStateStatements state ++ [statement]
            , blockStateNextValue = blockStateNextValue state + 1
            }
        )

store :: Tensor Output element -> [Index scope] -> Expr scope element -> BlockM scope ()
store (Tensor target) indices value =
    appendStatement
        Store
            { statementTarget = target
            , statementIndices = map indexExpression indices
            , statementValue = scalarValue value
            }

update :: Reducer element -> Tensor Output element -> [Index scope] -> Expr scope element -> BlockM scope ()
update (Reducer reducer) (Tensor target) indices value =
    appendStatement
        Update
            { statementReducer = reducer
            , statementTarget = target
            , statementIndices = map indexExpression indices
            , statementValue = scalarValue value
            }

appendStatement :: ComputeStatement -> BlockM scope ()
appendStatement statement = BlockM $ \state ->
    ((), state{blockStateStatements = blockStateStatements state ++ [statement]})

indexExpression :: Index scope -> ComputeIndexExpr
indexExpression (Index identifier) = IterationIndex identifier

add :: Reducer F32
add = Reducer AddReducer

multiply :: Reducer F32
multiply = Reducer MulReducer

minimumReducer :: Reducer F32
minimumReducer = Reducer MinReducer

maximumReducer :: Reducer F32
maximumReducer = Reducer MaxReducer

boolean :: Bool -> Expr scope Boolean
boolean = Expression . LiteralExpr . BoolLiteral

index :: Index scope -> Expr scope IndexValue
index (Index identifier) = Expression (IndexValueExpr identifier)

indexLiteral :: Word64 -> Expr scope IndexValue
indexLiteral = Expression . LiteralExpr . IndexLiteral

fma :: Expr scope F32 -> Expr scope F32 -> Expr scope F32 -> Expr scope F32
fma lhs rhs accumulator =
    Expression (FmaExpr (scalarValue lhs) (scalarValue rhs) (scalarValue accumulator))

min_ :: Expr scope F32 -> Expr scope F32 -> Expr scope F32
min_ lhs rhs = Expression (MinExpr (scalarValue lhs) (scalarValue rhs))

max_ :: Expr scope F32 -> Expr scope F32 -> Expr scope F32
max_ lhs rhs = Expression (MaxExpr (scalarValue lhs) (scalarValue rhs))

exp_ :: Expr scope F32 -> Expr scope F32
exp_ = Expression . ExpExpr . scalarValue

log_ :: Expr scope F32 -> Expr scope F32
log_ = Expression . LogExpr . scalarValue

compare_ :: ComparePredicate -> Expr scope element -> Expr scope element -> Expr scope Boolean
compare_ predicate lhs rhs =
    Expression (CompareExpr predicate (scalarValue lhs) (scalarValue rhs))

select :: Expr scope Boolean -> Expr scope element -> Expr scope element -> Expr scope element
select condition trueValue falseValue =
    Expression
        (SelectExpr (scalarValue condition) (scalarValue trueValue) (scalarValue falseValue))

scalarValue :: Expr scope element -> ScalarExpr
scalarValue (Expression value) = value

instance Num (Expr scope F32) where
    lhs + rhs = Expression (AddExpr (scalarValue lhs) (scalarValue rhs))
    lhs - rhs = Expression (SubExpr (scalarValue lhs) (scalarValue rhs))
    lhs * rhs = Expression (MulExpr (scalarValue lhs) (scalarValue rhs))
    negate value = 0 - value
    abs value = max_ value (negate value)
    signum value =
        select
            (compare_ Greater value 0)
            1
            (select (compare_ Less value 0) (-1) 0)
    fromInteger = Expression . LiteralExpr . F32Literal . fromInteger

instance Fractional (Expr scope F32) where
    lhs / rhs = Expression (DivExpr (scalarValue lhs) (scalarValue rhs))
    fromRational = Expression . LiteralExpr . F32Literal . fromRational

tensorAt :: TensorId -> IR operation -> TensorDecl
tensorAt = IR.tensorAt
