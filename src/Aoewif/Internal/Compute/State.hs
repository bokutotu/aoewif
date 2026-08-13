{-# LANGUAGE RoleAnnotations #-}

module Aoewif.Internal.Compute.State (
    F32,
    Boolean,
    IndexValue,
    Input,
    Output,
    DataTypeRep (..),
    f32,
    Tensor (..),
    Index (..),
    Compute,
    program,
    dim,
    staticDim,
    input,
    output,
    for,
    emitBlock,
) where

import           Aoewif.Internal.Compute.Operation (ComputeBlock)
import           Aoewif.Internal.IR                (Block (..), BlockId (..),
                                                    DimExpr (..), IR (..),
                                                    IndexBinding (..),
                                                    IndexExpr (..),
                                                    IndexId (..), Loop (..),
                                                    LoopIR (..), LoopId (..),
                                                    Name, Statement (..),
                                                    Symbol (..), SymbolId (..),
                                                    TensorDecl (..),
                                                    TensorId (..),
                                                    TensorKind (..))
import           Aoewif.Internal.Primitive         (DataType (..))
import           Data.Word                         (Word64)

data F32
data Boolean
data IndexValue
data Input
data Output

data DataTypeRep element where
    F32Rep :: DataTypeRep F32

f32 :: DataTypeRep F32
f32 = F32Rep

data Tensor access element where
    Tensor :: DataTypeRep element -> TensorId -> Tensor access element

type role Tensor nominal nominal

newtype Index scope = Index IndexId

type role Index nominal

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

dim :: Name -> Compute scope DimExpr
dim name = Compute $ \state ->
    let identifier = SymbolId (length (stateSymbols state))
        symbol = Symbol identifier name
     in ( SymbolDim identifier
        , state{stateSymbols = stateSymbols state ++ [symbol]}
        )

staticDim :: Word64 -> DimExpr
staticDim = StaticDim

input :: Name -> DataTypeRep element -> [DimExpr] -> Compute scope (Tensor Input element)
input name dataType shape = declareTensor dataType name shape InputTensor

output :: Name -> DataTypeRep element -> [DimExpr] -> Compute scope (Tensor Output element)
output name dataType shape = declareTensor dataType name shape OutputTensor

declareTensor :: DataTypeRep element -> Name -> [DimExpr] -> (Int -> TensorKind) -> Compute scope (Tensor access element)
declareTensor dataType name shape kind = Compute $ \state ->
    let identifier = TensorId (length (stateTensors state))
        ordinal = length (filter (sameKind . tensorKind) (stateTensors state))
        tensor = TensorDecl identifier name (dataTypeValue dataType) shape (kind ordinal)
     in ( Tensor dataType identifier
        , state{stateTensors = stateTensors state ++ [tensor]}
        )
  where
    sameKind tensorKindValue = case (kind 0, tensorKindValue) of
        (InputTensor _, InputTensor _)   -> True
        (OutputTensor _, OutputTensor _) -> True
        _                                -> False

dataTypeValue :: DataTypeRep element -> DataType
dataTypeValue F32Rep = F32Type

for :: Name -> DimExpr -> (Index scope -> Compute scope ()) -> Compute scope ()
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

emitBlock :: Name -> ComputeBlock -> Compute scope ()
emitBlock name operation = Compute $ \state ->
    let identifier = BlockId (stateNextBlock state)
        blockSite = Block identifier name (stateBindings state) operation
     in ( ()
        , state
            { stateStatements = stateStatements state ++ [Execute blockSite]
            , stateNextBlock = stateNextBlock state + 1
            }
        )
