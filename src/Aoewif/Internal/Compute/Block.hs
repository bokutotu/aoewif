{-# LANGUAGE RoleAnnotations #-}

module Aoewif.Internal.Compute.Block (
    Reducer,
    BlockM,
    block,
    load,
    store,
    update,
    add,
    multiply,
    minimumReducer,
    maximumReducer,
) where

import           Aoewif.Internal.Compute.Math      (Expr (..))
import           Aoewif.Internal.Compute.Operation (ComputeBlock (..),
                                                    ComputeIndexExpr (..),
                                                    ComputeStatement (..),
                                                    ComputeValueId (..),
                                                    DataExpr (..),
                                                    ReducerKind (..))
import           Aoewif.Internal.Compute.State     (Compute, DataTypeRep (..),
                                                    F32, Index (..), Output,
                                                    Tensor (..), emitBlock)
import           Aoewif.Internal.IR                (Name)

newtype Reducer element = Reducer ReducerKind

type role Reducer nominal

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

block :: Name -> BlockM scope () -> Compute scope ()
block name action =
    let (_, finalBlockState) = runBlockM action (BlockState [] 0)
     in emitBlock name (ComputeBlock (blockStateStatements finalBlockState))

load :: Tensor access element -> [Index scope] -> BlockM scope (Expr scope element)
load (Tensor F32Rep tensorIdentifier) indices = BlockM $ \state ->
    let identifier = ComputeValueId (blockStateNextValue state)
        statement =
            Load
                { statementResult = identifier
                , statementSource = tensorIdentifier
                , statementIndices = map indexExpression indices
                }
     in ( DataExpression (ValueExpr identifier)
        , state
            { blockStateStatements = blockStateStatements state ++ [statement]
            , blockStateNextValue = blockStateNextValue state + 1
            }
        )

store :: Tensor Output element -> [Index scope] -> Expr scope element -> BlockM scope ()
store (Tensor F32Rep target) indices (DataExpression value) =
    appendStatement
        Store
            { statementTarget = target
            , statementIndices = map indexExpression indices
            , statementValue = value
            }

update :: Reducer element -> Tensor Output element -> [Index scope] -> Expr scope element -> BlockM scope ()
update (Reducer reducer) (Tensor F32Rep target) indices (DataExpression value) =
    appendStatement
        Update
            { statementReducer = reducer
            , statementTarget = target
            , statementIndices = map indexExpression indices
            , statementValue = value
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
