{-# LANGUAGE RoleAnnotations #-}

module Aoewif.Internal.Schedule.Builder (
    Schedule (Schedule),
    ScheduleState (stateIR, stateNextLoop),
    Block,
    Loop (Loop),
    schedule,
    block,
    loopOf,
) where

import qualified Aoewif.Internal.IR as IR
import           Data.List          (find)
import           Data.Maybe         (fromJust, listToMaybe, mapMaybe)

newtype Block scope = Block IR.BlockId

type role Block nominal

newtype Loop scope = Loop IR.LoopId

type role Loop nominal

data ScheduleState operation = ScheduleState
    { stateIR       :: IR.IR operation
    , stateNextLoop :: !Int
    }

newtype Schedule operation scope value = Schedule
    { runSchedule :: ScheduleState operation -> (value, ScheduleState operation)
    }

type role Schedule nominal nominal representational

instance Functor (Schedule operation scope) where
    fmap transform (Schedule action) = Schedule $ \state ->
        let (value, nextState) = action state
         in (transform value, nextState)

instance Applicative (Schedule operation scope) where
    pure value = Schedule (value,)
    Schedule functionAction <*> Schedule valueAction = Schedule $ \state ->
        let (function, functionState) = functionAction state
            (value, valueState) = valueAction functionState
         in (function value, valueState)

instance Monad (Schedule operation scope) where
    Schedule action >>= next = Schedule $ \state ->
        let (value, nextState) = action state
         in runSchedule (next value) nextState

schedule :: IR.IR operation -> (forall scope. Schedule operation scope ()) -> IR.IR operation
schedule input build =
    let initialState = ScheduleState input (nextLoopId (IR.irBody input))
        (_, finalState) = runSchedule build initialState
     in stateIR finalState
  where
    nextLoopId body = maximum (-1 : map unwrapLoopId (allLoopIds body)) + 1

    allLoopIds (IR.LoopIR statements) = concatMap idsInStatement statements
      where
        idsInStatement statement = case statement of
            IR.For loop body -> IR.loopId loop : allLoopIds body
            IR.Guard _ body  -> allLoopIds body
            IR.Execute _     -> []

    unwrapLoopId (IR.LoopId identifier) = identifier

block :: IR.Name -> Schedule operation scope (Block scope)
block name = Schedule $ \state ->
    let target = fromJust (findBlock name (IR.irBody (stateIR state)))
     in (Block (IR.blockId target), state)
  where
    findBlock targetName (IR.LoopIR statements) =
        listToMaybe (mapMaybe findInStatement statements)
      where
        findInStatement statement = case statement of
            IR.For _ body -> findBlock targetName body
            IR.Guard _ body -> findBlock targetName body
            IR.Execute candidate
                | IR.blockName candidate == targetName -> Just candidate
                | otherwise -> Nothing

loopOf :: Block scope -> IR.Name -> Schedule operation scope (Loop scope)
loopOf (Block blockIdentifier) name = Schedule $ \state ->
    let loops = fromJust (loopsAround blockIdentifier (IR.irBody (stateIR state)))
        target = fromJust (find ((== name) . IR.loopName) loops)
     in (Loop (IR.loopId target), state)
  where
    loopsAround targetBlock (IR.LoopIR statements) =
        listToMaybe (mapMaybe findInStatement statements)
      where
        findInStatement statement = case statement of
            IR.For loop body -> (loop :) <$> loopsAround targetBlock body
            IR.Guard _ body -> loopsAround targetBlock body
            IR.Execute candidate
                | IR.blockId candidate == targetBlock -> Just []
                | otherwise -> Nothing
