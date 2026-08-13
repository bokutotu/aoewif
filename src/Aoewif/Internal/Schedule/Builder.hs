{-# LANGUAGE RoleAnnotations #-}

module Aoewif.Internal.Schedule.Builder (
    Schedule,
    Block,
    Loop,
    schedule,
    block,
    loopOf,
    split,
    reorder,
) where

import qualified Aoewif.Internal.IR as IR
import           Data.List          (find)
import           Data.Maybe         (catMaybes, fromJust)
import           Data.Word          (Word64)

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

block :: IR.Name -> Schedule operation scope (Block scope)
block name = Schedule $ \state ->
    let target = fromJust (findBlock name (IR.irBody (stateIR state)))
     in (Block (IR.blockId target), state)

loopOf :: Block scope -> IR.Name -> Schedule operation scope (Loop scope)
loopOf (Block blockIdentifier) name = Schedule $ \state ->
    let loops = fromJust (loopsAround blockIdentifier (IR.irBody (stateIR state)))
        target = fromJust (find ((== name) . IR.loopName) loops)
     in (Loop (IR.loopId target), state)

split :: Loop scope -> Word64 -> Schedule operation scope (Loop scope, Loop scope)
split (Loop target) factor = Schedule $ \state ->
    let outerIdentifier = IR.LoopId (stateNextLoop state)
        innerIdentifier = IR.LoopId (stateNextLoop state + 1)
        input = stateIR state
        output = input{IR.irBody = splitLoop target factor outerIdentifier innerIdentifier (IR.irBody input)}
        nextState =
            state
                { stateIR = output
                , stateNextLoop = stateNextLoop state + 2
                }
     in ((Loop outerIdentifier, Loop innerIdentifier), nextState)

reorder :: [Loop scope] -> Schedule operation scope ()
reorder handles = Schedule $ \state ->
    let order = map loopIdentifier handles
        input = stateIR state
        output = input{IR.irBody = reorderLoops order (IR.irBody input)}
     in ((), state{stateIR = output})

findBlock :: IR.Name -> IR.LoopIR operation -> Maybe (IR.Block operation)
findBlock name (IR.LoopIR statements) = firstJust (map findInStatement statements)
  where
    findInStatement statement = case statement of
        IR.For _ body -> findBlock name body
        IR.Guard _ body -> findBlock name body
        IR.Execute candidate
            | IR.blockName candidate == name -> Just candidate
            | otherwise -> Nothing

loopsAround :: IR.BlockId -> IR.LoopIR operation -> Maybe [IR.Loop]
loopsAround blockIdentifier (IR.LoopIR statements) = firstJust (map findInStatement statements)
  where
    findInStatement statement = case statement of
        IR.For loop body -> (loop :) <$> loopsAround blockIdentifier body
        IR.Guard _ body -> loopsAround blockIdentifier body
        IR.Execute candidate
            | IR.blockId candidate == blockIdentifier -> Just []
            | otherwise -> Nothing

splitLoop :: IR.LoopId -> Word64 -> IR.LoopId -> IR.LoopId -> IR.LoopIR operation -> IR.LoopIR operation
splitLoop target factor outerIdentifier innerIdentifier (IR.LoopIR statements) =
    IR.LoopIR (map splitStatement statements)
  where
    splitStatement statement = case statement of
        IR.For original body
            | IR.loopId original == target ->
                let replacement =
                        addLowerBound
                            (IR.loopLowerBound original)
                            ( IR.AddIndex
                                (IR.MulIndex (IR.LoopIndex outerIdentifier) (IR.ConstantIndex factor))
                                (IR.LoopIndex innerIdentifier)
                            )
                    outer =
                        original
                            { IR.loopId = outerIdentifier
                            , IR.loopName = appendName (IR.loopName original) "_outer"
                            , IR.loopLowerBound = IR.StaticDim 0
                            , IR.loopExtent = IR.CeilDivDim (IR.loopExtent original) factor
                            }
                    inner =
                        original
                            { IR.loopId = innerIdentifier
                            , IR.loopName = appendName (IR.loopName original) "_inner"
                            , IR.loopLowerBound = IR.StaticDim 0
                            , IR.loopExtent = IR.StaticDim factor
                            }
                    rewrittenBody = replaceLoopReferences target replacement body
                    innerBody
                        | needsTailGuard factor (IR.loopExtent original) =
                            addTailGuard
                                ( IR.IndexLessThan
                                    replacement
                                    ( addLowerBound
                                        (IR.loopLowerBound original)
                                        (IR.DimensionIndex (IR.loopExtent original))
                                    )
                                )
                                rewrittenBody
                        | otherwise = rewrittenBody
                 in IR.For outer (IR.LoopIR [IR.For inner innerBody])
            | otherwise -> IR.For original (splitLoop target factor outerIdentifier innerIdentifier body)
        IR.Guard predicate body -> IR.Guard predicate (splitLoop target factor outerIdentifier innerIdentifier body)
        IR.Execute _ -> statement

addLowerBound :: IR.DimExpr -> IR.IndexExpr -> IR.IndexExpr
addLowerBound lower expression = case lower of
    IR.StaticDim 0 -> expression
    _              -> IR.AddIndex (IR.DimensionIndex lower) expression

addTailGuard :: IR.Predicate -> IR.LoopIR operation -> IR.LoopIR operation
addTailGuard predicate (IR.LoopIR statements) =
    IR.LoopIR (map guardStatement statements)
  where
    guardStatement statement = case statement of
        IR.For loop body -> IR.For loop (addTailGuard predicate body)
        IR.Guard existing body -> IR.Guard existing (addTailGuard predicate body)
        IR.Execute _ -> IR.Guard predicate (IR.LoopIR [statement])

replaceLoopReferences :: IR.LoopId -> IR.IndexExpr -> IR.LoopIR operation -> IR.LoopIR operation
replaceLoopReferences target replacement (IR.LoopIR statements) =
    IR.LoopIR (map replaceStatement statements)
  where
    replaceStatement statement = case statement of
        IR.For loop body -> IR.For loop (replaceLoopReferences target replacement body)
        IR.Guard predicate body ->
            IR.Guard
                (replacePredicate target replacement predicate)
                (replaceLoopReferences target replacement body)
        IR.Execute blockOperation ->
            IR.Execute
                blockOperation
                    { IR.blockBindings = map replaceBinding (IR.blockBindings blockOperation)
                    }

    replaceBinding binding =
        binding
            { IR.bindingExpression = replaceIndex target replacement (IR.bindingExpression binding)
            }

replacePredicate :: IR.LoopId -> IR.IndexExpr -> IR.Predicate -> IR.Predicate
replacePredicate target replacement predicate = case predicate of
    IR.IndexLessThan lhs rhs ->
        IR.IndexLessThan (replaceIndex target replacement lhs) (replaceIndex target replacement rhs)
    IR.IndexEqual lhs rhs ->
        IR.IndexEqual (replaceIndex target replacement lhs) (replaceIndex target replacement rhs)

replaceIndex :: IR.LoopId -> IR.IndexExpr -> IR.IndexExpr -> IR.IndexExpr
replaceIndex target replacement expression = case expression of
    IR.LoopIndex identifier
        | identifier == target -> replacement
        | otherwise -> expression
    IR.DimensionIndex _ -> expression
    IR.ConstantIndex _ -> expression
    IR.AddIndex lhs rhs ->
        IR.AddIndex (replaceIndex target replacement lhs) (replaceIndex target replacement rhs)
    IR.MulIndex lhs rhs ->
        IR.MulIndex (replaceIndex target replacement lhs) (replaceIndex target replacement rhs)
    IR.CeilDivIndex value divisor ->
        IR.CeilDivIndex (replaceIndex target replacement value) divisor

reorderLoops :: [IR.LoopId] -> IR.LoopIR operation -> IR.LoopIR operation
reorderLoops [] body = body
reorderLoops order (IR.LoopIR statements) =
    IR.LoopIR (map reorderStatement statements)
  where
    reorderStatement statement = case statement of
        IR.For loop body
            | IR.loopId loop `elem` order -> singleStatement (reorderFrom loop body)
            | otherwise -> IR.For loop (reorderLoops order body)
        IR.Guard predicate body -> IR.Guard predicate (reorderLoops order body)
        IR.Execute _ -> statement

    reorderFrom firstLoop firstBody =
        let (loops, terminalBody) = takeLoopChain (length order - 1) firstBody
            selectedLoops = firstLoop : loops
         in nestLoops (map (loopFor selectedLoops) order) terminalBody

    singleStatement (IR.LoopIR [statement]) = statement
    singleStatement _ = error "reorder produced multiple statements"

takeLoopChain :: Int -> IR.LoopIR operation -> ([IR.Loop], IR.LoopIR operation)
takeLoopChain remaining body
    | remaining == 0 = ([], body)
takeLoopChain remaining (IR.LoopIR [IR.For loop child]) =
    let (rest, terminalBody) = takeLoopChain (remaining - 1) child
     in (loop : rest, terminalBody)
takeLoopChain _ _ = error "reorder requires a contiguous loop chain"

loopFor :: [IR.Loop] -> IR.LoopId -> IR.Loop
loopFor loops identifier = fromJust (find ((== identifier) . IR.loopId) loops)

nestLoops :: [IR.Loop] -> IR.LoopIR operation -> IR.LoopIR operation
nestLoops loops body = foldr (\loop child -> IR.LoopIR [IR.For loop child]) body loops

nextLoopId :: IR.LoopIR operation -> Int
nextLoopId body = maximum (-1 : map unwrapLoopId (allLoopIds body)) + 1

allLoopIds :: IR.LoopIR operation -> [IR.LoopId]
allLoopIds (IR.LoopIR statements) = concatMap idsInStatement statements
  where
    idsInStatement statement = case statement of
        IR.For loop body -> IR.loopId loop : allLoopIds body
        IR.Guard _ body  -> allLoopIds body
        IR.Execute _     -> []

unwrapLoopId :: IR.LoopId -> Int
unwrapLoopId (IR.LoopId identifier) = identifier

needsTailGuard :: Word64 -> IR.DimExpr -> Bool
needsTailGuard factor extent = case staticDimValue extent of
    Just value -> value `mod` factor /= 0
    Nothing    -> True

staticDimValue :: IR.DimExpr -> Maybe Word64
staticDimValue dimension = case dimension of
    IR.StaticDim value -> Just value
    IR.SymbolDim _ -> Nothing
    IR.CeilDivDim dividend divisor -> (`ceilDiv` divisor) <$> staticDimValue dividend

ceilDiv :: Word64 -> Word64 -> Word64
ceilDiv dividend divisor = (dividend + divisor - 1) `div` divisor

appendName :: IR.Name -> String -> IR.Name
appendName (IR.Name name) suffix = IR.Name (name ++ suffix)

loopIdentifier :: Loop scope -> IR.LoopId
loopIdentifier (Loop identifier) = identifier

firstJust :: [Maybe value] -> Maybe value
firstJust = listToMaybe . catMaybes

listToMaybe :: [value] -> Maybe value
listToMaybe values = case values of
    []        -> Nothing
    value : _ -> Just value
