module Aoewif.Internal.Schedule.Split (
    split,
) where

import qualified Aoewif.Internal.IR               as IR
import           Aoewif.Internal.Schedule.Builder (Loop (Loop),
                                                   Schedule (Schedule),
                                                   ScheduleState (stateIR, stateNextLoop))
import           Data.Word                        (Word64)

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
