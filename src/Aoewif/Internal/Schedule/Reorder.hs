module Aoewif.Internal.Schedule.Reorder (
    reorder,
) where

import qualified Aoewif.Internal.IR               as IR
import           Aoewif.Internal.Schedule.Builder (Loop (Loop),
                                                   Schedule (Schedule),
                                                   ScheduleState (stateIR))
import           Data.List                        (find)
import           Data.Maybe                       (fromJust)

reorder :: [Loop scope] -> Schedule operation scope ()
reorder handles = Schedule $ \state ->
    let order = map loopIdentifier handles
        input = stateIR state
        output = input{IR.irBody = reorderLoops order (IR.irBody input)}
     in ((), state{stateIR = output})
  where
    loopIdentifier (Loop identifier) = identifier

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
