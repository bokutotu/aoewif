{-# LANGUAGE RoleAnnotations #-}

module Aoewif.Internal.Schedule.Builder (
    Cpu,
    Cuda,
    Schedule,
    Block,
    Loop,
    CpuSchedule,
    CudaSchedule,
    block,
    axis,
    split,
    reorder,
    parallel,
    unrollBy,
    tile2,
    tile3,
    bind,
    cpu,
    cuda,
    cpuScheduleIR,
    cudaScheduleIR,
) where

import qualified Aoewif.Internal.Compute.IR    as Compute
import qualified Aoewif.Internal.Schedule.Cpu  as Cpu
import qualified Aoewif.Internal.Schedule.Cuda as Cuda
import qualified Aoewif.Internal.Schedule.IR   as IR
import           Data.List                     (find)
import           Data.Maybe                    (fromJust)
import           Data.Word                     (Word64)

data Cpu

data Cuda

type CpuSchedule = Cpu.CpuSchedule

type CudaSchedule = Cuda.CudaSchedule

newtype Block scope = Block Compute.BlockId

type role Block nominal

data Loop scope = Loop Compute.BlockId IR.LoopId

type role Loop nominal

data ScheduleState = ScheduleState
    { stateCompute   :: Compute.ComputeIR
    , stateSchedule  :: IR.ScheduleIR
    , stateAxisLoops :: [(Compute.BlockId, Compute.AxisId, IR.LoopId)]
    , stateNextLoop  :: !Int
    }

newtype Schedule backend scope value = Schedule
    { runSchedule :: ScheduleState -> (value, ScheduleState)
    }

type role Schedule nominal nominal representational

instance Functor (Schedule backend scope) where
    fmap transform (Schedule action) = Schedule $ \state ->
        let (value, nextState) = action state
         in (transform value, nextState)

instance Applicative (Schedule backend scope) where
    pure value = Schedule (value,)
    Schedule functionAction <*> Schedule valueAction = Schedule $ \state ->
        let (function, functionState) = functionAction state
            (value, valueState) = valueAction functionState
         in (function value, valueState)

instance Monad (Schedule backend scope) where
    Schedule action >>= next = Schedule $ \state ->
        let (value, nextState) = action state
         in runSchedule (next value) nextState

block :: Compute.Name -> Schedule backend scope (Block scope)
block name = Schedule $ \state ->
    let computeBlock = fromJust (find ((== name) . Compute.blockName) (Compute.computeBlocks (stateCompute state)))
     in (Block (Compute.blockId computeBlock), state)

axis :: Block scope -> Compute.Name -> Schedule backend scope (Loop scope)
axis (Block blockIdentifier) name = Schedule $ \state ->
    let computeBlock = Compute.blockAt blockIdentifier (stateCompute state)
        axisDecl = fromJust (find ((== name) . Compute.axisName) (Compute.blockAxes computeBlock))
        identifier = fromJust (lookupAxisLoop blockIdentifier (Compute.axisId axisDecl) (stateAxisLoops state))
     in (Loop blockIdentifier identifier, state)

split :: Loop scope -> Word64 -> Schedule backend scope (Loop scope, Loop scope)
split (Loop blockIdentifier targetLoop) factor = Schedule $ \state ->
    let outerIdentifier = IR.LoopId (stateNextLoop state)
        innerIdentifier = IR.LoopId (stateNextLoop state + 1)
        nextSchedule = IR.splitScheduleIR blockIdentifier targetLoop factor outerIdentifier innerIdentifier (stateSchedule state)
        nextState =
            state
                { stateSchedule = nextSchedule
                , stateNextLoop = stateNextLoop state + 2
                }
     in ((Loop blockIdentifier outerIdentifier, Loop blockIdentifier innerIdentifier), nextState)

reorder :: Block scope -> [Loop scope] -> Schedule backend scope ()
reorder (Block blockIdentifier) loops = Schedule $ \state ->
    let identifiers = map loopIdentifier loops
        nextSchedule = IR.reorderScheduleIR blockIdentifier identifiers (stateSchedule state)
     in ((), state{stateSchedule = nextSchedule})

parallel :: Loop scope -> Schedule backend scope ()
parallel loop = Schedule $ \state ->
    ((), state{stateSchedule = IR.parallelScheduleIR (loopIdentifier loop) (stateSchedule state)})

unrollBy :: Word64 -> Loop scope -> Schedule backend scope ()
unrollBy factor loop = Schedule $ \state ->
    ((), state{stateSchedule = IR.unrollScheduleIR (loopIdentifier loop) factor (stateSchedule state)})

tile2 ::
    (Loop scope, Loop scope) ->
    (Word64, Word64) ->
    Schedule backend scope ((Loop scope, Loop scope), (Loop scope, Loop scope))
tile2 (first, second) (firstFactor, secondFactor) = do
    firstTile <- split first firstFactor
    secondTile <- split second secondFactor
    pure (firstTile, secondTile)

tile3 ::
    (Loop scope, Loop scope, Loop scope) ->
    (Word64, Word64, Word64) ->
    Schedule backend scope ((Loop scope, Loop scope), (Loop scope, Loop scope), (Loop scope, Loop scope))
tile3 (first, second, third) (firstFactor, secondFactor, thirdFactor) = do
    firstTile <- split first firstFactor
    secondTile <- split second secondFactor
    thirdTile <- split third thirdFactor
    pure (firstTile, secondTile, thirdTile)

bind :: Loop scope -> IR.CudaBinding -> Schedule Cuda scope ()
bind loop cudaBinding = Schedule $ \state ->
    ((), state{stateSchedule = IR.bindCudaScheduleIR (loopIdentifier loop) cudaBinding (stateSchedule state)})

cpu :: Compute.ComputeIR -> (forall scope. Schedule Cpu scope ()) -> CpuSchedule
cpu computeIR build =
    let initial = initialState computeIR
        (_, final) = runSchedule build initial
     in Cpu.CpuSchedule computeIR (stateSchedule final)

cuda :: Compute.ComputeIR -> (forall scope. Schedule Cuda scope ()) -> CudaSchedule
cuda computeIR build =
    let initial = initialState computeIR
        (_, final) = runSchedule build initial
     in Cuda.CudaSchedule computeIR (stateSchedule final)

cpuScheduleIR :: CpuSchedule -> IR.ScheduleIR
cpuScheduleIR = Cpu.cpuScheduleIR

cudaScheduleIR :: CudaSchedule -> IR.ScheduleIR
cudaScheduleIR = Cuda.cudaScheduleIR

initialState :: Compute.ComputeIR -> ScheduleState
initialState computeIR =
    let (scheduleIR, handles, nextLoop) = IR.initialScheduleIR computeIR
     in ScheduleState
            { stateCompute = computeIR
            , stateSchedule = scheduleIR
            , stateAxisLoops = handles
            , stateNextLoop = nextLoop
            }

lookupAxisLoop :: Compute.BlockId -> Compute.AxisId -> [(Compute.BlockId, Compute.AxisId, IR.LoopId)] -> Maybe IR.LoopId
lookupAxisLoop _ _ [] = Nothing
lookupAxisLoop blockIdentifier axisIdentifier ((candidateBlock, candidateAxis, candidateLoop) : rest)
    | blockIdentifier == candidateBlock && axisIdentifier == candidateAxis = Just candidateLoop
    | otherwise = lookupAxisLoop blockIdentifier axisIdentifier rest

loopIdentifier :: Loop scope -> IR.LoopId
loopIdentifier (Loop _ identifier) = identifier
