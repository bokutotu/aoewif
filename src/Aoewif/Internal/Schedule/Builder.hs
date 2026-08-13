{-# LANGUAGE GADTs           #-}
{-# LANGUAGE RoleAnnotations #-}

module Aoewif.Internal.Schedule.Builder (
    Cpu,
    Cuda,
    Schedule,
    Loop,
    CpuSchedule,
    CudaSchedule,
    loop,
    split,
    reorder,
    bind,
    cpu,
    cuda,
    withCpuSchedule,
    withCudaSchedule,
) where

import qualified Aoewif.Internal.Compute       as Compute
import qualified Aoewif.Internal.Schedule.Cpu  as Cpu
import qualified Aoewif.Internal.Schedule.Cuda as Cuda
import qualified Aoewif.Internal.Schedule.IR   as IR
import           Data.Word                     (Word64)

data Cpu

data Cuda

newtype CpuSchedule = CpuSchedule Cpu.CpuSchedule

newtype CudaSchedule = CudaSchedule Cuda.CudaSchedule

newtype Loop scope = Loop IR.LoopId

type role Loop nominal

newtype Schedule backend scope value = Schedule
    { runSchedule :: ScheduleState backend -> Either IR.ScheduleError (value, ScheduleState backend)
    }

type role Schedule nominal nominal representational

data ScheduleState backend where
    CpuState :: IR.LoopPlan -> Cpu.CpuSchedule -> ScheduleState Cpu
    CudaState :: IR.LoopPlan -> Cuda.CudaSchedule -> ScheduleState Cuda

instance Functor (Schedule backend scope) where
    fmap transform (Schedule action) = Schedule $ \state -> do
        (value, nextState) <- action state
        pure (transform value, nextState)

instance Applicative (Schedule backend scope) where
    pure value = Schedule $ \state -> Right (value, state)
    Schedule functionAction <*> Schedule valueAction = Schedule $ \state -> do
        (function, functionState) <- functionAction state
        (value, valueState) <- valueAction functionState
        pure (function value, valueState)

instance Monad (Schedule backend scope) where
    Schedule action >>= next = Schedule $ \state -> do
        (value, nextState) <- action state
        runSchedule (next value) nextState

loop :: Compute.Axis scope Compute.Spatial -> Schedule backend scope (Loop scope)
loop axis = Schedule $ \state -> case state of
    CpuState initialPlan _  -> loopFromPlan axis initialPlan state
    CudaState initialPlan _ -> loopFromPlan axis initialPlan state

loopFromPlan ::
    Compute.Axis scope Compute.Spatial ->
    IR.LoopPlan ->
    ScheduleState backend ->
    Either IR.ScheduleError (Loop scope, ScheduleState backend)
loopFromPlan axis initialPlan state =
    Right (Loop (IR.loopFor (Compute.axisIndexId axis) initialPlan), state)

split :: Loop scope -> Word64 -> Schedule backend scope (Loop scope, Loop scope)
split (Loop loopId) factor = Schedule $ \state -> case state of
    CpuState initialPlan schedule -> do
        (outer, inner, nextSchedule) <- Cpu.splitCpuSchedule loopId factor schedule
        pure ((Loop outer, Loop inner), CpuState initialPlan nextSchedule)
    CudaState initialPlan schedule -> do
        (outer, inner, nextSchedule) <- Cuda.splitCudaSchedule loopId factor schedule
        pure ((Loop outer, Loop inner), CudaState initialPlan nextSchedule)

reorder :: [Loop scope] -> Schedule backend scope ()
reorder loops = Schedule $ \state -> case state of
    CpuState initialPlan schedule -> do
        nextSchedule <- Cpu.reorderCpuSchedule (completeOrder (Cpu.cpuSchedulePlan schedule)) schedule
        pure ((), CpuState initialPlan nextSchedule)
    CudaState initialPlan schedule -> do
        nextSchedule <- Cuda.reorderCudaSchedule (completeOrder (Cuda.cudaSchedulePlan schedule)) schedule
        pure ((), CudaState initialPlan nextSchedule)
  where
    loopIds = map unLoop loops
    unLoop (Loop loopId) = loopId
    completeOrder plan =
        loopIds
            ++ map IR.loopAxisId (filter ((== IR.ReductionLoop) . IR.loopKind) (IR.planLoops plan))

bind :: Loop scope -> IR.CudaBinding -> Schedule Cuda scope ()
bind (Loop loopId) binding = Schedule $ \(CudaState initialPlan schedule) -> do
    nextSchedule <- Cuda.bindCudaSchedule loopId binding schedule
    pure ((), CudaState initialPlan nextSchedule)

cpu ::
    Compute.Program axes ->
    (forall scope. axes scope -> Schedule Cpu scope ()) ->
    Either IR.ScheduleError CpuSchedule
cpu program build = Compute.withProgram program $ \computeProgram computation axes -> do
    let initial = Cpu.newCpuSchedule computeProgram computation
    (_, final) <- executeCpu (build axes) initial
    pure (CpuSchedule final)

cuda ::
    Cuda.CudaTarget ->
    Compute.Program axes ->
    (forall scope. axes scope -> Schedule Cuda scope ()) ->
    Either IR.ScheduleError CudaSchedule
cuda target program build = Compute.withProgram program $ \computeProgram computation axes -> do
    initial <- Cuda.newCudaSchedule computeProgram computation target
    (_, final) <- executeCuda (build axes) initial
    pure (CudaSchedule final)

withCpuSchedule :: CpuSchedule -> (Cpu.CpuSchedule -> result) -> result
withCpuSchedule (CpuSchedule schedule) consume = consume schedule

withCudaSchedule :: CudaSchedule -> (Cuda.CudaSchedule -> result) -> result
withCudaSchedule (CudaSchedule schedule) consume = consume schedule

executeCpu ::
    Schedule Cpu scope value ->
    Cpu.CpuSchedule ->
    Either IR.ScheduleError (value, Cpu.CpuSchedule)
executeCpu action initial = do
    (value, finalState) <- runSchedule action (CpuState (Cpu.cpuSchedulePlan initial) initial)
    case finalState of
        CpuState _ final -> pure (value, final)

executeCuda ::
    Schedule Cuda scope value ->
    Cuda.CudaSchedule ->
    Either IR.ScheduleError (value, Cuda.CudaSchedule)
executeCuda action initial = do
    (value, finalState) <- runSchedule action (CudaState (Cuda.cudaSchedulePlan initial) initial)
    case finalState of
        CudaState _ final -> pure (value, final)
