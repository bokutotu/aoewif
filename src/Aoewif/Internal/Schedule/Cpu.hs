module Aoewif.Internal.Schedule.Cpu (
    CpuSchedule,
    newCpuSchedule,
    cpuScheduleFunction,
    cpuScheduleOperation,
    cpuSchedulePlan,
    splitCpuSchedule,
    reorderCpuSchedule,
)
where

import           Aoewif.Internal.IR            (ComputeFunction, ComputeOp,
                                                ComputeOpId)
import           Aoewif.Internal.Schedule.Base (LoopId, LoopPlan, ScheduleError,
                                                createLoopPlan, operationFor,
                                                reorderLoopPlan, splitLoopPlan)
import           Data.Word                     (Word64)

data CpuSchedule = CpuSchedule
    { internalCpuScheduleFunction  :: ComputeFunction
    , internalCpuScheduleOperation :: ComputeOp
    , internalCpuSchedulePlan      :: LoopPlan
    }
    deriving (Eq, Show)

cpuScheduleFunction :: CpuSchedule -> ComputeFunction
cpuScheduleFunction = internalCpuScheduleFunction

cpuSchedulePlan :: CpuSchedule -> LoopPlan
cpuSchedulePlan = internalCpuSchedulePlan

newCpuSchedule :: ComputeFunction -> ComputeOpId -> Either ScheduleError CpuSchedule
newCpuSchedule function operationId = do
    operation <- operationFor function operationId
    pure
        CpuSchedule
            { internalCpuScheduleFunction = function
            , internalCpuScheduleOperation = operation
            , internalCpuSchedulePlan = createLoopPlan operation
            }

cpuScheduleOperation :: CpuSchedule -> ComputeOp
cpuScheduleOperation = internalCpuScheduleOperation

splitCpuSchedule :: LoopId -> Word64 -> CpuSchedule -> Either ScheduleError (LoopId, LoopId, CpuSchedule)
splitCpuSchedule loopId factor schedule = do
    (outer, inner, plan) <- splitLoopPlan loopId factor (cpuSchedulePlan schedule)
    pure (outer, inner, schedule{internalCpuSchedulePlan = plan})

reorderCpuSchedule :: [LoopId] -> CpuSchedule -> Either ScheduleError CpuSchedule
reorderCpuSchedule order schedule = do
    plan <- reorderLoopPlan order (cpuSchedulePlan schedule)
    pure schedule{internalCpuSchedulePlan = plan}
