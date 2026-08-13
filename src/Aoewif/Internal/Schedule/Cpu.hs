module Aoewif.Internal.Schedule.Cpu (
    CpuSchedule,
    newCpuSchedule,
    cpuScheduleProgram,
    cpuScheduleCompute,
    cpuSchedulePlan,
    splitCpuSchedule,
    reorderCpuSchedule,
) where

import qualified Aoewif.Internal.Compute.IR  as Compute
import           Aoewif.Internal.Schedule.IR (LoopId, LoopPlan, ScheduleError,
                                              createLoopPlan, reorderLoopPlan,
                                              splitLoopPlan)
import           Data.Word                   (Word64)

data CpuSchedule = CpuSchedule
    { internalCpuScheduleProgram :: Compute.Program
    , internalCpuScheduleCompute :: Compute.Compute
    , internalCpuSchedulePlan    :: LoopPlan
    }
    deriving stock (Eq, Show)

newCpuSchedule :: Compute.Program -> Compute.Compute -> CpuSchedule
newCpuSchedule program computation =
    CpuSchedule
        { internalCpuScheduleProgram = program
        , internalCpuScheduleCompute = computation
        , internalCpuSchedulePlan = createLoopPlan computation
        }

cpuScheduleProgram :: CpuSchedule -> Compute.Program
cpuScheduleProgram = internalCpuScheduleProgram

cpuScheduleCompute :: CpuSchedule -> Compute.Compute
cpuScheduleCompute = internalCpuScheduleCompute

cpuSchedulePlan :: CpuSchedule -> LoopPlan
cpuSchedulePlan = internalCpuSchedulePlan

splitCpuSchedule :: LoopId -> Word64 -> CpuSchedule -> Either ScheduleError (LoopId, LoopId, CpuSchedule)
splitCpuSchedule loopId factor schedule = do
    (outer, inner, plan) <- splitLoopPlan loopId factor (cpuSchedulePlan schedule)
    pure (outer, inner, schedule{internalCpuSchedulePlan = plan})

reorderCpuSchedule :: [LoopId] -> CpuSchedule -> Either ScheduleError CpuSchedule
reorderCpuSchedule order schedule = do
    plan <- reorderLoopPlan order (cpuSchedulePlan schedule)
    pure schedule{internalCpuSchedulePlan = plan}
