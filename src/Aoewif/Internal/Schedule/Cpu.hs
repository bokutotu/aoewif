module Aoewif.Internal.Schedule.Cpu (
    CpuSchedule (..),
) where

import qualified Aoewif.Internal.Compute.IR  as Compute
import qualified Aoewif.Internal.Schedule.IR as Schedule

data CpuSchedule = CpuSchedule
    { cpuScheduleCompute :: Compute.ComputeIR
    , cpuScheduleIR      :: Schedule.ScheduleIR
    }
    deriving stock (Eq, Show)
