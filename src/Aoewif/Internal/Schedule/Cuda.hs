module Aoewif.Internal.Schedule.Cuda (
    CudaSchedule (..),
) where

import qualified Aoewif.Internal.Compute.IR  as Compute
import qualified Aoewif.Internal.Schedule.IR as Schedule

data CudaSchedule = CudaSchedule
    { cudaScheduleCompute :: Compute.ComputeIR
    , cudaScheduleIR      :: Schedule.ScheduleIR
    }
    deriving stock (Eq, Show)
