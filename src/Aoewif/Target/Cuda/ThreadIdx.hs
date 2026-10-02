module Aoewif.Target.Cuda.ThreadIdx (
    ThreadIdx (..),
)
where

import           Aoewif.Target.Cuda.Render (Render (..))

data ThreadIdx
    = ThreadIdxX
    | ThreadIdxY
    | ThreadIdxZ
    deriving stock (Eq, Show)

instance Render ThreadIdx where
    render ThreadIdxX = "threadIdx.x"
    render ThreadIdxY = "threadIdx.y"
    render ThreadIdxZ = "threadIdx.z"
