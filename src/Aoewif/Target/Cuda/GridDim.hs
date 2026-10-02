module Aoewif.Target.Cuda.GridDim (
    GridDim (..),
)
where

import           Aoewif.Target.Cuda.Render (Render (..))

data GridDim
    = GridDimX
    | GridDimY
    | GridDimZ
    deriving stock (Eq, Show)

instance Render GridDim where
    render GridDimX = "gridDim.x"
    render GridDimY = "gridDim.y"
    render GridDimZ = "gridDim.z"
