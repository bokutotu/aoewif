module Aoewif.Target.Cuda.BlockDim (
    BlockDim (..),
)
where

import           Aoewif.Target.Cuda.Render (Render (..))

data BlockDim
    = BlockDimX
    | BlockDimY
    | BlockDimZ
    deriving stock (Eq, Show)

instance Render BlockDim where
    render BlockDimX = "blockDim.x"
    render BlockDimY = "blockDim.y"
    render BlockDimZ = "blockDim.z"
