module Aoewif.Target.Cuda.BlockIdx (
    BlockIdx (..),
)
where

import           Aoewif.Target.Cuda.Render (Render (..))

data BlockIdx
    = BlockIdxX
    | BlockIdxY
    | BlockIdxZ
    deriving stock (Eq, Show)

instance Render BlockIdx where
    render BlockIdxX = "blockIdx.x"
    render BlockIdxY = "blockIdx.y"
    render BlockIdxZ = "blockIdx.z"
