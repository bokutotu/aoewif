module Aoewif.Target.Cuda.TensorCoreOp (
    RenderOp (..),
    TensorCoreOp (TensorCoreOp),
)
where

class (Show op) => RenderOp op where
    renderOp :: Int -> op -> String

data TensorCoreOp = forall op. (RenderOp op) => TensorCoreOp op

instance Show TensorCoreOp where
    show (TensorCoreOp op) = show op

instance RenderOp TensorCoreOp where
    renderOp indentation (TensorCoreOp op) = renderOp indentation op
