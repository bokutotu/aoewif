module Aoewif.Target.Cuda.TensorCoreOp (
    RenderOp (..),
    TensorCoreOp (TensorCoreOp),
)
where

-- An op is any target-specific instruction family that renders itself as a
-- complete, newline-terminated statement. The Int is the current indentation
-- level; the renderer owns every byte of its output, including indentation.
class (Show op) => RenderOp op where
    renderOp :: Int -> op -> String

-- The single extension socket of the IR. Each instruction generation
-- (Ampere, Hopper, ...) defines its own op type with a RenderOp instance;
-- the core syntax never learns about any of them.
data TensorCoreOp = forall op. (RenderOp op) => TensorCoreOp op

instance Show TensorCoreOp where
    show (TensorCoreOp op) = show op

-- Consumers render TensorCoreOp values through the class; the wrapped op's own
-- instance does the work, so the existential never has to be unpacked.
instance RenderOp TensorCoreOp where
    renderOp indentation (TensorCoreOp op) = renderOp indentation op
