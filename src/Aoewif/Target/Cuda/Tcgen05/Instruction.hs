module Aoewif.Target.Cuda.Tcgen05.Instruction (
    Tcgen05Op (..),
) where

import           Aoewif.Target.Cuda.Syntax (Expr)

data Tcgen05Op
    = Tcgen05Alloc Expr Expr
    deriving stock (Eq, Show)
