module Aoewif.Target.Cuda.Tcgen05.Instruction (
    Tcgen05Op (..),
) where

import           Aoewif.Target.Cuda.Syntax (Expr)

data Tcgen05Op
    = Tcgen05Alloc Expr Expr
    | Tcgen05FenceAfterThreadSync
    | Tcgen05Mma Expr Expr Expr Expr Expr Expr
    deriving stock (Eq, Show)
