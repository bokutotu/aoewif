module Aoewif.Target.Cuda.DSL.CpAsync (
    CpAsync (..),
    CpAsyncCopy (..),
    CpAsyncSource (..),
    L2PrefetchSize (..),
    ca4,
    ca8,
    ca16,
    cg16,
    cpAsync,
    cpAsyncCommitGroup,
    cpAsyncWaitGroup,
)
where

import           Aoewif.Target.Cuda.AsmOperand  (AsmInput)
import           Aoewif.Target.Cuda.CpAsync     (CpAsync (..), CpAsyncCopy (..),
                                                 CpAsyncSource (..),
                                                 L2PrefetchSize (..))
import           Aoewif.Target.Cuda.DSL.Builder (Block, expr_)
import           Aoewif.Target.Cuda.Expr        (Expr (..))

cg16, ca4, ca8, ca16 :: AsmInput -> AsmInput -> CpAsync
cg16 dst src = CpAsync Cg16 dst src EntireSource Nothing
ca4 dst src = CpAsync Ca4 dst src EntireSource Nothing
ca8 dst src = CpAsync Ca8 dst src EntireSource Nothing
ca16 dst src = CpAsync Ca16 dst src EntireSource Nothing

cpAsync :: CpAsync -> Block ()
cpAsync = expr_ . CpAsyncExpr

cpAsyncCommitGroup :: Block ()
cpAsyncCommitGroup = expr_ CpAsyncCommitGroup

cpAsyncWaitGroup :: Integer -> Block ()
cpAsyncWaitGroup = expr_ . CpAsyncWaitGroup
