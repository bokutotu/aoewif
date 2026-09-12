module Aoewif.Target.Cuda.Sm90.MBarrier.Instruction (
    MBarrier (..),
    MBarrierOp (..),
) where

import           Aoewif.Target.Cuda.Sm90.Cluster.Instruction (ClusterAddressWidth)
import           Aoewif.Target.Cuda.Syntax                   (Expr)

newtype MBarrier = MBarrier Expr
    deriving stock (Eq, Show)

data MBarrierOp
    = MBarrierInit MBarrier Expr
    | MBarrierArrive (Maybe Expr) MBarrier (Maybe Expr)
    | MBarrierArriveRemote ClusterAddressWidth MBarrier (Maybe Expr)
    | MBarrierArriveExpectTx (Maybe Expr) MBarrier Expr
    | MBarrierArriveExpectTxRemote ClusterAddressWidth MBarrier Expr
    | MBarrierExpectTx MBarrier Expr
    | MBarrierExpectTxRemote ClusterAddressWidth MBarrier Expr
    | MBarrierTryWaitParity Expr MBarrier Expr (Maybe Expr)
    deriving stock (Eq, Show)
