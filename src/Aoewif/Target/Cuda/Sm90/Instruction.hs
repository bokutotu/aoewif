module Aoewif.Target.Cuda.Sm90.Instruction (
    Sm90Op (..),
    BulkWaitMode (..),
    ClusterAddressWidth (..),
    ClusterBarrierArrival (..),
    ClusterDimension (..),
    ClusterShape (..),
    ClusterSpecialRegister (..),
    ElectDestination (..),
    MBarrier (..),
    MBarrierOp (..),
    RegisterAdjustment (..),
    SharedScope (..),
    TmaCachePolicy (..),
    TmaCoordinates (..),
    TmaLoadDestination (..),
    TmaTensorMap (..),
    TmaTensorOp (..),
    WgmmaAccumulatorType (..),
    WgmmaDescriptor (..),
    WgmmaFloatN (..),
    WgmmaFragment (..),
    WgmmaHalfOperands (..),
    WgmmaMma (..),
    WgmmaScale (..),
    WgmmaTranspose (..),
) where

import           Aoewif.Target.Cuda.Sm90.Cluster.Instruction  (ClusterAddressWidth (..),
                                                               ClusterBarrierArrival (..),
                                                               ClusterDimension (..),
                                                               ClusterShape (..),
                                                               ClusterSpecialRegister (..))
import           Aoewif.Target.Cuda.Sm90.MBarrier.Instruction (MBarrier (..),
                                                               MBarrierOp (..))
import           Aoewif.Target.Cuda.Sm90.Tma.Instruction      (BulkWaitMode (..),
                                                               TmaCachePolicy (..),
                                                               TmaCoordinates (..),
                                                               TmaLoadDestination (..),
                                                               TmaTensorMap (..),
                                                               TmaTensorOp (..))
import           Aoewif.Target.Cuda.Sm90.Wgmma.Instruction    (WgmmaAccumulatorType (..),
                                                               WgmmaDescriptor (..),
                                                               WgmmaFloatN (..),
                                                               WgmmaFragment (..),
                                                               WgmmaHalfOperands (..),
                                                               WgmmaMma (..),
                                                               WgmmaScale (..),
                                                               WgmmaTranspose (..))
import           Aoewif.Target.Cuda.Syntax                    (Expr)

data SharedScope
    = SharedCta
    | SharedCluster
    deriving stock (Eq, Show)

data RegisterAdjustment
    = IncreaseRegisters
    | DecreaseRegisters
    deriving stock (Eq, Show)

data ElectDestination
    = ElectPredicate Expr
    | ElectLaneAndPredicate Expr Expr
    deriving stock (Eq, Show)

data Sm90Op
    = WgmmaMmaAsync WgmmaMma
    | WgmmaFence WgmmaMma
    | WgmmaCommitGroup
    | WgmmaWaitGroup Int
    | TmaTensor TmaTensorOp
    | BulkCommitGroup
    | BulkWaitGroup BulkWaitMode Int
    | MBarrierInstruction MBarrierOp
    | FenceProxyAsync SharedScope
    | FenceMBarrierInit
    | ElectSync ElectDestination Expr
    | SetMaxNReg RegisterAdjustment Int
    | ClusterBarrierArrive ClusterBarrierArrival
    | ClusterBarrierWait
    | ReadClusterSpecialRegister Expr ClusterSpecialRegister
    | MapSharedCluster ClusterAddressWidth Expr Expr Expr
    | GetCtaRank ClusterAddressWidth Expr Expr
    deriving stock (Eq, Show)
