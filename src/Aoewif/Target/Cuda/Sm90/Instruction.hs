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

import           Aoewif.Target.Cuda.Syntax (Expr)

data WgmmaFloatN
    = WgmmaFloatN8
    | WgmmaFloatN16
    | WgmmaFloatN24
    | WgmmaFloatN32
    | WgmmaFloatN40
    | WgmmaFloatN48
    | WgmmaFloatN56
    | WgmmaFloatN64
    | WgmmaFloatN72
    | WgmmaFloatN80
    | WgmmaFloatN88
    | WgmmaFloatN96
    | WgmmaFloatN104
    | WgmmaFloatN112
    | WgmmaFloatN120
    | WgmmaFloatN128
    | WgmmaFloatN136
    | WgmmaFloatN144
    | WgmmaFloatN152
    | WgmmaFloatN160
    | WgmmaFloatN168
    | WgmmaFloatN176
    | WgmmaFloatN184
    | WgmmaFloatN192
    | WgmmaFloatN200
    | WgmmaFloatN208
    | WgmmaFloatN216
    | WgmmaFloatN224
    | WgmmaFloatN232
    | WgmmaFloatN240
    | WgmmaFloatN248
    | WgmmaFloatN256
    deriving stock (Eq, Show)

data WgmmaAccumulatorType
    = WgmmaAccumulatorF16
    | WgmmaAccumulatorF32
    deriving stock (Eq, Show)

data WgmmaScale
    = WgmmaScaleOne
    | WgmmaScaleNegativeOne
    deriving stock (Eq, Show)

data WgmmaTranspose
    = WgmmaNotTransposed
    | WgmmaTransposed
    deriving stock (Eq, Show)

newtype WgmmaDescriptor = WgmmaDescriptor Expr
    deriving stock (Eq, Show)

newtype WgmmaFragment = WgmmaFragment
    { wgmmaFragmentRegisters :: [Expr]
    }
    deriving stock (Eq, Show)

data WgmmaHalfOperands
    = WgmmaHalfSharedOperands
        WgmmaDescriptor
        WgmmaDescriptor
        WgmmaTranspose
        WgmmaTranspose
    | WgmmaHalfRegisterOperands
        WgmmaFragment
        WgmmaDescriptor
        WgmmaTranspose
    deriving stock (Eq, Show)

data WgmmaMma
    = WgmmaF16
        WgmmaFloatN
        WgmmaAccumulatorType
        WgmmaFragment
        WgmmaHalfOperands
        Expr
        WgmmaScale
        WgmmaScale
    | WgmmaBF16
        WgmmaFloatN
        WgmmaFragment
        WgmmaHalfOperands
        Expr
        WgmmaScale
        WgmmaScale
    deriving stock (Eq, Show)

newtype TmaTensorMap = TmaTensorMap Expr
    deriving stock (Eq, Show)

newtype TmaCachePolicy = TmaCachePolicy Expr
    deriving stock (Eq, Show)

data TmaCoordinates
    = TmaCoordinates1D Expr
    | TmaCoordinates2D Expr Expr
    | TmaCoordinates3D Expr Expr Expr
    | TmaCoordinates4D Expr Expr Expr Expr
    | TmaCoordinates5D Expr Expr Expr Expr Expr
    deriving stock (Eq, Show)

data TmaLoadDestination
    = TmaCtaShared Expr
    | TmaClusterShared ClusterAddressWidth Expr
    | TmaMulticastClusterShared Expr Expr
    deriving stock (Eq, Show)

data TmaTensorOp
    = TmaTensorLoad
        TmaLoadDestination
        TmaTensorMap
        TmaCoordinates
        MBarrier
        (Maybe TmaCachePolicy)
    | TmaTensorStore
        TmaTensorMap
        TmaCoordinates
        Expr
        (Maybe TmaCachePolicy)
    deriving stock (Eq, Show)

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

data BulkWaitMode
    = BulkWaitComplete
    | BulkWaitRead
    deriving stock (Eq, Show)

data SharedScope
    = SharedCta
    | SharedCluster
    deriving stock (Eq, Show)

data RegisterAdjustment
    = IncreaseRegisters
    | DecreaseRegisters
    deriving stock (Eq, Show)

data ClusterShape = ClusterShape
    { clusterShapeX :: Int
    , clusterShapeY :: Int
    , clusterShapeZ :: Int
    }
    deriving stock (Eq, Show)

data ClusterDimension
    = ClusterX
    | ClusterY
    | ClusterZ
    deriving stock (Eq, Show)

data ClusterSpecialRegister
    = ClusterId ClusterDimension
    | NClusterId ClusterDimension
    | ClusterCtaId ClusterDimension
    | ClusterNCtaId ClusterDimension
    | ClusterCtaRank
    | ClusterNCtaRank
    | IsExplicitCluster
    deriving stock (Eq, Show)

data ClusterBarrierArrival
    = ClusterBarrierRelease
    | ClusterBarrierRelaxed
    deriving stock (Eq, Show)

data ClusterAddressWidth
    = ClusterAddressU32
    | ClusterAddressU64
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
