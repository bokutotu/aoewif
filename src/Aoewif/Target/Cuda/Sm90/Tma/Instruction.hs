module Aoewif.Target.Cuda.Sm90.Tma.Instruction (
    BulkWaitMode (..),
    TmaCachePolicy (..),
    TmaCoordinates (..),
    TmaLoadDestination (..),
    TmaTensorMap (..),
    TmaTensorOp (..),
) where

import           Aoewif.Target.Cuda.Sm90.Cluster.Instruction  (ClusterAddressWidth)
import           Aoewif.Target.Cuda.Sm90.MBarrier.Instruction (MBarrier)
import           Aoewif.Target.Cuda.Syntax                    (Expr)

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

data BulkWaitMode
    = BulkWaitComplete
    | BulkWaitRead
    deriving stock (Eq, Show)
