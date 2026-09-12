module Aoewif.Target.Cuda.Sm90.Cluster.Instruction (
    ClusterAddressWidth (..),
    ClusterBarrierArrival (..),
    ClusterDimension (..),
    ClusterShape (..),
    ClusterSpecialRegister (..),
) where

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
