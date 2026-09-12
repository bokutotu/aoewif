module Aoewif.Target.Cuda.Sm90.Cluster.Render (
    clusterAddressOperand,
    renderClusterBarrierArrive,
    renderClusterBarrierWait,
    renderClusterSpecialRegister,
    renderGetCtaRank,
    renderMapSharedCluster,
) where

import           Aoewif.Target.Cuda.Sm90.Asm                 (AsmOperand,
                                                              asmLine,
                                                              exprOperand,
                                                              renderAsm)
import           Aoewif.Target.Cuda.Sm90.Cluster.Instruction (ClusterAddressWidth (..),
                                                              ClusterBarrierArrival (..),
                                                              ClusterDimension (..),
                                                              ClusterSpecialRegister (..))
import           Aoewif.Target.Cuda.Syntax                   (Expr)

renderClusterBarrierArrive :: Int -> ClusterBarrierArrival -> String
renderClusterBarrierArrive indentation arrival =
    asmLine
        indentation
        ( "barrier.cluster.arrive."
            ++ clusterBarrierArrivalTag arrival
            ++ ".aligned;"
        )
        ["memory"]

renderClusterBarrierWait :: Int -> String
renderClusterBarrierWait indentation =
    asmLine
        indentation
        "barrier.cluster.wait.acquire.aligned;"
        ["memory"]

renderClusterSpecialRegister :: Int -> Expr -> ClusterSpecialRegister -> String
renderClusterSpecialRegister indentation destination specialRegister =
    case clusterSpecialRegisterInfo specialRegister of
        ClusterU32SpecialRegister registerName ->
            renderAsm
                indentation
                ("mov.u32 %0, %%" ++ registerName ++ ";")
                [exprOperand "=r" destination]
                []
                []
        ClusterPredicateSpecialRegister registerName ->
            renderAsm
                indentation
                ( "{ .reg .pred p; mov.pred p, %%"
                    ++ registerName
                    ++ "; selp.b32 %0, 1, 0, p; }"
                )
                [exprOperand "=r" destination]
                []
                []

data ClusterSpecialRegisterInfo
    = ClusterU32SpecialRegister String
    | ClusterPredicateSpecialRegister String

clusterSpecialRegisterInfo :: ClusterSpecialRegister -> ClusterSpecialRegisterInfo
clusterSpecialRegisterInfo specialRegister =
    case specialRegister of
        ClusterId dimension ->
            ClusterU32SpecialRegister ("clusterid." ++ clusterDimensionTag dimension)
        NClusterId dimension ->
            ClusterU32SpecialRegister ("nclusterid." ++ clusterDimensionTag dimension)
        ClusterCtaId dimension ->
            ClusterU32SpecialRegister ("cluster_ctaid." ++ clusterDimensionTag dimension)
        ClusterNCtaId dimension ->
            ClusterU32SpecialRegister ("cluster_nctaid." ++ clusterDimensionTag dimension)
        ClusterCtaRank ->
            ClusterU32SpecialRegister "cluster_ctarank"
        ClusterNCtaRank ->
            ClusterU32SpecialRegister "cluster_nctarank"
        IsExplicitCluster ->
            ClusterPredicateSpecialRegister "is_explicit_cluster"

renderMapSharedCluster :: Int -> ClusterAddressWidth -> Expr -> Expr -> Expr -> String
renderMapSharedCluster indentation width destination source ctaRank =
    renderAsm
        indentation
        ("mapa.shared::cluster." ++ widthTag ++ " %0, %1, %2;")
        [exprOperand outputConstraint destination]
        [exprOperand inputConstraint source, exprOperand "r" ctaRank]
        []
  where
    (widthTag, outputConstraint, inputConstraint) = clusterAddressWidthInfo width

renderGetCtaRank :: Int -> ClusterAddressWidth -> Expr -> Expr -> String
renderGetCtaRank indentation width destination address =
    renderAsm
        indentation
        ("getctarank.shared::cluster." ++ widthTag ++ " %0, %1;")
        [exprOperand "=r" destination]
        [exprOperand inputConstraint address]
        []
  where
    (widthTag, _, inputConstraint) = clusterAddressWidthInfo width

clusterBarrierArrivalTag :: ClusterBarrierArrival -> String
clusterBarrierArrivalTag ClusterBarrierRelease = "release"
clusterBarrierArrivalTag ClusterBarrierRelaxed = "relaxed"

clusterDimensionTag :: ClusterDimension -> String
clusterDimensionTag ClusterX = "x"
clusterDimensionTag ClusterY = "y"
clusterDimensionTag ClusterZ = "z"

clusterAddressOperand :: ClusterAddressWidth -> Expr -> AsmOperand
clusterAddressOperand width =
    exprOperand inputConstraint
  where
    (_, _, inputConstraint) = clusterAddressWidthInfo width

clusterAddressWidthInfo :: ClusterAddressWidth -> (String, String, String)
clusterAddressWidthInfo ClusterAddressU32 = ("u32", "=r", "r")
clusterAddressWidthInfo ClusterAddressU64 = ("u64", "=l", "l")
