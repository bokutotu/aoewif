{-# LANGUAGE QuasiQuotes #-}

module Aoewif.Target.Cuda.Sm90.Cluster.Render (
    clusterAddressConstraint,
    renderClusterBarrierArrive,
    renderClusterBarrierWait,
    renderClusterSpecialRegister,
    renderGetCtaRank,
    renderMapSharedCluster,
) where

import           Aoewif.Target.Cuda.Codegen                  (indent,
                                                              renderExpr)
import           Aoewif.Target.Cuda.Sm90.Cluster.Instruction (ClusterAddressWidth (..),
                                                              ClusterBarrierArrival (..),
                                                              ClusterDimension (..),
                                                              ClusterSpecialRegister (..))
import           Aoewif.Target.Cuda.Syntax                   (Expr)
import           Data.String.Interpolate                     (i)

renderClusterBarrierArrive :: Int -> ClusterBarrierArrival -> String
renderClusterBarrierArrive indentation arrival =
    case arrival of
        ClusterBarrierRelease ->
            [i|#{padding}asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
|]
        ClusterBarrierRelaxed ->
            [i|#{padding}asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
|]
  where
    padding = indent indentation

renderClusterBarrierWait :: Int -> String
renderClusterBarrierWait indentation =
    [i|#{padding}asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
|]
  where
    padding = indent indentation

renderClusterSpecialRegister :: Int -> Expr -> ClusterSpecialRegister -> String
renderClusterSpecialRegister indentation destination specialRegister =
    case clusterSpecialRegisterInfo specialRegister of
        ClusterU32SpecialRegister registerName ->
            [i|#{padding}asm volatile("mov.u32 %0, %%#{registerName};"
#{operandPadding}: "=r"(#{renderExpr destination})
#{padding});
|]
        ClusterPredicateSpecialRegister registerName ->
            [i|#{padding}asm volatile("{ .reg .pred p; mov.pred p, %%#{registerName}; selp.b32 %0, 1, 0, p; }"
#{operandPadding}: "=r"(#{renderExpr destination})
#{padding});
|]
  where
    padding = indent indentation
    operandPadding = indent (indentation + 1)

data ClusterSpecialRegisterInfo
    = ClusterU32SpecialRegister String
    | ClusterPredicateSpecialRegister String

clusterSpecialRegisterInfo :: ClusterSpecialRegister -> ClusterSpecialRegisterInfo
clusterSpecialRegisterInfo specialRegister =
    case specialRegister of
        ClusterId dimension ->
            ClusterU32SpecialRegister [i|clusterid.#{clusterDimensionTag dimension}|]
        NClusterId dimension ->
            ClusterU32SpecialRegister [i|nclusterid.#{clusterDimensionTag dimension}|]
        ClusterCtaId dimension ->
            ClusterU32SpecialRegister [i|cluster_ctaid.#{clusterDimensionTag dimension}|]
        ClusterNCtaId dimension ->
            ClusterU32SpecialRegister [i|cluster_nctaid.#{clusterDimensionTag dimension}|]
        ClusterCtaRank ->
            ClusterU32SpecialRegister "cluster_ctarank"
        ClusterNCtaRank ->
            ClusterU32SpecialRegister "cluster_nctarank"
        IsExplicitCluster ->
            ClusterPredicateSpecialRegister "is_explicit_cluster"

renderMapSharedCluster :: Int -> ClusterAddressWidth -> Expr -> Expr -> Expr -> String
renderMapSharedCluster indentation width destination source ctaRank =
    [i|#{padding}asm volatile("mapa.shared::cluster.#{widthTag} %0, %1, %2;"
#{operandPadding}: "#{outputConstraint}"(#{renderExpr destination})
#{operandPadding}: "#{inputConstraint}"(#{renderExpr source}), "r"(#{renderExpr ctaRank})
#{padding});
|]
  where
    (widthTag, outputConstraint, inputConstraint) = clusterAddressWidthInfo width
    padding = indent indentation
    operandPadding = indent (indentation + 1)

renderGetCtaRank :: Int -> ClusterAddressWidth -> Expr -> Expr -> String
renderGetCtaRank indentation width destination address =
    [i|#{padding}asm volatile("getctarank.shared::cluster.#{widthTag} %0, %1;"
#{operandPadding}: "=r"(#{renderExpr destination})
#{operandPadding}: "#{inputConstraint}"(#{renderExpr address})
#{padding});
|]
  where
    (widthTag, _, inputConstraint) = clusterAddressWidthInfo width
    padding = indent indentation
    operandPadding = indent (indentation + 1)

clusterDimensionTag :: ClusterDimension -> String
clusterDimensionTag ClusterX = "x"
clusterDimensionTag ClusterY = "y"
clusterDimensionTag ClusterZ = "z"

clusterAddressConstraint :: ClusterAddressWidth -> String
clusterAddressConstraint width =
    inputConstraint
  where
    (_, _, inputConstraint) = clusterAddressWidthInfo width

clusterAddressWidthInfo :: ClusterAddressWidth -> (String, String, String)
clusterAddressWidthInfo ClusterAddressU32 = ("u32", "=r", "r")
clusterAddressWidthInfo ClusterAddressU64 = ("u64", "=l", "l")
