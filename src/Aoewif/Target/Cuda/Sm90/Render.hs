{-# LANGUAGE QuasiQuotes #-}
{-# OPTIONS_GHC -Wno-orphans #-}

module Aoewif.Target.Cuda.Sm90.Render () where

import           Aoewif.Target.Cuda.Codegen              (indent, renderExpr)
import           Aoewif.Target.Cuda.Sm90.Cluster.Render  (renderClusterBarrierArrive,
                                                          renderClusterBarrierWait,
                                                          renderClusterSpecialRegister,
                                                          renderGetCtaRank,
                                                          renderMapSharedCluster)
import           Aoewif.Target.Cuda.Sm90.Instruction     (ElectDestination (..),
                                                          RegisterAdjustment (..),
                                                          SharedScope (..),
                                                          Sm90Op (..))
import           Aoewif.Target.Cuda.Sm90.MBarrier.Render (renderFenceMBarrierInit,
                                                          renderMBarrier)
import           Aoewif.Target.Cuda.Sm90.Tma.Render      (renderBulkCommitGroup,
                                                          renderBulkWaitGroup,
                                                          renderTmaTensor)
import           Aoewif.Target.Cuda.Sm90.Wgmma.Render    (renderWgmmaCommitGroup,
                                                          renderWgmmaFence,
                                                          renderWgmmaMmaAsync,
                                                          renderWgmmaWaitGroup)
import           Aoewif.Target.Cuda.Syntax               (Expr)
import           Aoewif.Target.Cuda.TensorCoreOp         (RenderOp (..))
import           Data.String.Interpolate                 (i)

instance RenderOp Sm90Op where
    renderOp indentation operation =
        case operation of
            WgmmaMmaAsync mmaOperation ->
                renderWgmmaMmaAsync indentation mmaOperation
            WgmmaFence mmaOperation ->
                renderWgmmaFence indentation mmaOperation
            WgmmaCommitGroup ->
                renderWgmmaCommitGroup indentation
            WgmmaWaitGroup groupCount ->
                renderWgmmaWaitGroup indentation groupCount
            TmaTensor tensorOperation ->
                renderTmaTensor indentation tensorOperation
            BulkCommitGroup ->
                renderBulkCommitGroup indentation
            BulkWaitGroup mode groupCount ->
                renderBulkWaitGroup indentation mode groupCount
            MBarrierInstruction barrierOperation ->
                renderMBarrier indentation barrierOperation
            FenceProxyAsync SharedCta ->
                [i|#{padding}cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
|]
            FenceProxyAsync SharedCluster ->
                [i|#{padding}cuda::ptx::fence_proxy_async(cuda::ptx::space_cluster);
|]
            FenceMBarrierInit ->
                renderFenceMBarrierInit indentation
            ElectSync destination memberMask ->
                renderElectSync indentation destination memberMask
            SetMaxNReg IncreaseRegisters registerCount ->
                [i|#{padding}asm volatile("setmaxnreg.inc.sync.aligned.u32 #{registerCount};" ::: "memory");
|]
            SetMaxNReg DecreaseRegisters registerCount ->
                [i|#{padding}asm volatile("setmaxnreg.dec.sync.aligned.u32 #{registerCount};" ::: "memory");
|]
            ClusterBarrierArrive arrival ->
                renderClusterBarrierArrive indentation arrival
            ClusterBarrierWait ->
                renderClusterBarrierWait indentation
            ReadClusterSpecialRegister destination specialRegister ->
                renderClusterSpecialRegister indentation destination specialRegister
            MapSharedCluster width destination source ctaRank ->
                renderMapSharedCluster indentation width destination source ctaRank
            GetCtaRank width destination address ->
                renderGetCtaRank indentation width destination address
            StMatrix base rowStride source ->
                renderStMatrix indentation base rowStride source
      where
        padding = indent indentation

renderStMatrix :: Int -> Expr -> Expr -> Expr -> String
renderStMatrix indentation base rowStride source =
    [i|#{padding}// Derive the lane ID automatically from the block-local linear thread ID.
#{padding}asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};"
#{operandPadding}:: "r"(static_cast<uint32_t>(__cvta_generic_to_shared((#{renderExpr base}) + ((threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) % 32 % 8) * (#{renderExpr rowStride})))), "r"(#{renderExpr source})
#{operandPadding}: "memory"
#{padding});
|]
  where
    padding = indent indentation
    operandPadding = indent (indentation + 1)

renderElectSync :: Int -> ElectDestination -> Expr -> String
renderElectSync indentation destination memberMask =
    case destination of
        ElectPredicate predicate ->
            [i|#{padding}asm volatile("{ .reg .pred p; elect.sync _|p, %1; selp.b32 %0, 1, 0, p; }"
#{operandPadding}: "=r"(#{renderExpr predicate})
#{operandPadding}: "r"(#{renderExpr memberMask})
#{padding});
|]
        ElectLaneAndPredicate lane predicate ->
            [i|#{padding}asm volatile("{ .reg .pred p; elect.sync %0|p, %2; selp.b32 %1, 1, 0, p; }"
#{operandPadding}: "=r"(#{renderExpr lane}), "=r"(#{renderExpr predicate})
#{operandPadding}: "r"(#{renderExpr memberMask})
#{padding});
|]
  where
    padding = indent indentation
    operandPadding = indent (indentation + 1)
