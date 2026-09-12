{-# OPTIONS_GHC -Wno-orphans #-}

module Aoewif.Target.Cuda.Sm90.Render () where

import           Aoewif.Target.Cuda.Sm90.Asm             (asmLine, exprOperand,
                                                          renderAsm)
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
            FenceProxyAsync scope ->
                asmLine
                    indentation
                    ("fence.proxy.async" ++ sharedScopeTag scope ++ ";")
                    ["memory"]
            FenceMBarrierInit ->
                renderFenceMBarrierInit indentation
            ElectSync destination memberMask ->
                renderElectSync indentation destination memberMask
            SetMaxNReg adjustment registerCount ->
                asmLine
                    indentation
                    ( "setmaxnreg."
                        ++ registerAdjustmentTag adjustment
                        ++ ".sync.aligned.u32 "
                        ++ show registerCount
                        ++ ";"
                    )
                    ["memory"]
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

renderElectSync :: Int -> ElectDestination -> Expr -> String
renderElectSync indentation destination memberMask =
    case destination of
        ElectPredicate predicate ->
            renderAsm
                indentation
                "{ .reg .pred p; elect.sync _|p, %1; selp.b32 %0, 1, 0, p; }"
                [exprOperand "=r" predicate]
                [exprOperand "r" memberMask]
                []
        ElectLaneAndPredicate lane predicate ->
            renderAsm
                indentation
                "{ .reg .pred p; elect.sync %0|p, %2; selp.b32 %1, 1, 0, p; }"
                [exprOperand "=r" lane, exprOperand "=r" predicate]
                [exprOperand "r" memberMask]
                []

sharedScopeTag :: SharedScope -> String
sharedScopeTag SharedCta     = ".shared::cta"
sharedScopeTag SharedCluster = ".shared::cluster"

registerAdjustmentTag :: RegisterAdjustment -> String
registerAdjustmentTag IncreaseRegisters = "inc"
registerAdjustmentTag DecreaseRegisters = "dec"
