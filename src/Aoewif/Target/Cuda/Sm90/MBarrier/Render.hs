module Aoewif.Target.Cuda.Sm90.MBarrier.Render (
    renderFenceMBarrierInit,
    renderMBarrier,
) where

import           Aoewif.Target.Cuda.Sm90.Asm                  (AsmOperand,
                                                               asmLine,
                                                               exprOperand,
                                                               localSharedOperand,
                                                               renderAsm)
import           Aoewif.Target.Cuda.Sm90.Cluster.Render       (clusterAddressOperand)
import           Aoewif.Target.Cuda.Sm90.MBarrier.Instruction (MBarrier (..),
                                                               MBarrierOp (..))
import           Aoewif.Target.Cuda.Syntax                    (Expr)

renderFenceMBarrierInit :: Int -> String
renderFenceMBarrierInit indentation =
    asmLine
        indentation
        "fence.mbarrier_init.release.cluster;"
        ["memory"]

renderMBarrier :: Int -> MBarrierOp -> String
renderMBarrier indentation operation =
    case operation of
        MBarrierInit (MBarrier barrier) arrivalCount ->
            renderAsm
                indentation
                "mbarrier.init.shared::cta.b64 [%0], %1;"
                []
                [localSharedOperand barrier, exprOperand "r" arrivalCount]
                ["memory"]
        MBarrierArrive destination (MBarrier barrier) arrivalCount ->
            renderMBarrierArrive
                indentation
                "mbarrier.arrive.shared::cta.b64"
                destination
                (localSharedOperand barrier)
                arrivalCount
        MBarrierArriveRemote width (MBarrier barrier) arrivalCount ->
            renderMBarrierArrive
                indentation
                "mbarrier.arrive.shared::cluster.b64"
                Nothing
                (clusterAddressOperand width barrier)
                arrivalCount
        MBarrierArriveExpectTx destination (MBarrier barrier) transactionCount ->
            renderMBarrierArrive
                indentation
                "mbarrier.arrive.expect_tx.shared::cta.b64"
                destination
                (localSharedOperand barrier)
                (Just transactionCount)
        MBarrierArriveExpectTxRemote width (MBarrier barrier) transactionCount ->
            renderMBarrierArrive
                indentation
                "mbarrier.arrive.expect_tx.shared::cluster.b64"
                Nothing
                (clusterAddressOperand width barrier)
                (Just transactionCount)
        MBarrierExpectTx (MBarrier barrier) transactionCount ->
            renderAsm
                indentation
                "mbarrier.expect_tx.shared::cta.b64 [%0], %1;"
                []
                [localSharedOperand barrier, exprOperand "r" transactionCount]
                ["memory"]
        MBarrierExpectTxRemote width (MBarrier barrier) transactionCount ->
            renderAsm
                indentation
                "mbarrier.expect_tx.shared::cluster.b64 [%0], %1;"
                []
                [clusterAddressOperand width barrier, exprOperand "r" transactionCount]
                ["memory"]
        MBarrierTryWaitParity destination (MBarrier barrier) parity suspendTime ->
            renderAsm
                indentation
                ( "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2"
                    ++ maybe "" (const ", %3") suspendTime
                    ++ "; selp.b32 %0, 1, 0, p; }"
                )
                [exprOperand "=r" destination]
                ( [localSharedOperand barrier, exprOperand "r" parity]
                    ++ maybe [] (pure . exprOperand "r") suspendTime
                )
                ["memory"]

renderMBarrierArrive :: Int -> String -> Maybe Expr -> AsmOperand -> Maybe Expr -> String
renderMBarrierArrive indentation instructionTag destination barrierOperand count =
    renderAsm indentation instruction outputOperands inputOperands ["memory"]
  where
    outputOperands = maybe [] (pure . exprOperand "=l") destination
    firstInputIndex = length outputOperands
    inputOperands =
        barrierOperand : maybe [] (pure . exprOperand "r") count
    instruction =
        instructionTag
            ++ " "
            ++ maybe "_" (const "%0") destination
            ++ ", [%"
            ++ show firstInputIndex
            ++ "]"
            ++ maybe "" (const (", %" ++ show (firstInputIndex + 1))) count
            ++ ";"
