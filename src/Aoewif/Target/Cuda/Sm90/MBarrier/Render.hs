{-# LANGUAGE QuasiQuotes #-}

module Aoewif.Target.Cuda.Sm90.MBarrier.Render (
    renderFenceMBarrierInit,
    renderMBarrier,
) where

import           Aoewif.Target.Cuda.Codegen                   (indent,
                                                               renderExpr)
import           Aoewif.Target.Cuda.Sm90.Asm                  (sharedAddress)
import           Aoewif.Target.Cuda.Sm90.Cluster.Render       (clusterAddressConstraint)
import           Aoewif.Target.Cuda.Sm90.MBarrier.Instruction (MBarrier (..),
                                                               MBarrierOp (..))
import           Data.String.Interpolate                      (i)

renderFenceMBarrierInit :: Int -> String
renderFenceMBarrierInit indentation =
    [i|#{padding}asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
|]
  where
    padding = indent indentation

renderMBarrier :: Int -> MBarrierOp -> String
renderMBarrier indentation operation =
    case operation of
        MBarrierInit (MBarrier barrier) arrivalCount ->
            [i|#{padding}asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
#{operandPadding}:: "l"(#{sharedAddress barrier}), "r"(#{renderExpr arrivalCount})
#{operandPadding}: "memory"
#{padding});
|]
        MBarrierArrive Nothing (MBarrier barrier) Nothing ->
            [i|#{padding}asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
#{operandPadding}:: "l"(#{sharedAddress barrier})
#{operandPadding}: "memory"
#{padding});
|]
        MBarrierArrive Nothing (MBarrier barrier) (Just arrivalCount) ->
            [i|#{padding}asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;"
#{operandPadding}:: "l"(#{sharedAddress barrier}), "r"(#{renderExpr arrivalCount})
#{operandPadding}: "memory"
#{padding});
|]
        MBarrierArrive (Just destination) (MBarrier barrier) Nothing ->
            [i|#{padding}asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];"
#{operandPadding}: "=l"(#{renderExpr destination})
#{operandPadding}: "l"(#{sharedAddress barrier})
#{operandPadding}: "memory"
#{padding});
|]
        MBarrierArrive (Just destination) (MBarrier barrier) (Just arrivalCount) ->
            [i|#{padding}asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1], %2;"
#{operandPadding}: "=l"(#{renderExpr destination})
#{operandPadding}: "l"(#{sharedAddress barrier}), "r"(#{renderExpr arrivalCount})
#{operandPadding}: "memory"
#{padding});
|]
        MBarrierArriveRemote width (MBarrier barrier) Nothing ->
            [i|#{padding}asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
#{operandPadding}:: "#{clusterAddressConstraint width}"(#{renderExpr barrier})
#{operandPadding}: "memory"
#{padding});
|]
        MBarrierArriveRemote width (MBarrier barrier) (Just arrivalCount) ->
            [i|#{padding}asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0], %1;"
#{operandPadding}:: "#{clusterAddressConstraint width}"(#{renderExpr barrier}), "r"(#{renderExpr arrivalCount})
#{operandPadding}: "memory"
#{padding});
|]
        MBarrierArriveExpectTx Nothing (MBarrier barrier) transactionCount ->
            [i|#{padding}asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
#{operandPadding}:: "l"(#{sharedAddress barrier}), "r"(#{renderExpr transactionCount})
#{operandPadding}: "memory"
#{padding});
|]
        MBarrierArriveExpectTx (Just destination) (MBarrier barrier) transactionCount ->
            [i|#{padding}asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;"
#{operandPadding}: "=l"(#{renderExpr destination})
#{operandPadding}: "l"(#{sharedAddress barrier}), "r"(#{renderExpr transactionCount})
#{operandPadding}: "memory"
#{padding});
|]
        MBarrierArriveExpectTxRemote width (MBarrier barrier) transactionCount ->
            [i|#{padding}asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;"
#{operandPadding}:: "#{clusterAddressConstraint width}"(#{renderExpr barrier}), "r"(#{renderExpr transactionCount})
#{operandPadding}: "memory"
#{padding});
|]
        MBarrierExpectTx (MBarrier barrier) transactionCount ->
            [i|#{padding}asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;"
#{operandPadding}:: "l"(#{sharedAddress barrier}), "r"(#{renderExpr transactionCount})
#{operandPadding}: "memory"
#{padding});
|]
        MBarrierExpectTxRemote width (MBarrier barrier) transactionCount ->
            [i|#{padding}asm volatile("mbarrier.expect_tx.shared::cluster.b64 [%0], %1;"
#{operandPadding}:: "#{clusterAddressConstraint width}"(#{renderExpr barrier}), "r"(#{renderExpr transactionCount})
#{operandPadding}: "memory"
#{padding});
|]
        MBarrierTryWaitParity destination (MBarrier barrier) parity Nothing ->
            [i|#{padding}asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.b32 %0, 1, 0, p; }"
#{operandPadding}: "=r"(#{renderExpr destination})
#{operandPadding}: "l"(#{sharedAddress barrier}), "r"(#{renderExpr parity})
#{operandPadding}: "memory"
#{padding});
|]
        MBarrierTryWaitParity destination (MBarrier barrier) parity (Just suspendTime) ->
            [i|#{padding}asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3; selp.b32 %0, 1, 0, p; }"
#{operandPadding}: "=r"(#{renderExpr destination})
#{operandPadding}: "l"(#{sharedAddress barrier}), "r"(#{renderExpr parity}), "r"(#{renderExpr suspendTime})
#{operandPadding}: "memory"
#{padding});
|]
  where
    padding = indent indentation
    operandPadding = indent (indentation + 1)
