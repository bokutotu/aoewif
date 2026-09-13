{-# LANGUAGE QuasiQuotes #-}

module Aoewif.Target.Cuda.Sm90.MBarrier.Render (
    renderFenceMBarrierInit,
    renderMBarrier,
) where

import           Aoewif.Target.Cuda.Codegen                   (indent,
                                                               renderExpr)
import           Aoewif.Target.Cuda.Sm90.Cluster.Render       (clusterAddressConstraint)
import           Aoewif.Target.Cuda.Sm90.MBarrier.Instruction (MBarrier (..),
                                                               MBarrierOp (..))
import           Data.String.Interpolate                      (i)

renderFenceMBarrierInit :: Int -> String
renderFenceMBarrierInit indentation =
    [i|#{padding}cuda::ptx::fence_mbarrier_init(cuda::ptx::sem_release, cuda::ptx::scope_cluster);
|]
  where
    padding = indent indentation

renderMBarrier :: Int -> MBarrierOp -> String
renderMBarrier indentation operation =
    case operation of
        MBarrierInit (MBarrier barrier) arrivalCount ->
            [i|#{padding}cuda::ptx::mbarrier_init(reinterpret_cast<uint64_t*>(&#{renderExpr barrier}), #{renderExpr arrivalCount});
|]
        MBarrierArrive Nothing (MBarrier barrier) Nothing ->
            [i|#{padding}cuda::ptx::mbarrier_arrive(reinterpret_cast<uint64_t*>(&#{renderExpr barrier}));
|]
        MBarrierArrive Nothing (MBarrier barrier) (Just arrivalCount) ->
            [i|#{padding}cuda::ptx::mbarrier_arrive(reinterpret_cast<uint64_t*>(&#{renderExpr barrier}), #{renderExpr arrivalCount});
|]
        MBarrierArrive (Just destination) (MBarrier barrier) Nothing ->
            [i|#{padding}#{renderExpr destination} = cuda::ptx::mbarrier_arrive(reinterpret_cast<uint64_t*>(&#{renderExpr barrier}));
|]
        MBarrierArrive (Just destination) (MBarrier barrier) (Just arrivalCount) ->
            [i|#{padding}#{renderExpr destination} = cuda::ptx::mbarrier_arrive(reinterpret_cast<uint64_t*>(&#{renderExpr barrier}), #{renderExpr arrivalCount});
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
            [i|#{padding}cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared, reinterpret_cast<uint64_t*>(&#{renderExpr barrier}), #{renderExpr transactionCount});
|]
        MBarrierArriveExpectTx (Just destination) (MBarrier barrier) transactionCount ->
            [i|#{padding}#{renderExpr destination} = cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared, reinterpret_cast<uint64_t*>(&#{renderExpr barrier}), #{renderExpr transactionCount});
|]
        MBarrierArriveExpectTxRemote width (MBarrier barrier) transactionCount ->
            [i|#{padding}asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;"
#{operandPadding}:: "#{clusterAddressConstraint width}"(#{renderExpr barrier}), "r"(#{renderExpr transactionCount})
#{operandPadding}: "memory"
#{padding});
|]
        MBarrierExpectTx (MBarrier barrier) transactionCount ->
            [i|#{padding}cuda::ptx::mbarrier_expect_tx(cuda::ptx::sem_relaxed, cuda::ptx::scope_cta, cuda::ptx::space_shared, reinterpret_cast<uint64_t*>(&#{renderExpr barrier}), #{renderExpr transactionCount});
|]
        MBarrierExpectTxRemote width (MBarrier barrier) transactionCount ->
            [i|#{padding}asm volatile("mbarrier.expect_tx.shared::cluster.b64 [%0], %1;"
#{operandPadding}:: "#{clusterAddressConstraint width}"(#{renderExpr barrier}), "r"(#{renderExpr transactionCount})
#{operandPadding}: "memory"
#{padding});
|]
        MBarrierTryWaitParity destination (MBarrier barrier) parity Nothing ->
            [i|#{padding}#{renderExpr destination} = cuda::ptx::mbarrier_try_wait_parity(reinterpret_cast<uint64_t*>(&#{renderExpr barrier}), #{renderExpr parity});
|]
        MBarrierTryWaitParity destination (MBarrier barrier) parity (Just suspendTime) ->
            [i|#{padding}#{renderExpr destination} = cuda::ptx::mbarrier_try_wait_parity(reinterpret_cast<uint64_t*>(&#{renderExpr barrier}), #{renderExpr parity}, #{renderExpr suspendTime});
|]
  where
    padding = indent indentation
    operandPadding = indent (indentation + 1)
