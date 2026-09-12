{-# LANGUAGE QuasiQuotes #-}

module Aoewif.Target.Cuda.Sm90.Tma.Render (
    renderTmaTensor,
    renderBulkCommitGroup,
    renderBulkWaitGroup,
) where

import           Aoewif.Target.Cuda.Codegen                   (indent,
                                                               renderExpr)
import           Aoewif.Target.Cuda.Sm90.Asm                  (constraints,
                                                               placeholders,
                                                               sharedAddress)
import           Aoewif.Target.Cuda.Sm90.Cluster.Render       (clusterAddressConstraint)
import           Aoewif.Target.Cuda.Sm90.MBarrier.Instruction (MBarrier (..))
import           Aoewif.Target.Cuda.Sm90.Tma.Instruction      (BulkWaitMode (..),
                                                               TmaCachePolicy (..),
                                                               TmaCoordinates (..),
                                                               TmaLoadDestination (..),
                                                               TmaTensorMap (..),
                                                               TmaTensorOp (..))
import           Aoewif.Target.Cuda.Syntax                    (Expr)
import           Data.String.Interpolate                      (i)

renderBulkCommitGroup :: Int -> String
renderBulkCommitGroup indentation =
    [i|#{padding}asm volatile("cp.async.bulk.commit_group;" ::: "memory");
|]
  where
    padding = indent indentation

renderBulkWaitGroup :: Int -> BulkWaitMode -> Int -> String
renderBulkWaitGroup indentation mode groupCount =
    case mode of
        BulkWaitComplete ->
            [i|#{padding}asm volatile("cp.async.bulk.wait_group #{groupCount};" ::: "memory");
|]
        BulkWaitRead ->
            [i|#{padding}asm volatile("cp.async.bulk.wait_group.read #{groupCount};" ::: "memory");
|]
  where
    padding = indent indentation

renderTmaTensor :: Int -> TmaTensorOp -> String
renderTmaTensor indentation operation =
    case operation of
        TmaTensorLoad destination tensorMap coordinates barrier cachePolicy ->
            renderTmaTensorLoad
                indentation
                destination
                tensorMap
                coordinates
                barrier
                cachePolicy
        TmaTensorStore tensorMap coordinates source cachePolicy ->
            renderTmaTensorStore
                indentation
                tensorMap
                coordinates
                source
                cachePolicy

renderTmaTensorLoad :: Int -> TmaLoadDestination -> TmaTensorMap -> TmaCoordinates -> MBarrier -> Maybe TmaCachePolicy -> String
renderTmaTensorLoad indentation destination (TmaTensorMap tensorMap) coordinates (MBarrier barrier) cachePolicy =
    case (destination, cachePolicy) of
        (TmaCtaShared destinationAddress, Nothing) ->
            [i|#{padding}asm volatile("cp.async.bulk.tensor.#{dimension}d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {#{coordinatePlaceholders}}], [%#{barrierIndex}];"
#{operandPadding}:: "l"(#{sharedAddress destinationAddress}), "l"(#{renderExpr tensorMap}), #{constraints "r" coordinateExpressions}, "l"(#{sharedAddress barrier})
#{operandPadding}: "memory"
#{padding});
|]
        (TmaCtaShared destinationAddress, Just (TmaCachePolicy policy)) ->
            [i|#{padding}asm volatile("cp.async.bulk.tensor.#{dimension}d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {#{coordinatePlaceholders}}], [%#{barrierIndex}], %#{cacheIndex};"
#{operandPadding}:: "l"(#{sharedAddress destinationAddress}), "l"(#{renderExpr tensorMap}), #{constraints "r" coordinateExpressions}, "l"(#{sharedAddress barrier}), "l"(#{renderExpr policy})
#{operandPadding}: "memory"
#{padding});
|]
        (TmaClusterShared width destinationAddress, Nothing) ->
            [i|#{padding}asm volatile("cp.async.bulk.tensor.#{dimension}d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {#{coordinatePlaceholders}}], [%#{barrierIndex}];"
#{operandPadding}:: "#{clusterAddressConstraint width}"(#{renderExpr destinationAddress}), "l"(#{renderExpr tensorMap}), #{constraints "r" coordinateExpressions}, "#{clusterAddressConstraint width}"(#{renderExpr barrier})
#{operandPadding}: "memory"
#{padding});
|]
        (TmaClusterShared width destinationAddress, Just (TmaCachePolicy policy)) ->
            [i|#{padding}asm volatile("cp.async.bulk.tensor.#{dimension}d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {#{coordinatePlaceholders}}], [%#{barrierIndex}], %#{cacheIndex};"
#{operandPadding}:: "#{clusterAddressConstraint width}"(#{renderExpr destinationAddress}), "l"(#{renderExpr tensorMap}), #{constraints "r" coordinateExpressions}, "#{clusterAddressConstraint width}"(#{renderExpr barrier}), "l"(#{renderExpr policy})
#{operandPadding}: "memory"
#{padding});
|]
        (TmaMulticastClusterShared destinationAddress ctaMask, Nothing) ->
            [i|#{padding}asm volatile("{ .reg .b16 cta_mask; .reg .b16 unused; mov.b32 {cta_mask, unused}, %#{maskIndex}; cp.async.bulk.tensor.#{dimension}d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {#{coordinatePlaceholders}}], [%#{barrierIndex}], cta_mask; }"
#{operandPadding}:: "l"(#{sharedAddress destinationAddress}), "l"(#{renderExpr tensorMap}), #{constraints "r" coordinateExpressions}, "l"(#{sharedAddress barrier}), "r"(#{renderExpr ctaMask})
#{operandPadding}: "memory"
#{padding});
|]
        (TmaMulticastClusterShared destinationAddress ctaMask, Just (TmaCachePolicy policy)) ->
            [i|#{padding}asm volatile("{ .reg .b16 cta_mask; .reg .b16 unused; mov.b32 {cta_mask, unused}, %#{maskIndex}; cp.async.bulk.tensor.#{dimension}d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1, {#{coordinatePlaceholders}}], [%#{barrierIndex}], cta_mask, %#{multicastCacheIndex}; }"
#{operandPadding}:: "l"(#{sharedAddress destinationAddress}), "l"(#{renderExpr tensorMap}), #{constraints "r" coordinateExpressions}, "l"(#{sharedAddress barrier}), "r"(#{renderExpr ctaMask}), "l"(#{renderExpr policy})
#{operandPadding}: "memory"
#{padding});
|]
  where
    (dimension, coordinateExpressions) = tmaCoordinateInfo coordinates
    coordinatePlaceholders = placeholders 2 dimension
    barrierIndex = 2 + dimension
    maskIndex = barrierIndex + 1
    cacheIndex = barrierIndex + 1
    multicastCacheIndex = maskIndex + 1
    padding = indent indentation
    operandPadding = indent (indentation + 1)

renderTmaTensorStore :: Int -> TmaTensorMap -> TmaCoordinates -> Expr -> Maybe TmaCachePolicy -> String
renderTmaTensorStore indentation (TmaTensorMap tensorMap) coordinates source cachePolicy =
    case cachePolicy of
        Nothing ->
            [i|#{padding}asm volatile("cp.async.bulk.tensor.#{dimension}d.global.shared::cta.bulk_group [%0, {#{coordinatePlaceholders}}], [%#{sourceIndex}];"
#{operandPadding}:: "l"(#{renderExpr tensorMap}), #{constraints "r" coordinateExpressions}, "l"(#{sharedAddress source})
#{operandPadding}: "memory"
#{padding});
|]
        Just (TmaCachePolicy policy) ->
            [i|#{padding}asm volatile("cp.async.bulk.tensor.#{dimension}d.global.shared::cta.bulk_group.L2::cache_hint [%0, {#{coordinatePlaceholders}}], [%#{sourceIndex}], %#{cacheIndex};"
#{operandPadding}:: "l"(#{renderExpr tensorMap}), #{constraints "r" coordinateExpressions}, "l"(#{sharedAddress source}), "l"(#{renderExpr policy})
#{operandPadding}: "memory"
#{padding});
|]
  where
    (dimension, coordinateExpressions) = tmaCoordinateInfo coordinates
    coordinatePlaceholders = placeholders 1 dimension
    sourceIndex = 1 + dimension
    cacheIndex = sourceIndex + 1
    padding = indent indentation
    operandPadding = indent (indentation + 1)

tmaCoordinateInfo :: TmaCoordinates -> (Int, [Expr])
tmaCoordinateInfo coordinates =
    case coordinates of
        TmaCoordinates1D coordinate0 ->
            (1, [coordinate0])
        TmaCoordinates2D coordinate0 coordinate1 ->
            (2, [coordinate0, coordinate1])
        TmaCoordinates3D coordinate0 coordinate1 coordinate2 ->
            (3, [coordinate0, coordinate1, coordinate2])
        TmaCoordinates4D coordinate0 coordinate1 coordinate2 coordinate3 ->
            (4, [coordinate0, coordinate1, coordinate2, coordinate3])
        TmaCoordinates5D coordinate0 coordinate1 coordinate2 coordinate3 coordinate4 ->
            (5, [coordinate0, coordinate1, coordinate2, coordinate3, coordinate4])
