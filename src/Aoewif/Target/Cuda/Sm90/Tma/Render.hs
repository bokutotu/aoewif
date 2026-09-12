module Aoewif.Target.Cuda.Sm90.Tma.Render (
    renderTmaTensor,
    renderBulkCommitGroup,
    renderBulkWaitGroup,
) where

import           Aoewif.Target.Cuda.Sm90.Asm                  (asmLine,
                                                               exprOperand,
                                                               localSharedOperand,
                                                               placeholders,
                                                               renderAsm)
import           Aoewif.Target.Cuda.Sm90.Cluster.Render       (clusterAddressOperand)
import           Aoewif.Target.Cuda.Sm90.MBarrier.Instruction (MBarrier (..))
import           Aoewif.Target.Cuda.Sm90.Tma.Instruction      (BulkWaitMode (..),
                                                               TmaCachePolicy (..),
                                                               TmaCoordinates (..),
                                                               TmaLoadDestination (..),
                                                               TmaTensorMap (..),
                                                               TmaTensorOp (..))
import           Aoewif.Target.Cuda.Syntax                    (Expr)

renderBulkCommitGroup :: Int -> String
renderBulkCommitGroup indentation =
    asmLine indentation "cp.async.bulk.commit_group;" ["memory"]

renderBulkWaitGroup :: Int -> BulkWaitMode -> Int -> String
renderBulkWaitGroup indentation mode groupCount =
    asmLine
        indentation
        ( "cp.async.bulk.wait_group"
            ++ bulkWaitModeTag mode
            ++ " "
            ++ show groupCount
            ++ ";"
        )
        ["memory"]

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
    renderAsm indentation instruction [] inputOperands ["memory"]
  where
    (dimension, coordinateExpressions) = tmaCoordinateInfo coordinates
    (destinationTag, destinationOperand, barrierOperand, multicastMask) =
        case destination of
            TmaCtaShared destinationAddress ->
                ( ".shared::cta"
                , localSharedOperand destinationAddress
                , localSharedOperand barrier
                , Nothing
                )
            TmaClusterShared width destinationAddress ->
                ( ".shared::cluster"
                , clusterAddressOperand width destinationAddress
                , clusterAddressOperand width barrier
                , Nothing
                )
            TmaMulticastClusterShared destinationAddress ctaMask ->
                ( ".shared::cluster"
                , localSharedOperand destinationAddress
                , localSharedOperand barrier
                , Just ctaMask
                )
    coordinateOperands = fmap (exprOperand "r") coordinateExpressions
    barrierIndex = 2 + length coordinateOperands
    maskIndex = barrierIndex + 1
    cacheIndex = maskIndex + maybe 0 (const 1) multicastMask
    inputOperands =
        [destinationOperand, exprOperand "l" tensorMap]
            ++ coordinateOperands
            ++ [barrierOperand]
            ++ maybe [] (pure . exprOperand "r") multicastMask
            ++ maybe [] (\(TmaCachePolicy policy) -> [exprOperand "l" policy]) cachePolicy
    tensorInstruction =
        "cp.async.bulk.tensor."
            ++ show dimension
            ++ "d"
            ++ destinationTag
            ++ ".global.mbarrier::complete_tx::bytes"
            ++ maybe "" (const ".multicast::cluster") multicastMask
            ++ maybe "" (const ".L2::cache_hint") cachePolicy
            ++ " [%0], [%1, {"
            ++ placeholders 2 (length coordinateOperands)
            ++ "}], [%"
            ++ show barrierIndex
            ++ "]"
            ++ maybe "" (const ", cta_mask") multicastMask
            ++ maybe "" (const (", %" ++ show cacheIndex)) cachePolicy
            ++ ";"
    instruction =
        case multicastMask of
            Nothing -> tensorInstruction
            Just _ ->
                "{ .reg .b16 cta_mask; .reg .b16 unused; mov.b32 {cta_mask, unused}, %"
                    ++ show maskIndex
                    ++ "; "
                    ++ tensorInstruction
                    ++ " }"

renderTmaTensorStore :: Int -> TmaTensorMap -> TmaCoordinates -> Expr -> Maybe TmaCachePolicy -> String
renderTmaTensorStore indentation (TmaTensorMap tensorMap) coordinates source cachePolicy =
    renderAsm indentation instruction [] inputOperands ["memory"]
  where
    (dimension, coordinateExpressions) = tmaCoordinateInfo coordinates
    coordinateOperands = fmap (exprOperand "r") coordinateExpressions
    sourceIndex = 1 + length coordinateOperands
    cacheIndex = sourceIndex + 1
    inputOperands =
        [exprOperand "l" tensorMap]
            ++ coordinateOperands
            ++ [localSharedOperand source]
            ++ maybe [] (\(TmaCachePolicy policy) -> [exprOperand "l" policy]) cachePolicy
    instruction =
        "cp.async.bulk.tensor."
            ++ show dimension
            ++ "d.global.shared::cta.bulk_group"
            ++ maybe "" (const ".L2::cache_hint") cachePolicy
            ++ " [%0, {"
            ++ placeholders 1 (length coordinateOperands)
            ++ "}], [%"
            ++ show sourceIndex
            ++ "]"
            ++ maybe "" (const (", %" ++ show cacheIndex)) cachePolicy
            ++ ";"

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

bulkWaitModeTag :: BulkWaitMode -> String
bulkWaitModeTag BulkWaitComplete = ""
bulkWaitModeTag BulkWaitRead     = ".read"
