{-# OPTIONS_GHC -Wno-orphans #-}

module Aoewif.Target.Cuda.Sm90.Render () where

import           Aoewif.Target.Cuda.Codegen          (indent, renderExpr)
import           Aoewif.Target.Cuda.Sm90.Instruction
import           Aoewif.Target.Cuda.Syntax           (Expr)
import           Aoewif.Target.Cuda.TensorCoreOp     (RenderOp (..))
import           Data.List                           (intercalate)

instance RenderOp Sm90Op where
    renderOp indentation operation =
        case operation of
            WgmmaMmaAsync mmaOperation ->
                renderWgmmaMmaAsync indentation mmaOperation
            WgmmaFence mmaOperation ->
                renderWgmmaFence indentation mmaOperation
            WgmmaCommitGroup ->
                asmLine indentation "wgmma.commit_group.sync.aligned;" ["memory"]
            WgmmaWaitGroup groupCount ->
                asmLine
                    indentation
                    ( "wgmma.wait_group.sync.aligned "
                        ++ show groupCount
                        ++ ";"
                    )
                    ["memory"]
            TmaTensor tensorOperation ->
                renderTmaTensor indentation tensorOperation
            BulkCommitGroup ->
                asmLine indentation "cp.async.bulk.commit_group;" ["memory"]
            BulkWaitGroup mode groupCount ->
                asmLine
                    indentation
                    ( "cp.async.bulk.wait_group"
                        ++ bulkWaitModeTag mode
                        ++ " "
                        ++ show groupCount
                        ++ ";"
                    )
                    ["memory"]
            MBarrierInstruction barrierOperation ->
                renderMBarrier indentation barrierOperation
            FenceProxyAsync scope ->
                asmLine
                    indentation
                    ("fence.proxy.async" ++ sharedScopeTag scope ++ ";")
                    ["memory"]
            FenceMBarrierInit ->
                asmLine
                    indentation
                    "fence.mbarrier_init.release.cluster;"
                    ["memory"]
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
                asmLine
                    indentation
                    ( "barrier.cluster.arrive."
                        ++ clusterBarrierArrivalTag arrival
                        ++ ".aligned;"
                    )
                    ["memory"]
            ClusterBarrierWait ->
                asmLine
                    indentation
                    "barrier.cluster.wait.acquire.aligned;"
                    ["memory"]
            ReadClusterSpecialRegister destination specialRegister ->
                renderClusterSpecialRegister indentation destination specialRegister
            MapSharedCluster width destination source ctaRank ->
                renderMapSharedCluster indentation width destination source ctaRank
            GetCtaRank width destination address ->
                renderGetCtaRank indentation width destination address

renderWgmmaFence :: Int -> WgmmaMma -> String
renderWgmmaFence indentation operation =
    renderAsm
        indentation
        "wgmma.fence.sync.aligned;"
        (wgmmaFenceOperands operation)
        []
        ["memory"]

wgmmaFenceOperands :: WgmmaMma -> [AsmOperand]
wgmmaFenceOperands operation =
    case operation of
        WgmmaF16 _ accumulatorType accumulator operands _ _ _ ->
            fenceFragmentOperands (accumulatorConstraint accumulatorType) accumulator ++ wgmmaHalfRegisterFenceOperands operands
        WgmmaBF16 _ accumulator operands _ _ _ -> fenceFragmentOperands "+f" accumulator ++ wgmmaHalfRegisterFenceOperands operands

fenceFragmentOperands :: String -> WgmmaFragment -> [AsmOperand]
fenceFragmentOperands constraint = fmap (exprOperand constraint) . wgmmaFragmentRegisters

wgmmaHalfRegisterFenceOperands :: WgmmaHalfOperands -> [AsmOperand]
wgmmaHalfRegisterFenceOperands operands =
    case operands of
        WgmmaHalfSharedOperands{} -> []
        WgmmaHalfRegisterOperands fragmentA _ _ -> fenceFragmentOperands "+r" fragmentA

renderWgmmaMmaAsync :: Int -> WgmmaMma -> String
renderWgmmaMmaAsync indentation operation =
    case operation of
        WgmmaF16
            shapeN
            accumulatorType
            accumulator
            operands
            scaleD
            scaleA
            scaleB ->
                renderWgmma
                    indentation
                    ( "m64n"
                        ++ wgmmaFloatNTag shapeN
                        ++ "k16."
                        ++ accumulatorTypeTag accumulatorType
                        ++ ".f16.f16"
                    )
                    (accumulatorConstraint accumulatorType)
                    accumulator
                    (wgmmaHalfOperandInfo operands)
                    scaleD
                    [wgmmaScaleTag scaleA, wgmmaScaleTag scaleB]
        WgmmaBF16 shapeN accumulator operands scaleD scaleA scaleB ->
            renderWgmma
                indentation
                ( "m64n"
                    ++ wgmmaFloatNTag shapeN
                    ++ "k16.f32.bf16.bf16"
                )
                "+f"
                accumulator
                (wgmmaHalfOperandInfo operands)
                scaleD
                [wgmmaScaleTag scaleA, wgmmaScaleTag scaleB]

data WgmmaOperandInfo = WgmmaOperandInfo
    { wgmmaOperandText       :: Int -> String
    , wgmmaOperandInputs     :: [AsmOperand]
    , wgmmaOperandImmediates :: [String]
    }

renderWgmma :: Int -> String -> String -> WgmmaFragment -> WgmmaOperandInfo -> Expr -> [String] -> String
renderWgmma indentation instructionTag outputConstraint accumulator operandInfo scaleD scaleImmediates =
    renderAsm indentation instruction outputOperands inputOperands []
  where
    outputRegisters = wgmmaFragmentRegisters accumulator
    outputCount = length outputRegisters
    operandInputs = wgmmaOperandInputs operandInfo
    scaleDIndex = outputCount + length operandInputs
    outputOperands = fmap (exprOperand outputConstraint) outputRegisters
    inputOperands = operandInputs ++ [exprOperand "r" scaleD]
    immediateOperands = scaleImmediates ++ wgmmaOperandImmediates operandInfo
    instruction =
        "{ .reg .pred p; setp.ne.b32 p, %"
            ++ show scaleDIndex
            ++ ", 0; wgmma.mma_async.sync.aligned."
            ++ instructionTag
            ++ " "
            ++ registerVector 0 outputCount
            ++ ", "
            ++ wgmmaOperandText operandInfo outputCount
            ++ ", p, "
            ++ intercalate ", " immediateOperands
            ++ "; }"

wgmmaHalfOperandInfo :: WgmmaHalfOperands -> WgmmaOperandInfo
wgmmaHalfOperandInfo operands =
    case operands of
        WgmmaHalfSharedOperands
            (WgmmaDescriptor descriptorA)
            (WgmmaDescriptor descriptorB)
            transposeA
            transposeB ->
                WgmmaOperandInfo
                    { wgmmaOperandText = \firstIndex ->
                        placeholder firstIndex
                            ++ ", "
                            ++ placeholder (firstIndex + 1)
                    , wgmmaOperandInputs =
                        [exprOperand "l" descriptorA, exprOperand "l" descriptorB]
                    , wgmmaOperandImmediates =
                        [ wgmmaTransposeTag transposeA
                        , wgmmaTransposeTag transposeB
                        ]
                    }
        WgmmaHalfRegisterOperands
            fragmentA
            (WgmmaDescriptor descriptorB)
            transposeB ->
                WgmmaOperandInfo
                    { wgmmaOperandText = \firstIndex ->
                        registerVector firstIndex (length registersA)
                            ++ ", "
                            ++ placeholder (firstIndex + length registersA)
                    , wgmmaOperandInputs =
                        fmap (exprOperand "r") registersA
                            ++ [exprOperand "l" descriptorB]
                    , wgmmaOperandImmediates = [wgmmaTransposeTag transposeB]
                    }
              where
                registersA = wgmmaFragmentRegisters fragmentA

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

data AsmOperand = AsmOperand String String

exprOperand :: String -> Expr -> AsmOperand
exprOperand constraint =
    AsmOperand constraint . renderExpr

localSharedOperand :: Expr -> AsmOperand
localSharedOperand =
    AsmOperand "l" . sharedAddress

clusterAddressOperand :: ClusterAddressWidth -> Expr -> AsmOperand
clusterAddressOperand width =
    exprOperand inputConstraint
  where
    (_, _, inputConstraint) = clusterAddressWidthInfo width

renderAsm :: Int -> String -> [AsmOperand] -> [AsmOperand] -> [String] -> String
renderAsm indentation instruction outputs inputs clobbers
    | null outputs && null inputs =
        asmLine indentation instruction clobbers
    | null outputs =
        unlines
            ( [ asmOpen indentation instruction
              , indent (indentation + 1) ++ ":: " ++ renderAsmOperands inputs
              ]
                ++ renderClobberLine indentation clobbers
                ++ [indent indentation ++ ");"]
            )
    | otherwise =
        unlines
            ( [ asmOpen indentation instruction
              , indent (indentation + 1) ++ ": " ++ renderAsmOperands outputs
              ]
                ++ renderInputLine indentation inputs clobbers
                ++ renderClobberLine indentation clobbers
                ++ [indent indentation ++ ");"]
            )

renderInputLine :: Int -> [AsmOperand] -> [String] -> [String]
renderInputLine indentation inputs clobbers
    | null inputs && null clobbers = []
    | null inputs = [indent (indentation + 1) ++ ":"]
    | otherwise =
        [indent (indentation + 1) ++ ": " ++ renderAsmOperands inputs]

renderClobberLine :: Int -> [String] -> [String]
renderClobberLine _ [] = []
renderClobberLine indentation clobbers =
    [indent (indentation + 1) ++ ": " ++ renderClobbers clobbers]

asmOpen :: Int -> String -> String
asmOpen indentation instruction =
    indent indentation ++ "asm volatile(\"" ++ instruction ++ "\""

asmLine :: Int -> String -> [String] -> String
asmLine indentation instruction clobbers =
    indent indentation
        ++ "asm volatile(\""
        ++ instruction
        ++ "\""
        ++ if null clobbers
            then ");\n"
            else " ::: " ++ renderClobbers clobbers ++ ");\n"

renderAsmOperands :: [AsmOperand] -> String
renderAsmOperands = intercalate ", " . fmap renderAsmOperand

renderAsmOperand :: AsmOperand -> String
renderAsmOperand (AsmOperand constraint value) = "\"" ++ constraint ++ "\"(" ++ value ++ ")"

renderClobbers :: [String] -> String
renderClobbers = intercalate ", " . fmap (\clobber -> "\"" ++ clobber ++ "\"")

placeholder :: Int -> String
placeholder index = "%" ++ show index

placeholders :: Int -> Int -> String
placeholders first count = intercalate ", " (fmap placeholder [first .. first + count - 1])

registerVector :: Int -> Int -> String
registerVector first count = "{" ++ placeholders first count ++ "}"

sharedAddress :: Expr -> String
sharedAddress address = "__cvta_generic_to_shared(&" ++ renderExpr address ++ ")"

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

accumulatorTypeTag :: WgmmaAccumulatorType -> String
accumulatorTypeTag WgmmaAccumulatorF16 = "f16"
accumulatorTypeTag WgmmaAccumulatorF32 = "f32"

accumulatorConstraint :: WgmmaAccumulatorType -> String
accumulatorConstraint WgmmaAccumulatorF16 = "+r"
accumulatorConstraint WgmmaAccumulatorF32 = "+f"

wgmmaScaleTag :: WgmmaScale -> String
wgmmaScaleTag WgmmaScaleOne         = "1"
wgmmaScaleTag WgmmaScaleNegativeOne = "-1"

wgmmaTransposeTag :: WgmmaTranspose -> String
wgmmaTransposeTag WgmmaNotTransposed = "0"
wgmmaTransposeTag WgmmaTransposed    = "1"

wgmmaFloatNTag :: WgmmaFloatN -> String
wgmmaFloatNTag WgmmaFloatN8   = "8"
wgmmaFloatNTag WgmmaFloatN16  = "16"
wgmmaFloatNTag WgmmaFloatN24  = "24"
wgmmaFloatNTag WgmmaFloatN32  = "32"
wgmmaFloatNTag WgmmaFloatN40  = "40"
wgmmaFloatNTag WgmmaFloatN48  = "48"
wgmmaFloatNTag WgmmaFloatN56  = "56"
wgmmaFloatNTag WgmmaFloatN64  = "64"
wgmmaFloatNTag WgmmaFloatN72  = "72"
wgmmaFloatNTag WgmmaFloatN80  = "80"
wgmmaFloatNTag WgmmaFloatN88  = "88"
wgmmaFloatNTag WgmmaFloatN96  = "96"
wgmmaFloatNTag WgmmaFloatN104 = "104"
wgmmaFloatNTag WgmmaFloatN112 = "112"
wgmmaFloatNTag WgmmaFloatN120 = "120"
wgmmaFloatNTag WgmmaFloatN128 = "128"
wgmmaFloatNTag WgmmaFloatN136 = "136"
wgmmaFloatNTag WgmmaFloatN144 = "144"
wgmmaFloatNTag WgmmaFloatN152 = "152"
wgmmaFloatNTag WgmmaFloatN160 = "160"
wgmmaFloatNTag WgmmaFloatN168 = "168"
wgmmaFloatNTag WgmmaFloatN176 = "176"
wgmmaFloatNTag WgmmaFloatN184 = "184"
wgmmaFloatNTag WgmmaFloatN192 = "192"
wgmmaFloatNTag WgmmaFloatN200 = "200"
wgmmaFloatNTag WgmmaFloatN208 = "208"
wgmmaFloatNTag WgmmaFloatN216 = "216"
wgmmaFloatNTag WgmmaFloatN224 = "224"
wgmmaFloatNTag WgmmaFloatN232 = "232"
wgmmaFloatNTag WgmmaFloatN240 = "240"
wgmmaFloatNTag WgmmaFloatN248 = "248"
wgmmaFloatNTag WgmmaFloatN256 = "256"

bulkWaitModeTag :: BulkWaitMode -> String
bulkWaitModeTag BulkWaitComplete = ""
bulkWaitModeTag BulkWaitRead     = ".read"

sharedScopeTag :: SharedScope -> String
sharedScopeTag SharedCta     = ".shared::cta"
sharedScopeTag SharedCluster = ".shared::cluster"

registerAdjustmentTag :: RegisterAdjustment -> String
registerAdjustmentTag IncreaseRegisters = "inc"
registerAdjustmentTag DecreaseRegisters = "dec"

clusterBarrierArrivalTag :: ClusterBarrierArrival -> String
clusterBarrierArrivalTag ClusterBarrierRelease = "release"
clusterBarrierArrivalTag ClusterBarrierRelaxed = "relaxed"

clusterDimensionTag :: ClusterDimension -> String
clusterDimensionTag ClusterX = "x"
clusterDimensionTag ClusterY = "y"
clusterDimensionTag ClusterZ = "z"

clusterAddressWidthInfo :: ClusterAddressWidth -> (String, String, String)
clusterAddressWidthInfo ClusterAddressU32 = ("u32", "=r", "r")
clusterAddressWidthInfo ClusterAddressU64 = ("u64", "=l", "l")
