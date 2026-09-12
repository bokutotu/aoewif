module Aoewif.Target.Cuda.Sm90.Wgmma.Render (
    renderWgmmaMmaAsync,
    renderWgmmaFence,
    renderWgmmaCommitGroup,
    renderWgmmaWaitGroup,
) where

import           Aoewif.Target.Cuda.Sm90.Asm               (AsmOperand, asmLine,
                                                            exprOperand,
                                                            placeholder,
                                                            registerVector,
                                                            renderAsm)
import           Aoewif.Target.Cuda.Sm90.Wgmma.Instruction (WgmmaAccumulatorType (..),
                                                            WgmmaDescriptor (..),
                                                            WgmmaFloatN (..),
                                                            WgmmaFragment (..),
                                                            WgmmaHalfOperands (..),
                                                            WgmmaMma (..),
                                                            WgmmaScale (..),
                                                            WgmmaTranspose (..))
import           Aoewif.Target.Cuda.Syntax                 (Expr)
import           Data.List                                 (intercalate)

renderWgmmaCommitGroup :: Int -> String
renderWgmmaCommitGroup indentation =
    asmLine indentation "wgmma.commit_group.sync.aligned;" ["memory"]

renderWgmmaWaitGroup :: Int -> Int -> String
renderWgmmaWaitGroup indentation groupCount =
    asmLine
        indentation
        ( "wgmma.wait_group.sync.aligned "
            ++ show groupCount
            ++ ";"
        )
        ["memory"]

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
