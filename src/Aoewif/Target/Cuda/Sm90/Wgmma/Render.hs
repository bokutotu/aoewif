{-# LANGUAGE QuasiQuotes #-}

module Aoewif.Target.Cuda.Sm90.Wgmma.Render (
    renderWgmmaMmaAsync,
    renderWgmmaFence,
    renderWgmmaCommitGroup,
    renderWgmmaWaitGroup,
) where

import           Aoewif.Target.Cuda.Codegen                (indent, renderExpr)
import           Aoewif.Target.Cuda.Sm90.Asm               (constraints,
                                                            placeholders)
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
import           Data.String.Interpolate                   (i)

renderWgmmaCommitGroup :: Int -> String
renderWgmmaCommitGroup indentation =
    [i|#{padding}asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
|]
  where
    padding = indent indentation

renderWgmmaWaitGroup :: Int -> Int -> String
renderWgmmaWaitGroup indentation groupCount =
    [i|#{padding}asm volatile("wgmma.wait_group.sync.aligned #{groupCount};" ::: "memory");
|]
  where
    padding = indent indentation

renderWgmmaFence :: Int -> WgmmaMma -> String
renderWgmmaFence indentation operation
    | null fenceOperands =
        [i|#{padding}asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
|]
    | otherwise =
        [i|#{padding}asm volatile("wgmma.fence.sync.aligned;"
#{operandPadding}: #{fenceOperands}
#{operandPadding}:
#{operandPadding}: "memory"
#{padding});
|]
  where
    (outputConstraint, accumulator, operands) =
        case operation of
            WgmmaF16 _ accumulatorType fragment halfOperands _ _ _ ->
                (accumulatorConstraint accumulatorType, fragment, halfOperands)
            WgmmaBF16 _ fragment halfOperands _ _ _ ->
                ("+f", fragment, halfOperands)
    accumulatorOperands = constraints outputConstraint (wgmmaFragmentRegisters accumulator)
    registerAOperands =
        case operands of
            WgmmaHalfSharedOperands{} -> ""
            WgmmaHalfRegisterOperands fragmentA _ _ ->
                constraints "+r" (wgmmaFragmentRegisters fragmentA)
    fenceOperands = intercalate ", " (filter (not . null) [accumulatorOperands, registerAOperands])
    padding = indent indentation
    operandPadding = indent (indentation + 1)

renderWgmmaMmaAsync :: Int -> WgmmaMma -> String
renderWgmmaMmaAsync indentation operation =
    case operation of
        WgmmaF16 shapeN accumulatorType accumulator operands scaleD scaleA scaleB ->
            renderWgmma
                indentation
                [i|m64n#{wgmmaFloatNTag shapeN}k16.#{accumulatorTypeTag accumulatorType}.f16.f16|]
                (accumulatorConstraint accumulatorType)
                accumulator
                operands
                scaleD
                scaleA
                scaleB
        WgmmaBF16 shapeN accumulator operands scaleD scaleA scaleB ->
            renderWgmma
                indentation
                [i|m64n#{wgmmaFloatNTag shapeN}k16.f32.bf16.bf16|]
                "+f"
                accumulator
                operands
                scaleD
                scaleA
                scaleB

renderWgmma :: Int -> String -> String -> WgmmaFragment -> WgmmaHalfOperands -> Expr -> WgmmaScale -> WgmmaScale -> String
renderWgmma indentation instructionTag outputConstraint accumulator operands scaleD scaleA scaleB =
    case operands of
        WgmmaHalfSharedOperands
            (WgmmaDescriptor descriptorA)
            (WgmmaDescriptor descriptorB)
            transposeA
            transposeB
                | null outputRegisters ->
                    [i|#{padding}asm volatile("{ .reg .pred p; setp.ne.b32 p, %#{scaleDIndex}, 0; wgmma.mma_async.sync.aligned.#{instructionTag} {#{dOperands}}, %#{descriptorAIndex}, %#{descriptorBIndex}, p, #{scaleATag}, #{scaleBTag}, #{wgmmaTransposeTag transposeA}, #{wgmmaTransposeTag transposeB}; }"
#{operandPadding}:: "l"(#{renderExpr descriptorA}), "l"(#{renderExpr descriptorB}), "r"(#{renderExpr scaleD})
#{padding});
|]
                | otherwise ->
                    [i|#{padding}asm volatile("{ .reg .pred p; setp.ne.b32 p, %#{scaleDIndex}, 0; wgmma.mma_async.sync.aligned.#{instructionTag} {#{dOperands}}, %#{descriptorAIndex}, %#{descriptorBIndex}, p, #{scaleATag}, #{scaleBTag}, #{wgmmaTransposeTag transposeA}, #{wgmmaTransposeTag transposeB}; }"
#{operandPadding}: #{constraints outputConstraint outputRegisters}
#{operandPadding}: "l"(#{renderExpr descriptorA}), "l"(#{renderExpr descriptorB}), "r"(#{renderExpr scaleD})
#{padding});
|]
              where
                descriptorAIndex = outputCount
                descriptorBIndex = outputCount + 1
                scaleDIndex = outputCount + 2
        WgmmaHalfRegisterOperands
            (WgmmaFragment registersA)
            (WgmmaDescriptor descriptorB)
            transposeB
                | null outputRegisters ->
                    [i|#{padding}asm volatile("{ .reg .pred p; setp.ne.b32 p, %#{scaleDIndex}, 0; wgmma.mma_async.sync.aligned.#{instructionTag} {#{dOperands}}, {#{aOperands}}, %#{descriptorBIndex}, p, #{scaleATag}, #{scaleBTag}, #{wgmmaTransposeTag transposeB}; }"
#{operandPadding}:: #{constraints "r" registersA}#{registerASeparator}"l"(#{renderExpr descriptorB}), "r"(#{renderExpr scaleD})
#{padding});
|]
                | otherwise ->
                    [i|#{padding}asm volatile("{ .reg .pred p; setp.ne.b32 p, %#{scaleDIndex}, 0; wgmma.mma_async.sync.aligned.#{instructionTag} {#{dOperands}}, {#{aOperands}}, %#{descriptorBIndex}, p, #{scaleATag}, #{scaleBTag}, #{wgmmaTransposeTag transposeB}; }"
#{operandPadding}: #{constraints outputConstraint outputRegisters}
#{operandPadding}: #{constraints "r" registersA}#{registerASeparator}"l"(#{renderExpr descriptorB}), "r"(#{renderExpr scaleD})
#{padding});
|]
              where
                aCount = length registersA
                aOperands = placeholders outputCount aCount
                descriptorBIndex = outputCount + aCount
                scaleDIndex = descriptorBIndex + 1
                registerASeparator = if null registersA then "" else ", "
  where
    outputRegisters = wgmmaFragmentRegisters accumulator
    outputCount = length outputRegisters
    dOperands = placeholders 0 outputCount
    scaleATag = wgmmaScaleTag scaleA
    scaleBTag = wgmmaScaleTag scaleB
    padding = indent indentation
    operandPadding = indent (indentation + 1)

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
