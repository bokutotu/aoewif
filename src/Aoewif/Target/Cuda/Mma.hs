{-# LANGUAGE QuasiQuotes #-}

module Aoewif.Target.Cuda.Mma (
    Mma (..),
    MmaOperation (..),
    Operand (..),
)
where

import           Aoewif.Target.Cuda.AsmOperand (AsmInput, AsmOutput)
import           Aoewif.Target.Cuda.Render     (Render (..))
import           Aoewif.Target.Cuda.Type       (Type (..))
import           Aoewif.Target.Template        (template)
import           Data.List                     (intercalate)

newtype Mma = M16N8K16 {operation :: MmaOperation}
    deriving stock (Eq, Show)

data MmaOperation
    = DABC {dest :: Operand AsmOutput, inputA :: Operand AsmInput, inputB :: Operand AsmInput, inputC :: Operand AsmInput}
    | CABC {accumulator :: Operand AsmOutput, inputA :: Operand AsmInput, inputB :: Operand AsmInput}
    deriving stock (Eq, Show)

data Operand parameter = Operand {elementType :: Type, parameters :: [parameter]}
    deriving stock (Eq, Show)

data OperandReferences = OperandReferences
    { registersD :: (Int, Int)
    , registersA :: (Int, Int)
    , registersB :: (Int, Int)
    , registersC :: (Int, Int)
    }

instance Render Mma where
    render (M16N8K16 op) = case op of
        DABC d a b c ->
            let startA = count d
                startB = startA + count a
                startC = startB + count b
             in renderOperation d a b c [a, b, c] $
                    OperandReferences
                        { registersD = (0, startA - 1)
                        , registersA = (startA, startB - 1)
                        , registersB = (startB, startC - 1)
                        , registersC = (startC, startC + count c - 1)
                        }
        CABC c a b ->
            let startA = count c
                startB = startA + count a
                rangeC = (0, startA - 1)
             in renderOperation c a b c [a, b] $
                    OperandReferences
                        { registersD = rangeC
                        , registersA = (startA, startB - 1)
                        , registersB = (startB, startB + count b - 1)
                        , registersC = rangeC
                        }
      where
        count :: Operand parameter -> Int
        count = length . parameters

        renderOperation :: Operand AsmOutput -> Operand AsmInput -> Operand AsmInput -> Operand c -> [Operand AsmInput] -> OperandReferences -> String
        renderOperation d a b c inputs refs =
            [template|
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.${typeD}.${typeA}.${typeB}.${typeC} {${refsD}}, {${refsA}}, {${refsB}}, {${refsC}};\n"
                : ${outputsText}
                : ${inputsText}
            )
            |]
          where
            typeD = renderPtxType (elementType d)
            typeA = renderPtxType (elementType a)
            typeB = renderPtxType (elementType b)
            typeC = renderPtxType (elementType c)
            refsD = uncurry renderRegisters (registersD refs)
            refsA = uncurry renderRegisters (registersA refs)
            refsB = uncurry renderRegisters (registersB refs)
            refsC = uncurry renderRegisters (registersC refs)
            outputsText = intercalate ", " (map render (parameters d))
            inputsText = intercalate ", " (map render (concatMap parameters inputs))

renderRegisters :: Int -> Int -> String
renderRegisters firstIndex lastIndex =
    intercalate ", " ["%" ++ show i | i <- [firstIndex .. lastIndex]]

renderPtxType :: Type -> String
renderPtxType Bool                  = "pred"
renderPtxType U32                   = "u32"
renderPtxType USize                 = "u64"
renderPtxType F16                   = "f16"
renderPtxType BF16                  = "bf16"
renderPtxType F32                   = "f32"
renderPtxType (Const valueType)     = renderPtxType valueType ++ " const"
renderPtxType (Pointer pointeeType) = renderPtxType pointeeType ++ "*"
