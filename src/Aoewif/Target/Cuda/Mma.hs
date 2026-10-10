module Aoewif.Target.Cuda.Mma (
    Mma (..),
    MmaOperation (..),
    Operand (..),
)
where

import           Aoewif.Target.Cuda.AsmOperand (AsmInput, AsmOutput)
import           Aoewif.Target.Cuda.Render     (Render (..), indent)
import           Aoewif.Target.Cuda.Type       (Type (..))
import           Data.List                     (intercalate)
import           Text.Printf                   (printf)

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
    render mma =
        "asm volatile(\n" ++ indent (show (renderInstruction mma ++ ";\n") ++ "\n" ++ renderBindings (operation mma) ++ "\n") ++ ")"

renderInstruction :: Mma -> String
renderInstruction (M16N8K16 op) = case op of
    DABC d a b c ->
        let startA = count d
            startB = startA + count a
            startC = startB + count b
         in instruction d a b c $
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
         in instruction c a b c $
                OperandReferences
                    { registersD = rangeC
                    , registersA = (startA, startB - 1)
                    , registersB = (startB, startB + count b - 1)
                    , registersC = rangeC
                    }
  where
    count :: Operand parameter -> Int
    count = length . parameters

    instruction :: Operand d -> Operand a -> Operand b -> Operand c -> OperandReferences -> String
    instruction d a b c refs =
        printf
            "mma.sync.aligned.m16n8k16.row.col.%s.%s.%s.%s {%s}, {%s}, {%s}, {%s}"
            (renderPtxType (elementType d))
            (renderPtxType (elementType a))
            (renderPtxType (elementType b))
            (renderPtxType (elementType c))
            (uncurry renderRegisters (registersD refs))
            (uncurry renderRegisters (registersA refs))
            (uncurry renderRegisters (registersB refs))
            (uncurry renderRegisters (registersC refs))

renderRegisters :: Int -> Int -> String
renderRegisters firstIndex lastIndex =
    intercalate ", " ["%" ++ show i | i <- [firstIndex .. lastIndex]]

renderBindings :: MmaOperation -> String
renderBindings op = case op of
    DABC d a b c -> bindings d [a, b, c]
    CABC c a b   -> bindings c [a, b]
  where
    bindings :: Operand AsmOutput -> [Operand AsmInput] -> String
    bindings output inputs =
        ": "
            ++ intercalate ", " (map render (parameters output))
            ++ "\n: "
            ++ intercalate ", " (map render (concatMap parameters inputs))

renderPtxType :: Type -> String
renderPtxType Bool                  = "pred"
renderPtxType U32                   = "u32"
renderPtxType USize                 = "u64"
renderPtxType F16                   = "f16"
renderPtxType BF16                  = "bf16"
renderPtxType F32                   = "f32"
renderPtxType (Const valueType)     = renderPtxType valueType ++ " const"
renderPtxType (Pointer pointeeType) = renderPtxType pointeeType ++ "*"
