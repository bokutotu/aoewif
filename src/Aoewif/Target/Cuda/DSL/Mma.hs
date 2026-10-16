module Aoewif.Target.Cuda.DSL.Mma (
    Mma,
    MmaOperation,
    Operand,
    accumulateInto,
    f16,
    f32,
    m16n8k16,
    mma,
    multiplyAddInto,
)
where

import           Aoewif.Target.Cuda.AsmOperand  (AsmInput, AsmOutput)
import           Aoewif.Target.Cuda.DSL.Builder (Block, expr_)
import           Aoewif.Target.Cuda.Expr        (Expr (..))
import           Aoewif.Target.Cuda.Mma         (Mma (..), MmaOperation (..),
                                                 Operand (..))
import           Aoewif.Target.Cuda.Type        (Type (..))

mma :: (MmaOperation -> Mma) -> MmaOperation -> Block ()
mma shape = expr_ . MmaExpr . shape

m16n8k16 :: MmaOperation -> Mma
m16n8k16 = M16N8K16

accumulateInto :: Operand AsmOutput -> Operand AsmInput -> Operand AsmInput -> MmaOperation
accumulateInto = CABC

multiplyAddInto :: Operand AsmOutput -> Operand AsmInput -> Operand AsmInput -> Operand AsmInput -> MmaOperation
multiplyAddInto = DABC

f16 :: [parameter] -> Operand parameter
f16 = Operand F16

f32 :: [parameter] -> Operand parameter
f32 = Operand F32
