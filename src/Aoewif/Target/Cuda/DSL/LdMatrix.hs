module Aoewif.Target.Cuda.DSL.LdMatrix (
    LdMatrix,
    LdMatrixSpace (..),
    MatrixCount (..),
    MatrixLayout (..),
    ldmatrix,
    m8n8b16,
)
where

import           Aoewif.Target.Cuda.AsmOperand  (AsmInput, AsmOutput)
import           Aoewif.Target.Cuda.DSL.Builder (Block, expr_)
import           Aoewif.Target.Cuda.Expr        (Expr (..))
import           Aoewif.Target.Cuda.LdMatrix    (LdMatrix (..),
                                                 LdMatrixSpace (..),
                                                 MatrixCount (..),
                                                 MatrixLayout (..))

m8n8b16 :: MatrixCount -> MatrixLayout -> LdMatrixSpace -> [AsmOutput] -> AsmInput -> LdMatrix
m8n8b16 = M8N8B16

ldmatrix :: LdMatrix -> Block ()
ldmatrix = expr_ . LdMatrixExpr
