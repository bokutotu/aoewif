module Aoewif.Target.Cuda.DSL.LoadStore (
    MemorySpace (..),
    VectorWidth (..),
    MemoryType (..),
    vectorLoad,
    vectorStore,
)
where

import           Aoewif.Target.Cuda.AsmOperand  (AsmInput, AsmOutput)
import           Aoewif.Target.Cuda.DSL.Builder (Block, expr_)
import           Aoewif.Target.Cuda.Expr        (Expr (..))
import           Aoewif.Target.Cuda.LoadStore   (MemorySpace (..),
                                                 MemoryType (..),
                                                 VectorLoad (VectorLoad),
                                                 VectorStore (VectorStore),
                                                 VectorWidth (..))

vectorLoad :: MemorySpace -> VectorWidth -> MemoryType -> [AsmOutput] -> AsmInput -> Block ()
vectorLoad space width ty outputs address =
    expr_ (VectorLoadExpr (VectorLoad space width ty outputs address))

vectorStore :: MemorySpace -> VectorWidth -> MemoryType -> AsmInput -> [AsmInput] -> Block ()
vectorStore space width ty address values =
    expr_ (VectorStoreExpr (VectorStore space width ty address values))
