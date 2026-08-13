module Aoewif.IR (
    Name (..),
    SymbolId (..),
    TensorId (..),
    IndexId (..),
    LoopId (..),
    BlockId (..),
    DimExpr (..),
    Symbol (..),
    DataType (..),
    ComparePredicate (..),
    TensorKind (..),
    TensorDecl (..),
    IndexExpr (..),
    Predicate (..),
    IndexBinding (..),
    Loop (..),
    Block (..),
    Statement (..),
    LoopIR (..),
    IR (..),
) where

import           Aoewif.Internal.IR
import           Aoewif.Internal.Primitive (ComparePredicate (..),
                                            DataType (..))
