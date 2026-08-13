module Aoewif.Internal.Kernel.Operation (
    ValueId (..),
    Buffer (..),
    ScalarOperation (..),
    Value (..),
    KernelStatement (..),
    KernelBlock (..),
) where

import           Aoewif.Internal.IR        (IndexExpr)
import           Aoewif.Internal.Primitive (ComparePredicate, ValueType)

newtype ValueId = ValueId Int
    deriving stock (Eq, Ord, Show)

data Buffer
    = InputBuffer Int
    | OutputBuffer Int
    deriving stock (Eq, Ord, Show)

data ScalarOperation
    = DataLiteralOperation Float
    | PredicateLiteralOperation Bool
    | IndexOperation IndexExpr
    | LoadOperation Buffer IndexExpr
    | AddOperation ValueId ValueId
    | SubOperation ValueId ValueId
    | MulOperation ValueId ValueId
    | DivOperation ValueId ValueId
    | FmaOperation ValueId ValueId ValueId
    | MinOperation ValueId ValueId
    | MaxOperation ValueId ValueId
    | ExpOperation ValueId
    | LogOperation ValueId
    | CompareOperation ComparePredicate ValueId ValueId
    | SelectOperation ValueId ValueId ValueId
    deriving stock (Eq, Show)

data Value = Value
    { valueId        :: ValueId
    , valueType      :: ValueType
    , valueOperation :: ScalarOperation
    }
    deriving stock (Eq, Show)

data KernelStatement
    = DefineValue Value
    | StoreBuffer Buffer IndexExpr ValueId
    deriving stock (Eq, Show)

newtype KernelBlock = KernelBlock
    { kernelStatements :: [KernelStatement]
    }
    deriving stock (Eq, Show)
