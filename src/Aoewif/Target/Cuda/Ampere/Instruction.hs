module Aoewif.Target.Cuda.Ampere.Instruction (
    AmpereOp (..),
    CacheOp (..),
    CpAsyncSize (..),
    Fragment (..),
    LdMatrixForm (..),
    MmaShape (..),
) where

import           Aoewif.Target.Cuda.Syntax (Expr)

-- One sm_80+ instruction family. The constructors carry their operands as
-- rendered C++ expressions; register counts are implied by the shape, never
-- checked.

data MmaShape
    = M16N8K8F16
    | M16N8K16Tf32
    | M16N8K16Bf16
    | M8N8K4F64
    deriving stock (Eq, Show)

data LdMatrixForm
    = LdX1
    | LdX2
    | LdX4
    deriving stock (Eq, Show)

data CacheOp
    = CacheAll
    | CacheGlobal
    deriving stock (Eq, Show)

data CpAsyncSize
    = Bytes4
    | Bytes8
    | Bytes16
    deriving stock (Eq, Show)

data AmpereOp
    = Mma MmaShape [Expr] [Expr] [Expr]
    | LdMatrix LdMatrixForm [Expr] Expr
    | CpAsync CacheOp CpAsyncSize Expr Expr
    | CommitGroup
    | WaitGroup (Maybe Int)
    deriving stock (Show)

-- Fragments are bundles of U32 registers; all-zero bits is +0.0 for every mma
-- accumulator type.

newtype Fragment = Fragment
    { fragmentRegisters :: [Expr]
    }
