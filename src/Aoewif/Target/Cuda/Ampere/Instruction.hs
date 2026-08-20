module Aoewif.Target.Cuda.Ampere.Instruction (
    AmpereOp (..),
    CacheOp (..),
    CpAsyncSize (..),
    Fragment (..),
    LdMatrixForm (..),
    LdMatrixMode (..),
    MmaShape (..),
    ldMatrixRegisterCount,
) where

import           Aoewif.Target.Cuda.Syntax (Expr)

data MmaShape
    = M16N8K8F16
    | M16N8K8Tf32
    | M16N8K16Bf16
    | M8N8K4F64
    deriving stock (Eq, Show)

data LdMatrixForm
    = LdX1
    | LdX2
    | LdX4
    deriving stock (Eq, Show)

data LdMatrixMode
    = LdMatrixNormal
    | LdMatrixTranspose
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
    | LdMatrix LdMatrixMode LdMatrixForm [Expr] Expr
    | MovMatrix Expr
    | CpAsync CacheOp CpAsyncSize (Maybe Expr) Expr Expr
    | CommitGroup
    | WaitGroup (Maybe Int)
    deriving stock (Show)

newtype Fragment = Fragment
    { fragmentRegisters :: [Expr]
    }

ldMatrixRegisterCount :: LdMatrixForm -> Int
ldMatrixRegisterCount LdX1 = 1
ldMatrixRegisterCount LdX2 = 2
ldMatrixRegisterCount LdX4 = 4
