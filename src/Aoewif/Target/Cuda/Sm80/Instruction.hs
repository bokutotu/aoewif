module Aoewif.Target.Cuda.Sm80.Instruction (
    Sm80Op (..),
    CpAsyncShape (..),
    Fragment (..),
    LdMatrixForm (..),
    LdMatrixMode (..),
    MmaShape (..),
    ldMatrixRegisterCount,
) where

import           Aoewif.Target.Cuda.Syntax (Expr)

data MmaShape
    = M8N8K4F16
    | M16N8K8F16
    | M16N8K16F16
    | M16N8K8BF16
    | M16N8K16BF16
    | M16N8K4TF32
    | M16N8K8TF32
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

data CpAsyncShape
    = CacheAll4
    | CacheAll8
    | CacheAll16
    | CacheGlobal16
    deriving stock (Eq, Show)

data Sm80Op
    = Mma MmaShape [Expr] [Expr] [Expr]
    | LdMatrix LdMatrixMode LdMatrixForm [Expr] Expr
    | MovMatrix Expr
    | CpAsync CpAsyncShape (Maybe Expr) Expr Expr
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
