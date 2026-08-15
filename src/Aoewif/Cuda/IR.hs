module Aoewif.Cuda.IR (
    Access (..),
    Argument (..),
    Block (..),
    DType (..),
    Element (..),
    Grid (..),
    Guard (..),
    Index (..),
    Kernel (..),
    KernelOp (..),
    Launch (..),
    Name (..),
    Parallel1D (..),
    Program (..),
    Statement (..),
)
where

newtype Name = Name String
    deriving stock (Eq, Show)

data DType
    = USize
    | F32
    deriving stock (Eq, Show)

data Access
    = ReadOnly
    | ReadWrite
    deriving stock (Eq, Show)

data Argument
    = ScalarArg Name DType
    | TensorArg Name Access DType
    deriving stock (Eq, Show)

data Program = Program [Argument] Kernel
    deriving stock (Eq, Show)

data Kernel = Kernel Name Launch [KernelOp]
    deriving stock (Eq, Show)

data Launch = Launch Grid Block
    deriving stock (Eq, Show)

newtype Grid = Grid1D Index
    deriving stock (Eq, Show)

newtype Block = Block1D Word
    deriving stock (Eq, Show)

newtype KernelOp = Parallel Parallel1D
    deriving stock (Eq, Show)

{- | A logical one-dimensional parallel iteration space.

It is named "parallel" rather than "thread" because its index describes
work exposed to lowering, not a guaranteed physical CUDA thread. The MVP
maps it one-to-one to @threadIdx.x@.
-}
data Parallel1D = Parallel1D Index [Statement]
    deriving stock (Eq, Show)

data Index
    = Size Name
    | Literal Word
    | BlockIndex
    | -- | The logical index bound by 'Parallel1D'.
      ParallelIndex
    | AddIndex Index Index
    | MultiplyIndex Index Index
    | CeilDiv Index Index
    deriving stock (Eq, Show)

data Element
    = Load Name Index
    | Add Element Element
    deriving stock (Eq, Show)

data Guard = InBounds Index Index
    deriving stock (Eq, Show)

data Statement
    = When Guard [Statement]
    | Store Name Index Element
    deriving stock (Eq, Show)
