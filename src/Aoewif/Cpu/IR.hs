module Aoewif.Cpu.IR (
    Name (..),
    BufferId (..),
    IndexId (..),
    ValueId (..),
    Access (..),
    Extent (..),
    Buffer (..),
    Expr (..),
    LoopKind (..),
    Loop (..),
    Statement (..),
    Program (..),
) where

import           Data.Proxy           (Proxy (..))
import           Data.Word            (Word64)
import           GHC.OverloadedLabels (IsLabel (..))
import           GHC.TypeLits         (KnownSymbol, symbolVal)

newtype Name = Name String
    deriving stock (Eq, Ord, Show)

instance (KnownSymbol label) => IsLabel label Name where
    fromLabel = Name (symbolVal (Proxy @label))

newtype BufferId = BufferId Int
    deriving stock (Eq, Ord, Show)

newtype IndexId = IndexId Int
    deriving stock (Eq, Ord, Show)

newtype ValueId = ValueId Int
    deriving stock (Eq, Ord, Show)

data Access
    = ReadOnly
    | ReadWrite
    deriving stock (Eq, Show)

data Extent
    = StaticExtent Word64
    | DynamicExtent Name
    deriving stock (Eq, Show)

data Buffer = Buffer
    { bufferId     :: BufferId
    , bufferName   :: Name
    , bufferAccess :: Access
    , bufferShape  :: [Extent]
    }
    deriving stock (Eq, Show)

data Expr
    = F32Literal Float
    | ValueExpr ValueId
    | AddExpr Expr Expr
    | SubExpr Expr Expr
    | MulExpr Expr Expr
    | DivExpr Expr Expr
    deriving stock (Eq, Show)

data LoopKind
    = Serial
    | Parallel
    deriving stock (Eq, Show)

data Loop = Loop
    { loopKind   :: LoopKind
    , loopIndex  :: IndexId
    , loopName   :: Name
    , loopExtent :: Extent
    , loopBody   :: [Statement]
    }
    deriving stock (Eq, Show)

data Statement
    = Let ValueId BufferId [IndexId]
    | Store BufferId [IndexId] Expr
    | For Loop
    | Allocate Buffer [Statement]
    deriving stock (Eq, Show)

data Program = Program
    { programName    :: Name
    , programExtents :: [Name]
    , programBuffers :: [Buffer]
    , programBody    :: [Statement]
    }
    deriving stock (Eq, Show)
