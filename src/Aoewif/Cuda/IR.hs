module Aoewif.Cuda.IR (
    SymbolId (..),
    BufferId (..),
    ValueId (..),
    LoopId (..),
    BufferAccess (..),
    Extent (..),
    Symbol (..),
    BufferDecl (..),
    SharedDecl (..),
    Launch (..),
    IndexExpr (..),
    Predicate (..),
    F32Expr (..),
    Statement (..),
    Kernel (..),
) where

import           Data.Word (Word32, Word64)

newtype SymbolId = SymbolId Int
    deriving stock (Eq, Ord, Show)

newtype BufferId = BufferId Int
    deriving stock (Eq, Ord, Show)

newtype ValueId = ValueId Int
    deriving stock (Eq, Ord, Show)

newtype LoopId = LoopId Int
    deriving stock (Eq, Ord, Show)

data BufferAccess
    = ReadOnly
    | ReadWrite
    deriving stock (Eq, Show)

data Extent
    = StaticExtent Word64
    | DynamicExtent SymbolId
    | CeilDivExtent Extent Word64
    deriving stock (Eq, Ord, Show)

data Symbol = Symbol
    { symbolId   :: SymbolId
    , symbolName :: String
    }
    deriving stock (Eq, Show)

data BufferDecl = BufferDecl
    { bufferId     :: BufferId
    , bufferName   :: String
    , bufferAccess :: BufferAccess
    , bufferShape  :: [Extent]
    }
    deriving stock (Eq, Show)

data SharedDecl = SharedDecl
    { sharedBufferId :: BufferId
    , sharedName     :: String
    , sharedShape    :: [Word64]
    }
    deriving stock (Eq, Show)

data Launch = Launch
    { launchGridExtent :: Extent
    , launchThreads    :: Word32
    }
    deriving stock (Eq, Show)

data IndexExpr
    = BlockIndexX
    | ThreadIndexX
    | LoopIndex LoopId
    | ConstantIndex Word64
    | ExtentIndex Extent
    | AddIndex IndexExpr IndexExpr
    | MulIndex IndexExpr IndexExpr
    deriving stock (Eq, Ord, Show)

data Predicate
    = IndexLessThan IndexExpr IndexExpr
    deriving stock (Eq, Show)

data F32Expr
    = F32Literal Float
    | F32Value ValueId
    | LoadF32 BufferId IndexExpr
    | AddF32 F32Expr F32Expr
    | SubF32 F32Expr F32Expr
    | MulF32 F32Expr F32Expr
    | DivF32 F32Expr F32Expr
    deriving stock (Eq, Show)

data Statement
    = LetF32 ValueId F32Expr
    | StoreF32 BufferId IndexExpr F32Expr
    | SerialFor LoopId Extent [Statement]
    | IfThen Predicate [Statement]
    | AllocateShared SharedDecl [Statement]
    | SyncThreads
    deriving stock (Eq, Show)

data Kernel = Kernel
    { kernelName    :: String
    , kernelSymbols :: [Symbol]
    , kernelBuffers :: [BufferDecl]
    , kernelLaunch  :: Launch
    , kernelBody    :: [Statement]
    }
    deriving stock (Eq, Show)
