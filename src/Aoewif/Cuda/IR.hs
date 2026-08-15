module Aoewif.Cuda.IR (
    Name (..),
    SymbolId (..),
    BufferId (..),
    ValueId (..),
    LoopId (..),
    Access (..),
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

import           Aoewif.Access   (Access (..))
import           Aoewif.BufferId (BufferId (..))
import           Aoewif.LoopId   (LoopId (..))
import           Aoewif.Name     (Name (..))
import           Aoewif.ValueId  (ValueId (..))
import           Data.Word       (Word32, Word64)

newtype SymbolId = SymbolId Int
    deriving stock (Eq, Ord, Show)

data Extent
    = StaticExtent Word64
    | DynamicExtent SymbolId
    | CeilDivExtent Extent Word64
    deriving stock (Eq, Ord, Show)

data Symbol = Symbol
    { symbolId   :: SymbolId
    , symbolName :: Name
    }
    deriving stock (Eq, Show)

data BufferDecl = BufferDecl
    { bufferId     :: BufferId
    , bufferName   :: Name
    , bufferAccess :: Access
    , bufferShape  :: [Extent]
    }
    deriving stock (Eq, Show)

data SharedDecl = SharedDecl
    { sharedBufferId :: BufferId
    , sharedName     :: Name
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
    { kernelName    :: Name
    , kernelSymbols :: [Symbol]
    , kernelBuffers :: [BufferDecl]
    , kernelLaunch  :: Launch
    , kernelBody    :: [Statement]
    }
    deriving stock (Eq, Show)
