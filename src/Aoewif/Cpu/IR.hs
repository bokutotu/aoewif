module Aoewif.Cpu.IR (
    Name (..),
    BufferId (..),
    LoopId (..),
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

import           Aoewif.Access   (Access (..))
import           Aoewif.BufferId (BufferId (..))
import           Aoewif.LoopId   (LoopId (..))
import           Aoewif.Name     (Name (..))
import           Aoewif.ValueId  (ValueId (..))
import           Data.Word       (Word64)

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
    , loopIndex  :: LoopId
    , loopName   :: Name
    , loopExtent :: Extent
    , loopBody   :: [Statement]
    }
    deriving stock (Eq, Show)

data Statement
    = Let ValueId BufferId [LoopId]
    | Store BufferId [LoopId] Expr
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
