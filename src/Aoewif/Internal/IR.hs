module Aoewif.Internal.IR (
    Name (..),
    SymbolId (..),
    TensorId (..),
    IndexId (..),
    LoopId (..),
    BlockId (..),
    DimExpr (..),
    Symbol (..),
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
    tensorAt,
) where

import           Aoewif.Internal.Primitive (DataType)
import           Data.List                 (find)
import           Data.Maybe                (fromJust)
import           Data.Proxy                (Proxy (..))
import           Data.Word                 (Word64)
import           GHC.OverloadedLabels      (IsLabel (..))
import           GHC.TypeLits              (KnownSymbol, symbolVal)

newtype Name = Name String
    deriving stock (Eq, Ord, Show)

instance (KnownSymbol label) => IsLabel label Name where
    fromLabel = Name (symbolVal (Proxy @label))

newtype SymbolId = SymbolId Int
    deriving stock (Eq, Ord, Show)

newtype TensorId = TensorId Int
    deriving stock (Eq, Ord, Show)

newtype IndexId = IndexId Int
    deriving stock (Eq, Ord, Show)

newtype LoopId = LoopId Int
    deriving stock (Eq, Ord, Show)

newtype BlockId = BlockId Int
    deriving stock (Eq, Ord, Show)

data DimExpr
    = StaticDim Word64
    | SymbolDim SymbolId
    | CeilDivDim DimExpr Word64
    deriving stock (Eq, Ord, Show)

data Symbol = Symbol
    { symbolId   :: SymbolId
    , symbolName :: Name
    }
    deriving stock (Eq, Show)

data TensorKind
    = InputTensor Int
    | OutputTensor Int
    deriving stock (Eq, Show)

data TensorDecl = TensorDecl
    { tensorId       :: TensorId
    , tensorName     :: Name
    , tensorDataType :: DataType
    , tensorShape    :: [DimExpr]
    , tensorKind     :: TensorKind
    }
    deriving stock (Eq, Show)

data IndexExpr
    = LoopIndex LoopId
    | DimensionIndex DimExpr
    | ConstantIndex Word64
    | AddIndex IndexExpr IndexExpr
    | MulIndex IndexExpr IndexExpr
    | CeilDivIndex IndexExpr Word64
    deriving stock (Eq, Ord, Show)

data Predicate
    = IndexLessThan IndexExpr IndexExpr
    | IndexEqual IndexExpr IndexExpr
    deriving stock (Eq, Show)

data IndexBinding = IndexBinding
    { bindingIndex      :: IndexId
    , bindingExpression :: IndexExpr
    }
    deriving stock (Eq, Show)

data Loop = Loop
    { loopId         :: LoopId
    , loopName       :: Name
    , loopLowerBound :: DimExpr
    , loopExtent     :: DimExpr
    }
    deriving stock (Eq, Show)

data Block operation = Block
    { blockId        :: BlockId
    , blockName      :: Name
    , blockBindings  :: [IndexBinding]
    , blockOperation :: operation
    }
    deriving stock (Eq, Show)

data Statement operation
    = For Loop (LoopIR operation)
    | Guard Predicate (LoopIR operation)
    | Execute (Block operation)
    deriving stock (Eq, Show)

newtype LoopIR operation = LoopIR
    { loopIRStatements :: [Statement operation]
    }
    deriving stock (Eq, Show)

data IR operation = IR
    { irName    :: Name
    , irSymbols :: [Symbol]
    , irTensors :: [TensorDecl]
    , irBody    :: LoopIR operation
    }
    deriving stock (Eq, Show)

tensorAt :: TensorId -> IR operation -> TensorDecl
tensorAt identifier = fromJust . find ((== identifier) . tensorId) . irTensors
