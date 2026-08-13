module Aoewif.Internal.Compute.IR (
    Name (..),
    SymbolId (..),
    TensorId (..),
    BlockId (..),
    AxisId (..),
    ComputeNodeId (..),
    DimExpr (..),
    Symbol (..),
    DType (..),
    TensorKind (..),
    TensorDecl (..),
    AxisKind (..),
    AxisDecl (..),
    IndexExpr (..),
    ScalarLiteral (..),
    scalarLiteralType,
    ComparePredicate (..),
    ScalarExpr (..),
    exprType,
    ReducerKind (..),
    Definition (..),
    ComputeBlock (..),
    ComputeIR (..),
    tensorAt,
    blockAt,
    axisAt,
) where

import           Data.List            (find)
import           Data.Maybe           (fromJust)
import           Data.Proxy           (Proxy (..))
import           Data.Word            (Word64)
import           GHC.OverloadedLabels (IsLabel (..))
import           GHC.TypeLits         (KnownSymbol, symbolVal)

newtype Name = Name String
    deriving stock (Eq, Ord, Show)

instance (KnownSymbol label) => IsLabel label Name where
    fromLabel = Name (symbolVal (Proxy @label))

newtype SymbolId = SymbolId Int
    deriving stock (Eq, Ord, Show)

newtype TensorId = TensorId Int
    deriving stock (Eq, Ord, Show)

newtype BlockId = BlockId Int
    deriving stock (Eq, Ord, Show)

newtype AxisId = AxisId Int
    deriving stock (Eq, Ord, Show)

newtype ComputeNodeId = ComputeNodeId Int
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

data DType
    = F32Type
    | BoolType
    | IndexType
    deriving stock (Eq, Show)

data TensorKind
    = InputTensor Int
    | OutputTensor Int
    deriving stock (Eq, Show)

data TensorDecl = TensorDecl
    { tensorId    :: TensorId
    , tensorName  :: Name
    , tensorType  :: DType
    , tensorShape :: [DimExpr]
    , tensorKind  :: TensorKind
    }
    deriving stock (Eq, Show)

data AxisKind
    = Spatial
    | Reduction
    deriving stock (Eq, Show)

data AxisDecl = AxisDecl
    { axisId     :: AxisId
    , axisName   :: Name
    , axisKind   :: AxisKind
    , axisLower  :: DimExpr
    , axisExtent :: DimExpr
    }
    deriving stock (Eq, Show)

data IndexExpr
    = AxisIndex AxisId
    | ConstantIndex Word64
    | AddIndex IndexExpr IndexExpr
    | MulIndex IndexExpr IndexExpr
    | CeilDivIndex IndexExpr Word64
    deriving stock (Eq, Ord, Show)

data ScalarLiteral
    = F32Literal Float
    | BoolLiteral Bool
    | IndexLiteral Word64
    deriving stock (Eq, Show)

scalarLiteralType :: ScalarLiteral -> DType
scalarLiteralType literal = case literal of
    F32Literal _   -> F32Type
    BoolLiteral _  -> BoolType
    IndexLiteral _ -> IndexType

data ComparePredicate
    = Equal
    | NotEqual
    | Less
    | LessEqual
    | Greater
    | GreaterEqual
    deriving stock (Eq, Show)

data ScalarExpr
    = LiteralExpr ScalarLiteral
    | IndexValueExpr AxisId
    | LoadExpr TensorId [IndexExpr]
    | AddExpr ScalarExpr ScalarExpr
    | SubExpr ScalarExpr ScalarExpr
    | MulExpr ScalarExpr ScalarExpr
    | DivExpr ScalarExpr ScalarExpr
    | FmaExpr ScalarExpr ScalarExpr ScalarExpr
    | MinExpr ScalarExpr ScalarExpr
    | MaxExpr ScalarExpr ScalarExpr
    | ExpExpr ScalarExpr
    | LogExpr ScalarExpr
    | CompareExpr ComparePredicate ScalarExpr ScalarExpr
    | SelectExpr ScalarExpr ScalarExpr ScalarExpr
    | NamedExpr ComputeNodeId Name ScalarExpr
    deriving stock (Eq, Show)

exprType :: ScalarExpr -> DType
exprType expression = case expression of
    LiteralExpr literal      -> scalarLiteralType literal
    IndexValueExpr _         -> IndexType
    LoadExpr _ _             -> F32Type
    AddExpr _ _              -> F32Type
    SubExpr _ _              -> F32Type
    MulExpr _ _              -> F32Type
    DivExpr _ _              -> F32Type
    FmaExpr{}                -> F32Type
    MinExpr _ _              -> F32Type
    MaxExpr _ _              -> F32Type
    ExpExpr _                -> F32Type
    LogExpr _                -> F32Type
    CompareExpr{}            -> BoolType
    SelectExpr _ trueValue _ -> exprType trueValue
    NamedExpr _ _ value      -> exprType value

data ReducerKind
    = AddReducer
    | MulReducer
    | MinReducer
    | MaxReducer
    deriving stock (Eq, Show)

data Definition
    = PointwiseDef
        { definitionTarget  :: TensorId
        , definitionIndices :: [IndexExpr]
        , definitionValue   :: ScalarExpr
        }
    | ReductionDef
        { definitionTarget     :: TensorId
        , definitionIndices    :: [IndexExpr]
        , definitionReducer    :: ReducerKind
        , definitionIdentity   :: ScalarExpr
        , definitionReduceAxes :: [AxisId]
        , definitionValue      :: ScalarExpr
        }
    deriving stock (Eq, Show)

data ComputeBlock = ComputeBlock
    { blockId          :: BlockId
    , blockName        :: Name
    , blockAxes        :: [AxisDecl]
    , blockDefinitions :: [Definition]
    }
    deriving stock (Eq, Show)

data ComputeIR = ComputeIR
    { computeName    :: Name
    , computeSymbols :: [Symbol]
    , computeTensors :: [TensorDecl]
    , computeBlocks  :: [ComputeBlock]
    }
    deriving stock (Eq, Show)

tensorAt :: TensorId -> ComputeIR -> TensorDecl
tensorAt identifier = fromJust . find ((== identifier) . tensorId) . computeTensors

blockAt :: BlockId -> ComputeIR -> ComputeBlock
blockAt identifier = fromJust . find ((== identifier) . blockId) . computeBlocks

axisAt :: AxisId -> ComputeBlock -> AxisDecl
axisAt identifier = fromJust . find ((== identifier) . axisId) . blockAxes
