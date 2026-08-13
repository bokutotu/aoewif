module Aoewif.Internal.Compute.IR (
    SymbolId (..),
    TensorId (..),
    ComputeId (..),
    IndexId (..),
    AccumulatorId (..),
    ComputeNodeId (..),
    Dim (..),
    Symbol (..),
    TensorKind (..),
    Tensor (..),
    IndexVar (..),
    IndexExpression (..),
    ScalarType (..),
    ScalarLiteral (..),
    scalarLiteralType,
    ComparePredicate (..),
    Expression (..),
    expressionType,
    reductionIndices,
    Compute (..),
    computeId,
    Program (..),
    computeAt,
) where

import           Data.Word (Word64)

newtype SymbolId = SymbolId Int
    deriving stock (Eq, Ord, Show)

newtype TensorId = TensorId Int
    deriving stock (Eq, Ord, Show)

newtype ComputeId = ComputeId Int
    deriving stock (Eq, Ord, Show)

newtype IndexId = IndexId Int
    deriving stock (Eq, Ord, Show)

newtype AccumulatorId = AccumulatorId Int
    deriving stock (Eq, Ord, Show)

newtype ComputeNodeId = ComputeNodeId Int
    deriving stock (Eq, Ord, Show)

data Dim
    = StaticDim Word64
    | SymbolDim SymbolId
    deriving stock (Eq, Show)

data Symbol = Symbol
    { symbolId   :: SymbolId
    , symbolName :: String
    }
    deriving stock (Eq, Show)

data TensorKind
    = InputTensor Int
    | ResultTensor ComputeId
    deriving stock (Eq, Show)

data Tensor = Tensor
    { tensorId    :: TensorId
    , tensorName  :: String
    , tensorShape :: [Dim]
    , tensorKind  :: TensorKind
    }
    deriving stock (Eq, Show)

data IndexVar = IndexVar
    { indexId     :: IndexId
    , indexName   :: String
    , indexExtent :: Dim
    }
    deriving stock (Eq, Show)

data IndexExpression
    = VariableIndex IndexVar
    | ConstantIndex Word64
    deriving stock (Eq, Show)

data ScalarType
    = F32Type
    | BoolType
    | IndexType
    deriving stock (Eq, Show)

data ScalarLiteral
    = F32Literal Float
    | BoolLiteral Bool
    | IndexLiteral Word64
    deriving stock (Eq, Show)

scalarLiteralType :: ScalarLiteral -> ScalarType
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

data Expression
    = LiteralExpression ScalarLiteral
    | IndexValueExpression IndexVar
    | ReadExpression Tensor [IndexExpression]
    | AccumulatorExpression AccumulatorId
    | AddExpression Expression Expression
    | SubExpression Expression Expression
    | MulExpression Expression Expression
    | DivExpression Expression Expression
    | FmaExpression Expression Expression Expression
    | MinExpression Expression Expression
    | MaxExpression Expression Expression
    | ExpExpression Expression
    | LogExpression Expression
    | CompareExpression ComparePredicate Expression Expression
    | SelectExpression Expression Expression Expression
    | FoldExpression IndexVar Float AccumulatorId Expression
    | NamedExpression ComputeNodeId String Expression
    deriving stock (Eq, Show)

expressionType :: Expression -> ScalarType
expressionType expression = case expression of
    LiteralExpression literal        -> scalarLiteralType literal
    IndexValueExpression _           -> IndexType
    ReadExpression _ _               -> F32Type
    AccumulatorExpression _          -> F32Type
    AddExpression _ _                -> F32Type
    SubExpression _ _                -> F32Type
    MulExpression _ _                -> F32Type
    DivExpression _ _                -> F32Type
    FmaExpression _ _ _              -> F32Type
    MinExpression _ _                -> F32Type
    MaxExpression _ _                -> F32Type
    ExpExpression _                  -> F32Type
    LogExpression _                  -> F32Type
    CompareExpression _ _ _          -> BoolType
    SelectExpression _ trueValue _   -> expressionType trueValue
    FoldExpression _ _ _ _           -> F32Type
    NamedExpression _ _ namedCompute -> expressionType namedCompute

reductionIndices :: Expression -> [IndexVar]
reductionIndices expression = case expression of
    LiteralExpression _ -> []
    IndexValueExpression _ -> []
    ReadExpression _ _ -> []
    AccumulatorExpression _ -> []
    AddExpression lhs rhs -> nested [lhs, rhs]
    SubExpression lhs rhs -> nested [lhs, rhs]
    MulExpression lhs rhs -> nested [lhs, rhs]
    DivExpression lhs rhs -> nested [lhs, rhs]
    FmaExpression lhs rhs accumulator -> nested [lhs, rhs, accumulator]
    MinExpression lhs rhs -> nested [lhs, rhs]
    MaxExpression lhs rhs -> nested [lhs, rhs]
    ExpExpression value -> reductionIndices value
    LogExpression value -> reductionIndices value
    CompareExpression _ lhs rhs -> nested [lhs, rhs]
    SelectExpression condition trueValue falseValue -> nested [condition, trueValue, falseValue]
    FoldExpression index _ _ body -> index : reductionIndices body
    NamedExpression _ _ namedCompute -> reductionIndices namedCompute
  where
    nested = concatMap reductionIndices

data Compute = Compute
    { computeName    :: String
    , computeIndices :: [IndexVar]
    , computeResult  :: Tensor
    , computeBody    :: Expression
    }
    deriving stock (Eq, Show)

computeId :: Compute -> ComputeId
computeId compute = case tensorKind (computeResult compute) of
    ResultTensor identifier -> identifier
    InputTensor _           -> error "a compute result must be a result tensor"

data Program = Program
    { programName     :: String
    , programSymbols  :: [Symbol]
    , programInputs   :: [Tensor]
    , programComputes :: [Compute]
    , programOutput   :: Tensor
    }
    deriving stock (Eq, Show)

computeAt :: ComputeId -> Program -> Compute
computeAt (ComputeId identifier) program = programComputes program !! identifier
