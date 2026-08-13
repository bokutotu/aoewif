module Aoewif.Internal.Compute.Operation (
    ComputeIndexExpr (..),
    ComputeValueId (..),
    ScalarExpr (..),
    exprType,
    ReducerKind (..),
    ComputeStatement (..),
    ComputeBlock (..),
) where

import           Aoewif.Internal.IR
import           Data.Word          (Word64)

data ComputeIndexExpr
    = IterationIndex IndexId
    | ConstantComputeIndex Word64
    | AddComputeIndex ComputeIndexExpr ComputeIndexExpr
    | MulComputeIndex ComputeIndexExpr ComputeIndexExpr
    | CeilDivComputeIndex ComputeIndexExpr Word64
    deriving stock (Eq, Ord, Show)

newtype ComputeValueId = ComputeValueId Int
    deriving stock (Eq, Ord, Show)

data ScalarExpr
    = LiteralExpr ScalarLiteral
    | IndexValueExpr IndexId
    | ValueExpr ComputeValueId
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
    deriving stock (Eq, Show)

exprType :: ScalarExpr -> DType
exprType expression = case expression of
    LiteralExpr literal      -> scalarLiteralType literal
    IndexValueExpr _         -> IndexType
    ValueExpr _              -> F32Type
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

data ReducerKind
    = AddReducer
    | MulReducer
    | MinReducer
    | MaxReducer
    deriving stock (Eq, Show)

data ComputeStatement
    = Load
        { statementResult  :: ComputeValueId
        , statementSource  :: TensorId
        , statementIndices :: [ComputeIndexExpr]
        }
    | Store
        { statementTarget  :: TensorId
        , statementIndices :: [ComputeIndexExpr]
        , statementValue   :: ScalarExpr
        }
    | Update
        { statementReducer :: ReducerKind
        , statementTarget  :: TensorId
        , statementIndices :: [ComputeIndexExpr]
        , statementValue   :: ScalarExpr
        }
    deriving stock (Eq, Show)

newtype ComputeBlock = ComputeBlock
    { computeStatements :: [ComputeStatement]
    }
    deriving stock (Eq, Show)
