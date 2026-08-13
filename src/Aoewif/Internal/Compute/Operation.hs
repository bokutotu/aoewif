module Aoewif.Internal.Compute.Operation (
    ComputeIndexExpr (..),
    ComputeValueId (..),
    DataExpr (..),
    PredicateExpr (..),
    IndexValueExpr (..),
    ReducerKind (..),
    ComputeStatement (..),
    ComputeBlock (..),
) where

import           Aoewif.Internal.IR        (IndexId, TensorId)
import           Aoewif.Internal.Primitive (ComparePredicate)
import           Data.Word                 (Word64)

data ComputeIndexExpr
    = IterationIndex IndexId
    | ConstantComputeIndex Word64
    | AddComputeIndex ComputeIndexExpr ComputeIndexExpr
    | MulComputeIndex ComputeIndexExpr ComputeIndexExpr
    | CeilDivComputeIndex ComputeIndexExpr Word64
    deriving stock (Eq, Ord, Show)

newtype ComputeValueId = ComputeValueId Int
    deriving stock (Eq, Ord, Show)

data DataExpr
    = DataLiteralExpr Float
    | ValueExpr ComputeValueId
    | AddExpr DataExpr DataExpr
    | SubExpr DataExpr DataExpr
    | MulExpr DataExpr DataExpr
    | DivExpr DataExpr DataExpr
    | FmaExpr DataExpr DataExpr DataExpr
    | MinExpr DataExpr DataExpr
    | MaxExpr DataExpr DataExpr
    | ExpExpr DataExpr
    | LogExpr DataExpr
    | SelectDataExpr PredicateExpr DataExpr DataExpr
    deriving stock (Eq, Show)

data PredicateExpr
    = PredicateLiteralExpr Bool
    | CompareDataExpr ComparePredicate DataExpr DataExpr
    | CompareBooleanExpr ComparePredicate PredicateExpr PredicateExpr
    | CompareIndexExpr ComparePredicate IndexValueExpr IndexValueExpr
    | SelectPredicateExpr PredicateExpr PredicateExpr PredicateExpr
    deriving stock (Eq, Show)

data IndexValueExpr
    = ComputeIndexValueExpr ComputeIndexExpr
    | SelectIndexValueExpr PredicateExpr IndexValueExpr IndexValueExpr
    deriving stock (Eq, Show)

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
        , statementValue   :: DataExpr
        }
    | Update
        { statementReducer :: ReducerKind
        , statementTarget  :: TensorId
        , statementIndices :: [ComputeIndexExpr]
        , statementValue   :: DataExpr
        }
    deriving stock (Eq, Show)

newtype ComputeBlock = ComputeBlock
    { computeStatements :: [ComputeStatement]
    }
    deriving stock (Eq, Show)
