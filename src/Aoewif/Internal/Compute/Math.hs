{-# LANGUAGE RoleAnnotations #-}

module Aoewif.Internal.Compute.Math (
    Expr (..),
    boolean,
    index,
    indexLiteral,
    fma,
    min_,
    max_,
    exp_,
    log_,
    compare_,
    select,
) where

import           Aoewif.Internal.Compute.Operation (ComputeIndexExpr (..),
                                                    DataExpr (..),
                                                    IndexValueExpr (..),
                                                    PredicateExpr (..))
import           Aoewif.Internal.Compute.State     (Boolean, F32, Index (..),
                                                    IndexValue)
import           Aoewif.Internal.Primitive         (ComparePredicate (..))
import           Data.Word                         (Word64)
import           Prelude                           hiding (exp, log, maximum,
                                                    minimum)

data Expr scope element where
    DataExpression :: DataExpr -> Expr scope F32
    PredicateExpression :: PredicateExpr -> Expr scope Boolean
    IndexValueExpression :: IndexValueExpr -> Expr scope IndexValue

type role Expr representational nominal

boolean :: Bool -> Expr scope Boolean
boolean = PredicateExpression . PredicateLiteralExpr

index :: Index scope -> Expr scope IndexValue
index (Index identifier) =
    IndexValueExpression (ComputeIndexValueExpr (IterationIndex identifier))

indexLiteral :: Word64 -> Expr scope IndexValue
indexLiteral =
    IndexValueExpression . ComputeIndexValueExpr . ConstantComputeIndex

fma :: Expr scope F32 -> Expr scope F32 -> Expr scope F32 -> Expr scope F32
fma lhs rhs accumulator =
    DataExpression (FmaExpr (dataValue lhs) (dataValue rhs) (dataValue accumulator))

min_ :: Expr scope F32 -> Expr scope F32 -> Expr scope F32
min_ lhs rhs = DataExpression (MinExpr (dataValue lhs) (dataValue rhs))

max_ :: Expr scope F32 -> Expr scope F32 -> Expr scope F32
max_ lhs rhs = DataExpression (MaxExpr (dataValue lhs) (dataValue rhs))

exp_ :: Expr scope F32 -> Expr scope F32
exp_ = DataExpression . ExpExpr . dataValue

log_ :: Expr scope F32 -> Expr scope F32
log_ = DataExpression . LogExpr . dataValue

compare_ :: ComparePredicate -> Expr scope element -> Expr scope element -> Expr scope Boolean
compare_ predicate (DataExpression lhs) (DataExpression rhs) =
    PredicateExpression (CompareDataExpr predicate lhs rhs)
compare_ predicate (PredicateExpression lhs) (PredicateExpression rhs) =
    PredicateExpression (CompareBooleanExpr predicate lhs rhs)
compare_ predicate (IndexValueExpression lhs) (IndexValueExpression rhs) =
    PredicateExpression (CompareIndexExpr predicate lhs rhs)

select :: Expr scope Boolean -> Expr scope element -> Expr scope element -> Expr scope element
select (PredicateExpression condition) (DataExpression trueValue) (DataExpression falseValue) = DataExpression (SelectDataExpr condition trueValue falseValue)
select (PredicateExpression condition) (PredicateExpression trueValue) (PredicateExpression falseValue) = PredicateExpression (SelectPredicateExpr condition trueValue falseValue)
select (PredicateExpression condition) (IndexValueExpression trueValue) (IndexValueExpression falseValue) = IndexValueExpression (SelectIndexValueExpr condition trueValue falseValue)

dataValue :: Expr scope F32 -> DataExpr
dataValue (DataExpression value) = value

instance Num (Expr scope F32) where
    lhs + rhs = DataExpression (AddExpr (dataValue lhs) (dataValue rhs))
    lhs - rhs = DataExpression (SubExpr (dataValue lhs) (dataValue rhs))
    lhs * rhs = DataExpression (MulExpr (dataValue lhs) (dataValue rhs))
    negate value = 0 - value
    abs value = max_ value (negate value)
    signum value =
        select
            (compare_ Greater value 0)
            1
            (select (compare_ Less value 0) (-1) 0)
    fromInteger = DataExpression . DataLiteralExpr . fromInteger

instance Fractional (Expr scope F32) where
    lhs / rhs = DataExpression (DivExpr (dataValue lhs) (dataValue rhs))
    fromRational = DataExpression . DataLiteralExpr . fromRational
