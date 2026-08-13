module Aoewif.Internal.Primitive (
    DataType (..),
    ValueType (..),
    ComparePredicate (..),
) where

data DataType
    = F32Type
    deriving stock (Eq, Show)

data ValueType
    = DataValueType DataType
    | PredicateValueType
    | IndexValueType
    deriving stock (Eq, Show)

data ComparePredicate
    = Equal
    | NotEqual
    | Less
    | LessEqual
    | Greater
    | GreaterEqual
    deriving stock (Eq, Show)
