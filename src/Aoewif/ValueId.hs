module Aoewif.ValueId (
    ValueId (..),
) where

newtype ValueId = ValueId Int
    deriving stock (Eq, Ord, Show)
