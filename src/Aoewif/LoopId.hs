module Aoewif.LoopId (
    LoopId (..),
) where

newtype LoopId = LoopId Int
    deriving stock (Eq, Ord, Show)
