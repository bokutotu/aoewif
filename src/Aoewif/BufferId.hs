module Aoewif.BufferId (
    BufferId (..),
) where

newtype BufferId = BufferId Int
    deriving stock (Eq, Ord, Show)
