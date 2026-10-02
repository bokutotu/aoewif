module Aoewif.Target.Cuda.Name (
    Name (..),
)
where

import           Aoewif.Target.Cuda.Render (Render (..))

newtype Name = Name String
    deriving stock (Eq, Ord, Show)

instance Render Name where
    render (Name name) = name
