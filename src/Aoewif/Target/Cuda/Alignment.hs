module Aoewif.Target.Cuda.Alignment (
    Alignment (..),
)
where

import           Aoewif.Target.Cuda.Render (Render (..))

data Alignment
    = NaturalAlignment
    | Align16
    | Align128
    deriving stock (Eq, Show)

instance Render Alignment where
    render NaturalAlignment = ""
    render Align16          = "__align__(16) "
    render Align128         = "__align__(128) "
