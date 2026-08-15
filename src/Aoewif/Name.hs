module Aoewif.Name (
    Name (..),
) where

import           Data.Proxy           (Proxy (..))
import           GHC.OverloadedLabels (IsLabel (..))
import           GHC.TypeLits         (KnownSymbol, symbolVal)

newtype Name = Name String
    deriving stock (Eq, Ord, Show)

instance (KnownSymbol label) => IsLabel label Name where
    fromLabel = Name (symbolVal (Proxy @label))
