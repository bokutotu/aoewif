module Aoewif.Access (
    Access (..),
) where

data Access
    = ReadOnly
    | ReadWrite
    deriving stock (Eq, Show)
