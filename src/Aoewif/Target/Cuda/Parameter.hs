module Aoewif.Target.Cuda.Parameter (
    Parameter (..),
)
where

import           Aoewif.Target.Cuda.Name   (Name)
import           Aoewif.Target.Cuda.Render (Render (..))
import           Aoewif.Target.Cuda.Type   (Type)

data Parameter = Parameter Type Name
    deriving stock (Eq, Show)

instance Render Parameter where
    render (Parameter parameterType name) =
        render parameterType ++ " " ++ render name
