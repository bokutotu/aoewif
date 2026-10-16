{-# LANGUAGE QuasiQuotes #-}

module Aoewif.Target.Cuda.Parameter (
    Parameter (..),
)
where

import           Aoewif.Target.Cuda.Name   (Name)
import           Aoewif.Target.Cuda.Render (Render (..))
import           Aoewif.Target.Cuda.Type   (Type)
import           Aoewif.Target.Template    (template)

data Parameter = Parameter Type Name
    deriving stock (Eq, Show)

instance Render Parameter where
    render (Parameter parameterType name) =
        [template|${typeText} ${nameText}|]
      where
        typeText = render parameterType
        nameText = render name
