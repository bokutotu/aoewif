{-# LANGUAGE QuasiQuotes #-}

module Aoewif.Target.Cuda.Kernel (
    Kernel (..),
)
where

import           Aoewif.Target.Cuda.Name      (Name)
import           Aoewif.Target.Cuda.Parameter (Parameter)
import           Aoewif.Target.Cuda.Render    (Render (..))
import           Aoewif.Target.Cuda.Stmt      (Stmt)
import           Aoewif.Target.Template       (template)
import           Data.List                    (intercalate)

data Kernel = Kernel
    { kernelName       :: Name
    , kernelParameters :: [Parameter]
    , kernelBody       :: [Stmt]
    }
    deriving stock (Show)

instance Render Kernel where
    render (Kernel name parameters body) =
        [template|
        extern "C" __global__ void ${nameText}(${parametersText}) {
            ${bodyText}
        }
        |]
            ++ "\n"
      where
        nameText = render name
        parametersText = intercalate ", " (map render parameters)
        bodyText = concatMap render body
