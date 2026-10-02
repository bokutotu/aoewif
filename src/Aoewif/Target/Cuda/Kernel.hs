module Aoewif.Target.Cuda.Kernel (
    Kernel (..),
)
where

import           Aoewif.Target.Cuda.Name      (Name)
import           Aoewif.Target.Cuda.Parameter (Parameter)
import           Aoewif.Target.Cuda.Render    (Render (..), indent)
import           Aoewif.Target.Cuda.Stmt      (Stmt)
import           Data.List                    (intercalate)

data Kernel = Kernel
    { kernelName       :: Name
    , kernelParameters :: [Parameter]
    , kernelBody       :: [Stmt]
    }
    deriving stock (Show)

instance Render Kernel where
    render (Kernel name parameters body) =
        "extern \"C\" __global__ void "
            ++ render name
            ++ "("
            ++ intercalate ", " (map render parameters)
            ++ ") {\n"
            ++ indent (concatMap render body)
            ++ "}\n"
