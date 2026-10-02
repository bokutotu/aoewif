module Aoewif.Target.Cuda.Type (
    Type (..),
)
where

import           Aoewif.Target.Cuda.Render (Render (..))

data Type
    = Bool
    | U32
    | USize
    | F16
    | BF16
    | F32
    | Const Type
    | Pointer Type
    deriving stock (Eq, Show)

instance Render Type where
    render Bool                  = "bool"
    render U32                   = "uint32_t"
    render USize                 = "size_t"
    render F16                   = "__half"
    render BF16                  = "__nv_bfloat16"
    render F32                   = "float"
    render (Const valueType)     = render valueType ++ " const"
    render (Pointer pointeeType) = render pointeeType ++ "*"
