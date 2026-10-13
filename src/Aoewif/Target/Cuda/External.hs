module Aoewif.Target.Cuda.External (
    ExternalFunction,
    external,
)
where

import           Aoewif.Target.Cuda.DSL (Expr, call, var)

class ExternalFunction function where
    buildExternal :: Expr -> [Expr] -> function

instance ExternalFunction Expr where
    buildExternal function arguments =
        call function (reverse arguments)

instance (ExternalFunction result) => ExternalFunction (Expr -> result) where
    buildExternal function arguments argument =
        buildExternal function (argument : arguments)

external :: (ExternalFunction function) => String -> function
external name = buildExternal (var name) []
