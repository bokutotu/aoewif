{-# LANGUAGE QuasiQuotes #-}

module Aoewif.Target.Cuda.Sm90.Asm (
    constraints,
    placeholders,
    sharedAddress,
) where

import           Aoewif.Target.Cuda.Codegen (renderExpr)
import           Aoewif.Target.Cuda.Syntax  (Expr)
import           Data.List                  (intercalate)
import           Data.String.Interpolate    (i)

constraints :: String -> [Expr] -> String
constraints constraint registers =
    intercalate ", " [[i|"#{constraint}"(#{renderExpr register})|] | register <- registers]

placeholders :: Int -> Int -> String
placeholders first count =
    intercalate ", " [[i|%#{index}|] | index <- [first .. first + count - 1]]

sharedAddress :: Expr -> String
sharedAddress address =
    [i|__cvta_generic_to_shared(&#{renderExpr address})|]
