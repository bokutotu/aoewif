module Aoewif.Target.Cuda.Render (
    Render (..),
    indent,
)
where

class Render value where
    render :: value -> String

indent :: String -> String
indent = unlines . map ("    " ++) . lines
