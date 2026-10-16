{-# LANGUAGE TemplateHaskell #-}

module Aoewif.Target.Template (template) where

import           Data.List                 (dropWhileEnd, intercalate)
import           Language.Haskell.TH       (Exp, Q, listE, mkName, stringE,
                                            varE)
import           Language.Haskell.TH.Quote (QuasiQuoter (..))

data NormalizedLine = NormalizedLine
    { lineIndentation :: String
    , lineBody        :: String
    }

data TemplatePart
    = Literal String
    | Variable String

data LineContent
    = Inline [TemplatePart]
    | StandaloneVariable String

data TemplateLine = TemplateLine
    { indentation :: String
    , content     :: LineContent
    }

template :: QuasiQuoter
template =
    QuasiQuoter
        { quoteExp = quoteTemplate
        , quotePat = const (fail "template: only expression quotation is supported")
        , quoteType = const (fail "template: only expression quotation is supported")
        , quoteDec = const (fail "template: only expression quotation is supported")
        }

quoteTemplate :: String -> Q Exp
quoteTemplate source = do
    parsedLines <- traverse parseLine (normalize source)
    [|concat $(listE (compileLines parsedLines))|]

normalize :: String -> [NormalizedLine]
normalize =
    removeCommonIndentation
        . removeBoundaryLines
        . map normalizeLine
        . lines

normalizeLine :: String -> NormalizedLine
normalizeLine line =
    let (indent, body) = span horizontalSpace line
     in NormalizedLine
            { lineIndentation = indent
            , lineBody = dropWhileEnd horizontalSpace body
            }

removeBoundaryLines :: [NormalizedLine] -> [NormalizedLine]
removeBoundaryLines = removeLastBlank . removeFirstBlank
  where
    removeFirstBlank (line : rest)
        | null (lineBody line) = rest
    removeFirstBlank xs = xs

    removeLastBlank [] = []
    removeLastBlank xs
        | null (lineBody (last xs)) = init xs
        | otherwise = xs

removeCommonIndentation :: [NormalizedLine] -> [NormalizedLine]
removeCommonIndentation sourceLines =
    map removeIndent sourceLines
  where
    widths = [length (lineIndentation line) | line <- sourceLines, not (null (lineBody line))]
    commonWidth = case widths of
        [] -> 0
        _  -> minimum widths

    removeIndent line = line{lineIndentation = drop commonWidth (lineIndentation line)}

horizontalSpace :: Char -> Bool
horizontalSpace c = c == ' ' || c == '\t'

parseLine :: NormalizedLine -> Q TemplateLine
parseLine line = do
    parts <- parseParts (lineBody line)
    let lineContent = case parts of
            [Variable name] -> StandaloneVariable name
            _               -> Inline parts
    pure
        TemplateLine
            { indentation = lineIndentation line
            , content = lineContent
            }

parseParts :: String -> Q [TemplatePart]
parseParts [] = pure []
parseParts ('$' : '$' : rest) = (Literal "$" :) <$> parseParts rest
parseParts ('$' : '{' : rest) =
    case break (== '}') rest of
        (_, [])               -> fail "template: unclosed placeholder"
        (name, _ : remaining) -> (Variable name :) <$> parseParts remaining
parseParts ('$' : rest) = (Literal "$" :) <$> parseParts rest
parseParts text =
    let (literal, rest) = span (/= '$') text
     in (Literal literal :) <$> parseParts rest

compileLines :: [TemplateLine] -> [Q Exp]
compileLines []            = []
compileLines [line]        = [compileLine line False]
compileLines (line : rest) = compileLine line True : compileLines rest

compileLine :: TemplateLine -> Bool -> Q Exp
compileLine line appendNewline =
    case content line of
        Inline parts ->
            let ending = if appendNewline then "\n" else ""
                completeLine = Literal (indentation line) : parts ++ [Literal ending]
             in [|concat $(listE (map compilePart completeLine))|]
        StandaloneVariable name ->
            [|
                indentVariable
                    $(stringE (indentation line))
                    $(if appendNewline then [|True|] else [|False|])
                    $(compilePart (Variable name))
                |]

compilePart :: TemplatePart -> Q Exp
compilePart (Literal text)  = stringE text
compilePart (Variable name) = [|$(varE (mkName name)) :: String|]

indentVariable :: String -> Bool -> String -> String
indentVariable _ _ "" = ""
indentVariable indent appendNewline value =
    intercalate "\n" (map (indent ++) (lines value))
        ++ if appendNewline then "\n" else ""
