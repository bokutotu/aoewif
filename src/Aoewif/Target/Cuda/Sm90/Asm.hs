module Aoewif.Target.Cuda.Sm90.Asm (
    AsmOperand,
    asmLine,
    exprOperand,
    localSharedOperand,
    placeholder,
    placeholders,
    registerVector,
    renderAsm,
) where

import           Aoewif.Target.Cuda.Codegen (indent, renderExpr)
import           Aoewif.Target.Cuda.Syntax  (Expr)
import           Data.List                  (intercalate)

data AsmOperand = AsmOperand String String

exprOperand :: String -> Expr -> AsmOperand
exprOperand constraint =
    AsmOperand constraint . renderExpr

localSharedOperand :: Expr -> AsmOperand
localSharedOperand =
    AsmOperand "l" . sharedAddress

renderAsm :: Int -> String -> [AsmOperand] -> [AsmOperand] -> [String] -> String
renderAsm indentation instruction outputs inputs clobbers
    | null outputs && null inputs =
        asmLine indentation instruction clobbers
    | null outputs =
        unlines
            ( [ asmOpen indentation instruction
              , indent (indentation + 1) ++ ":: " ++ renderAsmOperands inputs
              ]
                ++ renderClobberLine indentation clobbers
                ++ [indent indentation ++ ");"]
            )
    | otherwise =
        unlines
            ( [ asmOpen indentation instruction
              , indent (indentation + 1) ++ ": " ++ renderAsmOperands outputs
              ]
                ++ renderInputLine indentation inputs clobbers
                ++ renderClobberLine indentation clobbers
                ++ [indent indentation ++ ");"]
            )

renderInputLine :: Int -> [AsmOperand] -> [String] -> [String]
renderInputLine indentation inputs clobbers
    | null inputs && null clobbers = []
    | null inputs = [indent (indentation + 1) ++ ":"]
    | otherwise =
        [indent (indentation + 1) ++ ": " ++ renderAsmOperands inputs]

renderClobberLine :: Int -> [String] -> [String]
renderClobberLine _ [] = []
renderClobberLine indentation clobbers =
    [indent (indentation + 1) ++ ": " ++ renderClobbers clobbers]

asmOpen :: Int -> String -> String
asmOpen indentation instruction =
    indent indentation ++ "asm volatile(\"" ++ instruction ++ "\""

asmLine :: Int -> String -> [String] -> String
asmLine indentation instruction clobbers =
    indent indentation
        ++ "asm volatile(\""
        ++ instruction
        ++ "\""
        ++ if null clobbers
            then ");\n"
            else " ::: " ++ renderClobbers clobbers ++ ");\n"

renderAsmOperands :: [AsmOperand] -> String
renderAsmOperands = intercalate ", " . fmap renderAsmOperand

renderAsmOperand :: AsmOperand -> String
renderAsmOperand (AsmOperand constraint value) = "\"" ++ constraint ++ "\"(" ++ value ++ ")"

renderClobbers :: [String] -> String
renderClobbers = intercalate ", " . fmap (\clobber -> "\"" ++ clobber ++ "\"")

placeholder :: Int -> String
placeholder index = "%" ++ show index

placeholders :: Int -> Int -> String
placeholders first count = intercalate ", " (fmap placeholder [first .. first + count - 1])

registerVector :: Int -> Int -> String
registerVector first count = "{" ++ placeholders first count ++ "}"

sharedAddress :: Expr -> String
sharedAddress address = "__cvta_generic_to_shared(&" ++ renderExpr address ++ ")"
