module Aoewif.Internal.Codegen.Cpu (
    CSource,
    cSourceText,
    cFunctionName,
    generateC,
) where

import           Aoewif.Internal.Codegen.Base      (Backend (..),
                                                    generateSource,
                                                    generatedName,
                                                    generatedText)
import qualified Aoewif.Internal.Compute.Operation as Compute
import qualified Aoewif.Internal.IR                as IR
import qualified Aoewif.Internal.Kernel.Lower      as Kernel

data CSource = CSource String String
    deriving stock (Eq, Show)

cSourceText :: CSource -> String
cSourceText (CSource sourceText _) = sourceText

cFunctionName :: CSource -> String
cFunctionName (CSource _ functionName) = functionName

generateC :: IR.IR Compute.ComputeBlock -> CSource
generateC computeIR =
    let generated = generateSource cpuBackend (Kernel.lower computeIR)
     in CSource (generatedText generated) (generatedName generated)

cpuBackend :: Backend
cpuBackend =
    Backend
        { backendPreambleLines =
            [ "#include <math.h>"
            , "#include <stdbool.h>"
            , "#include <stddef.h>"
            , ""
            , "#pragma STDC FP_CONTRACT OFF"
            , ""
            ]
        , backendFunctionPrefix = "void "
        , backendAddExpression = infixExpression " + "
        , backendSubExpression = infixExpression " - "
        , backendMulExpression = infixExpression " * "
        , backendDivExpression = infixExpression " / "
        , backendFmaExpression = function3 "fmaf"
        }

infixExpression :: String -> String -> String -> String
infixExpression operator lhs rhs = "(" ++ lhs ++ operator ++ rhs ++ ")"

function3 :: String -> String -> String -> String -> String
function3 function first second third =
    function ++ "(" ++ first ++ ", " ++ second ++ ", " ++ third ++ ")"
