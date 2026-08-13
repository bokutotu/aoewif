module Aoewif.Internal.Codegen.Cpu (
    CSource,
    cSourceText,
    cFunctionName,
    generateC,
) where

import           Aoewif.Internal.Codegen.Base     (Backend (..), generateSource,
                                                   generatedName, generatedText)
import qualified Aoewif.Internal.Kernel.Lower     as Kernel
import qualified Aoewif.Internal.Schedule.Builder as Builder

data CSource = CSource String String
    deriving stock (Eq, Show)

cSourceText :: CSource -> String
cSourceText (CSource sourceText _) = sourceText

cFunctionName :: CSource -> String
cFunctionName (CSource _ functionName) = functionName

generateC :: Builder.CpuSchedule -> CSource
generateC schedule =
    let generated = generateSource cpuBackend (Kernel.lowerCpuSchedule schedule)
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
        , backendParallelDirective = Just "#pragma omp parallel for"
        , backendUnrollDirective = \factor -> "#pragma GCC unroll " ++ show factor
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
