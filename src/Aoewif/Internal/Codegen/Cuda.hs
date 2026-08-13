module Aoewif.Internal.Codegen.Cuda (
    CudaSource,
    cudaSourceText,
    cudaKernelName,
    generateCuda,
) where

import           Aoewif.Internal.Codegen.Base     (Backend (..), generateSource,
                                                   generatedName, generatedText)
import qualified Aoewif.Internal.Kernel.Lower     as Kernel
import qualified Aoewif.Internal.Schedule.Builder as Builder

data CudaSource = CudaSource String String
    deriving stock (Eq, Show)

cudaSourceText :: CudaSource -> String
cudaSourceText (CudaSource sourceText _) = sourceText

cudaKernelName :: CudaSource -> String
cudaKernelName (CudaSource _ kernelName) = kernelName

generateCuda :: Builder.CudaSchedule -> CudaSource
generateCuda schedule =
    let generated = generateSource cudaBackend (Kernel.lowerCudaSchedule schedule)
     in CudaSource (generatedText generated) (generatedName generated)

cudaBackend :: Backend
cudaBackend =
    Backend
        { backendPreambleLines =
            [ "#include <cuda_runtime.h>"
            , "#include <math.h>"
            , "#include <stdbool.h>"
            , "#include <stddef.h>"
            , ""
            ]
        , backendFunctionPrefix = "__global__ void "
        , backendParallelDirective = Nothing
        , backendUnrollDirective = \factor -> "#pragma unroll " ++ show factor
        , backendAddExpression = function2 "__fadd_rn"
        , backendSubExpression = function2 "__fsub_rn"
        , backendMulExpression = function2 "__fmul_rn"
        , backendDivExpression = function2 "__fdiv_rn"
        , backendFmaExpression = function3 "__fmaf_rn"
        }

function2 :: String -> String -> String -> String
function2 function first second = function ++ "(" ++ first ++ ", " ++ second ++ ")"

function3 :: String -> String -> String -> String -> String
function3 function first second third =
    function ++ "(" ++ first ++ ", " ++ second ++ ", " ++ third ++ ")"
