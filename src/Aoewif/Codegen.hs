module Aoewif.Codegen (
    CSource,
    cSourceText,
    cFunctionName,
    CudaSource,
    cudaSourceText,
    cudaKernelName,
    generateC,
    generateCuda,
) where

import           Aoewif.Internal.Codegen.Cpu  (CSource, cFunctionName,
                                               cSourceText, generateC)
import           Aoewif.Internal.Codegen.Cuda (CudaSource, cudaKernelName,
                                               cudaSourceText, generateCuda)
