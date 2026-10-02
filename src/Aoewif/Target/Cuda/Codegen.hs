module Aoewif.Target.Cuda.Codegen (
    Config (..),
    Include (..),
    generate,
    generateWith,
)
where

import           Aoewif.Target.Cuda.Kernel (Kernel)
import           Aoewif.Target.Cuda.Render (Render (..))

data Include
    = CudaFp16Header
    | CudaBf16Header
    deriving stock (Eq, Show)

instance Render Include where
    render CudaFp16Header = "#include <cuda_fp16.h>"
    render CudaBf16Header = "#include <cuda_bf16.h>"

newtype Config = Config
    { includes :: [Include]
    }
    deriving stock (Eq, Show)

generate :: Kernel -> String
generate = generateWith (Config [])

generateWith :: Config -> Kernel -> String
generateWith config kernel =
    renderIncludes (includes config) ++ render kernel

renderIncludes :: [Include] -> String
renderIncludes configuredIncludes =
    unlines (["#include <stdint.h>", "#include <cuda/ptx>"] ++ map render configuredIncludes) ++ "\n"
