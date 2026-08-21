module Aoewif.Target.Cuda.Sm80 (
    Sm80Op (..),
    CpAsyncShape (..),
    Fragment,
    LdMatrixForm (..),
    LdMatrixMode (..),
    MmaShape (..),
    commitGroup,
    cpAsync,
    declareFragment,
    ldMatrix,
    mma,
    movMatrix,
    waitGroup,
    zeroFragment,
) where

import           Aoewif.Target.Cuda.Sm80.DSL
import           Aoewif.Target.Cuda.Sm80.Instruction
import           Aoewif.Target.Cuda.Sm80.Render      ()
