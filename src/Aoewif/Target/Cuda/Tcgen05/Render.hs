{-# LANGUAGE QuasiQuotes #-}
{-# OPTIONS_GHC -Wno-orphans #-}

module Aoewif.Target.Cuda.Tcgen05.Render () where

import           Aoewif.Target.Cuda.Codegen             (indent, renderExpr)
import           Aoewif.Target.Cuda.Tcgen05.Instruction (Tcgen05Op (..))
import           Aoewif.Target.Cuda.TensorCoreOp        (RenderOp (..))
import           Data.String.Interpolate                (i)

instance RenderOp Tcgen05Op where
    renderOp indentation operation =
        case operation of
            Tcgen05Alloc destination columns ->
                [i|#{padding}cuda::ptx::tcgen05_alloc(cuda::ptx::cta_group_1, #{renderExpr destination}, #{renderExpr columns});
|]
            Tcgen05FenceAfterThreadSync ->
                [i|#{padding}cuda::ptx::tcgen05_fence_after_thread_sync();
|]
            Tcgen05Mma accumulator descriptorA descriptorB instructionDescriptor disableOutputLane enableInputD ->
                [i|#{padding}cuda::ptx::tcgen05_mma(cuda::ptx::kind_f16, cuda::ptx::cta_group_1, #{renderExpr accumulator}, #{renderExpr descriptorA}, #{renderExpr descriptorB}, #{renderExpr instructionDescriptor}, #{renderExpr disableOutputLane}, #{renderExpr enableInputD});
|]
      where
        padding = indent indentation
