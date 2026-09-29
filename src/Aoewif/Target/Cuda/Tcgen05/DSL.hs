module Aoewif.Target.Cuda.Tcgen05.DSL (
    tcgen05Alloc,
    tcgen05FenceAfterThreadSync,
    tcgen05Mma,
) where

import           Aoewif.Target.Cuda.DSL                 (Block, emit)
import           Aoewif.Target.Cuda.Syntax              (Expr, Stmt (Op))
import           Aoewif.Target.Cuda.Tcgen05.Instruction (Tcgen05Op (..))
import           Aoewif.Target.Cuda.Tcgen05.Render      ()
import           Aoewif.Target.Cuda.TensorCoreOp        (TensorCoreOp (TensorCoreOp))

tcgen05Alloc :: Expr -> Expr -> Block ()
tcgen05Alloc destination columns =
    emit (Op (TensorCoreOp (Tcgen05Alloc destination columns)))

tcgen05FenceAfterThreadSync :: Block ()
tcgen05FenceAfterThreadSync =
    emit (Op (TensorCoreOp Tcgen05FenceAfterThreadSync))

tcgen05Mma :: Expr -> Expr -> Expr -> Expr -> Expr -> Expr -> Block ()
tcgen05Mma accumulator descriptorA descriptorB instructionDescriptor disableOutputLane enableInputD =
    emit
        ( Op
            ( TensorCoreOp
                ( Tcgen05Mma
                    accumulator
                    descriptorA
                    descriptorB
                    instructionDescriptor
                    disableOutputLane
                    enableInputD
                )
            )
        )
