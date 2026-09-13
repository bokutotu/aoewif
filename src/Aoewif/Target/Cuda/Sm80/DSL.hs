module Aoewif.Target.Cuda.Sm80.DSL (
    commitGroup,
    cpAsync,
    ldMatrix,
    mma,
    movMatrix,
    waitGroup,
) where

import           Aoewif.Target.Cuda.DSL              (Block, Type (U32), emit,
                                                      int, zeroArray, (!))
import           Aoewif.Target.Cuda.Sm80.Instruction (CpAsyncShape,
                                                      Fragment (..),
                                                      LdMatrixForm,
                                                      LdMatrixMode, MmaShape,
                                                      Sm80Op (..),
                                                      ldMatrixRegisterCount)
import           Aoewif.Target.Cuda.Sm80.Render      ()
import           Aoewif.Target.Cuda.Syntax           (Expr, Stmt (Op))
import           Aoewif.Target.Cuda.TensorCoreOp     (TensorCoreOp (TensorCoreOp))

ldMatrix :: String -> LdMatrixForm -> LdMatrixMode -> Expr -> Block Fragment
ldMatrix name form mode address = do
    let registerCount = ldMatrixRegisterCount form
    array <- zeroArray U32 name [int (fromIntegral registerCount)]
    let registers = [array ! int (fromIntegral index) | index <- [0 .. registerCount - 1]]
    emit (Op (TensorCoreOp (LdMatrix mode form registers address)))
    pure (Fragment registers)

movMatrix :: Fragment -> Block ()
movMatrix (Fragment registers) =
    mapM_ (emit . Op . TensorCoreOp . MovMatrix) registers

mma :: MmaShape -> Fragment -> Fragment -> Fragment -> Block ()
mma shape (Fragment aRegisters) (Fragment bRegisters) (Fragment dRegisters) =
    emit
        ( Op
            ( TensorCoreOp
                ( Mma
                    shape
                    aRegisters
                    bRegisters
                    dRegisters
                )
            )
        )

cpAsync :: CpAsyncShape -> Maybe Expr -> Expr -> Expr -> Block ()
cpAsync shape sourceSize destination source = emit (Op (TensorCoreOp (CpAsync shape sourceSize destination source)))

commitGroup :: Block ()
commitGroup = emit (Op (TensorCoreOp CommitGroup))

waitGroup :: Maybe Int -> Block ()
waitGroup = emit . Op . TensorCoreOp . WaitGroup
