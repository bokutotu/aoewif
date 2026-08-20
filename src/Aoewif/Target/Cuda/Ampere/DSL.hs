module Aoewif.Target.Cuda.Ampere.DSL (
    commitGroup,
    cpAsync,
    declareFragment,
    ldMatrix,
    mma,
    movMatrix,
    waitGroup,
    zeroFragment,
) where

import           Aoewif.Target.Cuda.Ampere.Instruction (AmpereOp (..), CacheOp,
                                                        CpAsyncSize,
                                                        Fragment (..),
                                                        LdMatrixForm,
                                                        LdMatrixMode, MmaShape,
                                                        ldMatrixRegisterCount)
import           Aoewif.Target.Cuda.Ampere.Render      ()
import           Aoewif.Target.Cuda.DSL                (Block, Type (U32),
                                                        declare, emit, int,
                                                        (.=))
import           Aoewif.Target.Cuda.Syntax             (Expr, Stmt (Op))
import           Aoewif.Target.Cuda.TensorCoreOp       (TensorCoreOp (TensorCoreOp))

declareFragment :: String -> Int -> Block Fragment
declareFragment prefix registerCount =
    Fragment <$> mapM declareRegister [0 .. registerCount - 1]
  where
    declareRegister index =
        declare U32 (prefix ++ show index)

zeroFragment :: Fragment -> Block ()
zeroFragment =
    mapM_ (.= int 0) . fragmentRegisters

ldMatrix :: String -> LdMatrixForm -> LdMatrixMode -> Expr -> Block Fragment
ldMatrix prefix form mode address = do
    fragment <- declareFragment prefix (ldMatrixRegisterCount form)
    emit (Op (TensorCoreOp (LdMatrix mode form (fragmentRegisters fragment) address)))
    pure fragment

movMatrix :: Fragment -> Block ()
movMatrix (Fragment registers) =
    mapM_
        (emit . Op . TensorCoreOp . MovMatrix)
        registers

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

cpAsync :: CacheOp -> CpAsyncSize -> Maybe Expr -> Expr -> Expr -> Block ()
cpAsync cache size sourceSize destination source =
    emit (Op (TensorCoreOp (CpAsync cache size sourceSize destination source)))

commitGroup :: Block ()
commitGroup =
    emit (Op (TensorCoreOp CommitGroup))

waitGroup :: Maybe Int -> Block ()
waitGroup =
    emit . Op . TensorCoreOp . WaitGroup
