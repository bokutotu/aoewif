module Aoewif.Target.Cuda.Ampere.DSL (
    commitGroup,
    cpAsync,
    declareFragment,
    ldMatrix,
    mma,
    waitGroup,
    zeroFragment,
) where

import           Aoewif.Target.Cuda.Ampere.Instruction (AmpereOp (..), CacheOp,
                                                        CpAsyncSize,
                                                        Fragment (..),
                                                        LdMatrixForm, MmaShape)
import           Aoewif.Target.Cuda.Ampere.Render      (ldMatrixInfo)
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

ldMatrix :: String -> LdMatrixForm -> Expr -> Block Fragment
ldMatrix prefix form address = do
    fragment <- declareFragment prefix (snd (ldMatrixInfo form))
    emit (Op (TensorCoreOp (LdMatrix form (fragmentRegisters fragment) address)))
    pure fragment

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

-- With Just sourceSize, fewer than cp-size bytes are copied from the source
-- and the rest of the 16B-aligned destination is zero-filled, so a partial
-- k-tile can be loaded without reading out of bounds; only the .ca variant
-- exists in PTX.
cpAsync :: CacheOp -> CpAsyncSize -> Maybe Expr -> Expr -> Expr -> Block ()
cpAsync cache size sourceSize destination source =
    emit (Op (TensorCoreOp (CpAsync cache size sourceSize destination source)))

commitGroup :: Block ()
commitGroup =
    emit (Op (TensorCoreOp CommitGroup))

waitGroup :: Maybe Int -> Block ()
waitGroup =
    emit . Op . TensorCoreOp . WaitGroup
