module Aoewif.Target.Cuda.AsmOperand (
    AsmInput (..),
    AsmOutput (..),
    OutputMode (..),
    RegisterConstraint (..),
    input,
    readWrite,
    writeOnly,
)
where

import {-# SOURCE #-}           Aoewif.Target.Cuda.Expr   (Expr)
import                          Aoewif.Target.Cuda.Render (Render (..))

data RegisterConstraint
    = RegU16
    | RegU32
    | RegU64
    | RegU128
    | RegF32
    | RegF64
    deriving stock (Eq, Show)

data OutputMode = WriteOnly | ReadWrite
    deriving stock (Eq, Show)

data AsmOutput = AsmOutput
    { outputMode       :: OutputMode
    , outputConstraint :: RegisterConstraint
    , outputValue      :: Expr
    }
    deriving stock (Eq, Show)

data AsmInput = AsmInput
    { inputConstraint :: RegisterConstraint
    , inputValue      :: Expr
    }
    deriving stock (Eq, Show)

input :: RegisterConstraint -> Expr -> AsmInput
input = AsmInput

writeOnly :: RegisterConstraint -> Expr -> AsmOutput
writeOnly = AsmOutput WriteOnly

readWrite :: RegisterConstraint -> Expr -> AsmOutput
readWrite = AsmOutput ReadWrite

instance Render RegisterConstraint where
    render RegU16  = "h"
    render RegU32  = "r"
    render RegU64  = "l"
    render RegU128 = "q"
    render RegF32  = "f"
    render RegF64  = "d"

instance Render OutputMode where
    render WriteOnly = "="
    render ReadWrite = "+"

instance Render AsmOutput where
    render (AsmOutput mode constraint value) =
        show (render mode ++ render constraint) ++ "(" ++ render value ++ ")"

instance Render AsmInput where
    render (AsmInput constraint value) =
        show (render constraint) ++ "(" ++ render value ++ ")"
