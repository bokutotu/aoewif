module Aoewif.Target.Cuda.LdMatrix (
    LdMatrix (..),
    LdMatrixSpace (..),
    MatrixCount (..),
    MatrixLayout (..),
)
where

import           Aoewif.Target.Cuda.AsmOperand (AsmInput, AsmOutput)
import           Aoewif.Target.Cuda.Render     (Render (..), indent)
import           Data.List                     (intercalate)

data MatrixCount = X1 | X2 | X4
    deriving stock (Eq, Show)

data MatrixLayout = Normal | Transposed
    deriving stock (Eq, Show)

data LdMatrixSpace = Generic | Shared | SharedCta
    deriving stock (Eq, Show)

data LdMatrix = M8N8B16
    { matrixCount  :: MatrixCount
    , layout       :: MatrixLayout
    , addressSpace :: LdMatrixSpace
    , destinations :: [AsmOutput]
    , source       :: AsmInput
    }
    deriving stock (Eq, Show)

instance Render MatrixCount where
    render X1 = ".x1"
    render X2 = ".x2"
    render X4 = ".x4"

instance Render MatrixLayout where
    render Normal     = ""
    render Transposed = ".trans"

instance Render LdMatrixSpace where
    render Generic   = ""
    render Shared    = ".shared"
    render SharedCta = ".shared::cta"

instance Render LdMatrix where
    render (M8N8B16 count matrixLayout space outputs address) =
        "asm volatile(\n"
            ++ indent
                ( show instruction
                    ++ "\n"
                    ++ ": "
                    ++ intercalate ", " (map render outputs)
                    ++ "\n"
                    ++ ": "
                    ++ render address
                    ++ "\n"
                    ++ ": \"memory\"\n"
                )
            ++ ")"
      where
        register index = "%" ++ show index
        instruction =
            "ldmatrix.sync.aligned.m8n8"
                ++ render count
                ++ render matrixLayout
                ++ render space
                ++ ".b16 {"
                ++ intercalate ", " (map register [0 .. length outputs - 1])
                ++ "}, ["
                ++ register (length outputs)
                ++ "];\n"
