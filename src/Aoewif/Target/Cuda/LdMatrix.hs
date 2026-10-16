{-# LANGUAGE QuasiQuotes #-}

module Aoewif.Target.Cuda.LdMatrix (
    LdMatrix (..),
    LdMatrixSpace (..),
    MatrixCount (..),
    MatrixLayout (..),
)
where

import           Aoewif.Target.Cuda.AsmOperand (AsmInput, AsmOutput)
import           Aoewif.Target.Cuda.Render     (Render (..))
import           Aoewif.Target.Template        (template)
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
        [template|
        asm volatile(
            "ldmatrix.sync.aligned.m8n8${countText}${layoutText}${spaceText}.b16 {${registersText}}, [${addressRegister}];\n"
            : ${outputsText}
            : ${addressText}
            : "memory"
        )
        |]
      where
        register index = "%" ++ show index
        outputsText = intercalate ", " (map render outputs)
        addressText = render address
        countText = render count
        layoutText = render matrixLayout
        spaceText = render space
        registersText = intercalate ", " (map register [0 .. length outputs - 1])
        addressRegister = register (length outputs)
