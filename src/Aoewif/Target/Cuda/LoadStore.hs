{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE QuasiQuotes           #-}

module Aoewif.Target.Cuda.LoadStore (
    MemorySpace (..),
    VectorWidth (..),
    MemoryType (..),
    VectorLoad (..),
    VectorStore (..),
)
where

import           Aoewif.Target.Cuda.AsmOperand (AsmInput, AsmOutput)
import           Aoewif.Target.Cuda.Render     (Render (..))
import           Aoewif.Target.Template        (template)
import           Data.List                     (intercalate)

data MemorySpace = Shared | Global
    deriving stock (Eq, Show)

data VectorWidth = V2 | V4
    deriving stock (Eq, Show)

data MemoryType = B32 | U32 | F32
    deriving stock (Eq, Show)

data VectorLoad = VectorLoad
    { addressSpace :: MemorySpace
    , vectorWidth  :: VectorWidth
    , elementType  :: MemoryType
    , destinations :: [AsmOutput]
    , source       :: AsmInput
    }
    deriving stock (Eq, Show)

data VectorStore = VectorStore
    { addressSpace :: MemorySpace
    , vectorWidth  :: VectorWidth
    , elementType  :: MemoryType
    , destination  :: AsmInput
    , sources      :: [AsmInput]
    }
    deriving stock (Eq, Show)

instance Render MemorySpace where
    render Shared = ".shared"
    render Global = ".global"

instance Render VectorWidth where
    render V2 = ".v2"
    render V4 = ".v4"

instance Render MemoryType where
    render B32 = ".b32"
    render U32 = ".u32"
    render F32 = ".f32"

instance Render VectorLoad where
    render (VectorLoad space width ty outputs address) =
        [template|
        asm volatile(
            "ld${spaceText}${widthText}${typeText} {${registersText}}, [${addressRegister}];\n"
            : ${outputsText}
            : ${addressText}
            : "memory"
        )
        |]
      where
        spaceText = render space
        widthText = render width
        typeText = render ty
        registersText = intercalate ", " ["%" ++ show i | i <- [0 .. length outputs - 1]]
        addressRegister = "%" ++ show (length outputs)
        outputsText = intercalate ", " (map render outputs)
        addressText = render address

instance Render VectorStore where
    render (VectorStore space width ty address values) =
        [template|
        asm volatile(
            "st${spaceText}${widthText}${typeText} [%0], {${registersText}};\n"
            :
            : ${inputsText}
            : "memory"
        )
        |]
      where
        spaceText = render space
        widthText = render width
        typeText = render ty
        registersText = intercalate ", " ["%" ++ show i | i <- [1 .. length values]]
        inputsText = intercalate ", " (map render (address : values))
