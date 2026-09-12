module Aoewif.Target.Cuda.Sm90.Wgmma.Instruction (
    WgmmaAccumulatorType (..),
    WgmmaDescriptor (..),
    WgmmaFloatN (..),
    WgmmaFragment (..),
    WgmmaHalfOperands (..),
    WgmmaMma (..),
    WgmmaScale (..),
    WgmmaTranspose (..),
) where

import           Aoewif.Target.Cuda.Syntax (Expr)

data WgmmaFloatN
    = WgmmaFloatN8
    | WgmmaFloatN16
    | WgmmaFloatN24
    | WgmmaFloatN32
    | WgmmaFloatN40
    | WgmmaFloatN48
    | WgmmaFloatN56
    | WgmmaFloatN64
    | WgmmaFloatN72
    | WgmmaFloatN80
    | WgmmaFloatN88
    | WgmmaFloatN96
    | WgmmaFloatN104
    | WgmmaFloatN112
    | WgmmaFloatN120
    | WgmmaFloatN128
    | WgmmaFloatN136
    | WgmmaFloatN144
    | WgmmaFloatN152
    | WgmmaFloatN160
    | WgmmaFloatN168
    | WgmmaFloatN176
    | WgmmaFloatN184
    | WgmmaFloatN192
    | WgmmaFloatN200
    | WgmmaFloatN208
    | WgmmaFloatN216
    | WgmmaFloatN224
    | WgmmaFloatN232
    | WgmmaFloatN240
    | WgmmaFloatN248
    | WgmmaFloatN256
    deriving stock (Eq, Show)

data WgmmaAccumulatorType
    = WgmmaAccumulatorF16
    | WgmmaAccumulatorF32
    deriving stock (Eq, Show)

data WgmmaScale
    = WgmmaScaleOne
    | WgmmaScaleNegativeOne
    deriving stock (Eq, Show)

data WgmmaTranspose
    = WgmmaNotTransposed
    | WgmmaTransposed
    deriving stock (Eq, Show)

newtype WgmmaDescriptor = WgmmaDescriptor Expr
    deriving stock (Eq, Show)

newtype WgmmaFragment = WgmmaFragment
    { wgmmaFragmentRegisters :: [Expr]
    }
    deriving stock (Eq, Show)

data WgmmaHalfOperands
    = WgmmaHalfSharedOperands
        WgmmaDescriptor
        WgmmaDescriptor
        WgmmaTranspose
        WgmmaTranspose
    | WgmmaHalfRegisterOperands
        WgmmaFragment
        WgmmaDescriptor
        WgmmaTranspose
    deriving stock (Eq, Show)

data WgmmaMma
    = WgmmaF16
        WgmmaFloatN
        WgmmaAccumulatorType
        WgmmaFragment
        WgmmaHalfOperands
        Expr
        WgmmaScale
        WgmmaScale
    | WgmmaBF16
        WgmmaFloatN
        WgmmaFragment
        WgmmaHalfOperands
        Expr
        WgmmaScale
        WgmmaScale
    deriving stock (Eq, Show)
