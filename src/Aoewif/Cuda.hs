{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE KindSignatures        #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables   #-}

module Aoewif.Cuda (
    Cuda,
    Element,
    Guard,
    Index,
    Kernel,
    KernelBody,
    Label,
    ParallelBody,
    Program,
    Tensor,
    add,
    ceilDiv,
    cuda,
    inBounds,
    kernel1D,
    kernelArg,
    linearIndex,
    load,
    parFor_,
    readTensor,
    readWriteTensor,
    store,
    when_,
)
where

import           Aoewif.Cuda.IR
import           Data.Proxy           (Proxy (..))
import           GHC.OverloadedLabels (IsLabel (..))
import           GHC.TypeLits         (KnownSymbol, symbolVal)

newtype Builder output value = Builder ([output], value)
    deriving newtype (Functor, Applicative, Monad)

type Cuda = Builder Argument

type KernelBody = Builder KernelOp

type ParallelBody = Builder Statement

newtype Label = Label Name

instance (KnownSymbol name) => IsLabel name Label where
    fromLabel = Label (Name (symbolVal (Proxy :: Proxy name)))

newtype Tensor (access :: Access) = Tensor Name

cuda :: Cuda Kernel -> Program
cuda (Builder (arguments, kernel)) = Program arguments kernel

kernelArg :: Label -> Cuda Index
kernelArg (Label name) = Builder ([ScalarArg name USize], Size name)

readTensor :: Label -> Cuda (Tensor 'ReadOnly)
readTensor (Label name) = Builder ([TensorArg name ReadOnly F32], Tensor name)

readWriteTensor :: Label -> Cuda (Tensor 'ReadWrite)
readWriteTensor (Label name) = Builder ([TensorArg name ReadWrite F32], Tensor name)

kernel1D :: Label -> Index -> Word -> (Index -> KernelBody ()) -> Cuda Kernel
kernel1D (Label name) blockCount threadCount buildBody =
    pure
        ( Kernel
            name
            (Launch (Grid1D blockCount) (Block1D threadCount))
            (outputs (buildBody BlockIndex))
        )

parFor_ :: Word -> (Index -> ParallelBody ()) -> KernelBody ()
parFor_ extent buildBody =
    emit
        ( Parallel
            (Parallel1D (Literal extent) (outputs (buildBody ParallelIndex)))
        )

ceilDiv :: Index -> Word -> Index
ceilDiv value divisor = CeilDiv value (Literal divisor)

linearIndex :: Word -> Index -> Index -> Index
linearIndex blockSize block =
    AddIndex (MultiplyIndex block (Literal blockSize))

inBounds :: Index -> Index -> Guard
inBounds = InBounds

load :: Tensor access -> Index -> Element
load (Tensor name) = Load name

add :: Element -> Element -> Element
add = Add

when_ :: Guard -> ParallelBody () -> ParallelBody ()
when_ guard body = emit (When guard (outputs body))

store :: Tensor 'ReadWrite -> Index -> Element -> ParallelBody ()
store (Tensor name) index value = emit (Store name index value)

emit :: output -> Builder output ()
emit output = Builder ([output], ())

outputs :: Builder output () -> [output]
outputs (Builder (result, ())) = result
