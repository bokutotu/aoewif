{-# LANGUAGE BlockArguments   #-}
{-# LANGUAGE OverloadedLabels #-}

module CudaSpec (spec) where

import           Aoewif.Cuda
import qualified Aoewif.Cuda.IR as IR
import           Test.Hspec     (Spec, describe, it, shouldBe)

spec :: Spec
spec =
    describe "cuda" $
        it "builds a 1D in-place add program" $ do
            let actual = cuda do
                    n <- kernelArg #N
                    a <- readTensor #A
                    c <- readWriteTensor #C

                    kernel1D #add (ceilDiv n 128) 128 \block ->
                        parFor_ 128 \parallel -> do
                            let index = linearIndex 128 block parallel

                            when_ (inBounds index n) do
                                store c index $
                                    add
                                        (load c index)
                                        (load a index)

                expectedIndex =
                    IR.AddIndex
                        (IR.MultiplyIndex IR.BlockIndex (IR.Literal 128))
                        IR.ParallelIndex

            actual
                `shouldBe` IR.Program
                    [ IR.ScalarArg (IR.Name "N") IR.USize
                    , IR.TensorArg (IR.Name "A") IR.ReadOnly IR.F32
                    , IR.TensorArg (IR.Name "C") IR.ReadWrite IR.F32
                    ]
                    ( IR.Kernel
                        (IR.Name "add")
                        ( IR.Launch
                            ( IR.Grid1D
                                (IR.CeilDiv (IR.Size (IR.Name "N")) (IR.Literal 128))
                            )
                            (IR.Block1D 128)
                        )
                        [ IR.Parallel
                            ( IR.Parallel1D
                                (IR.Literal 128)
                                [ IR.When
                                    (IR.InBounds expectedIndex (IR.Size (IR.Name "N")))
                                    [ IR.Store
                                        (IR.Name "C")
                                        expectedIndex
                                        ( IR.Add
                                            (IR.Load (IR.Name "C") expectedIndex)
                                            (IR.Load (IR.Name "A") expectedIndex)
                                        )
                                    ]
                                ]
                            )
                        ]
                    )
