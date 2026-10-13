{-# LANGUAGE TemplateHaskell #-}

module Target.Cuda.AmpereGemmSpec (spec) where

import qualified Aoewif.Target.Cuda.Codegen  as Codegen
import           Aoewif.Target.Cuda.DSL
import           Aoewif.Target.Cuda.External (external)
import           Aoewif.Target.Cuda.TH       (cuda)
import           Test.Hspec                  (Spec, describe, it, shouldBe)

halfAsUshort :: Expr -> Expr
halfAsUshort = external "__half_as_ushort"

spec :: Spec
spec = describe "reference kernels" $ do
    it "generates the Ampere MMA reference kernel" $ do
        expected <- readFile "reference_kernel/ampere_mma.cu"
        let generated =
                Codegen.generateWith (Codegen.Config [Codegen.CudaFp16Header]) $
                    $( cuda
                        [|
                            kernel "ampere_mma" $ do
                                matrixA <- parameter (Pointer (Const F16))
                                matrixB <- parameter (Pointer (Const F16))
                                matrixD <- parameter (Pointer F32)
                                body $ do
                                    groupId <- define (Const U32) (threadIdxX ./ int 4)
                                    threadInGroup <- define (Const U32) (threadIdxX .% int 4)
                                    fragmentA <- zeroArray U32 [int 4]
                                    for_ U32 (int 0) (.+ int 1) (.< int 4) $ \packedIndex -> do
                                        row <- define (Const U32) (groupId .+ (packedIndex .% int 2) .* int 8)
                                        column <- define (Const U32) (threadInGroup .* int 2 .+ (packedIndex ./ int 2) .* int 8)
                                        fragmentA
                                            ! packedIndex
                                            .= ( cast U32 (halfAsUshort (matrixA ! (row .* int 16 .+ column)))
                                                    .|. shiftL
                                                        (cast U32 (halfAsUshort (matrixA ! (row .* int 16 .+ column .+ int 1))))
                                                        (int 16)
                                               )
                                    fragmentB <- zeroArray U32 [int 2]
                                    fragmentD <- zeroArray F32 [int 4]
                                    for_ U32 (int 0) (.+ int 1) (.< int 2) $ \packedIndex -> do
                                        row <- define (Const U32) (threadInGroup .* int 2 .+ packedIndex .* int 8)
                                        column <- define (Const U32) groupId
                                        fragmentB
                                            ! packedIndex
                                            .= ( cast U32 (halfAsUshort (matrixB ! (row .* int 8 .+ column)))
                                                    .|. shiftL
                                                        (cast U32 (halfAsUshort (matrixB ! ((row .+ int 1) .* int 8 .+ column))))
                                                        (int 16)
                                               )
                                    mma m16n8k16 $
                                        accumulateInto
                                            ( f32
                                                [ readWrite RegF32 (fragmentD ! int 0)
                                                , readWrite RegF32 (fragmentD ! int 1)
                                                , readWrite RegF32 (fragmentD ! int 2)
                                                , readWrite RegF32 (fragmentD ! int 3)
                                                ]
                                            )
                                            ( f16
                                                [ input RegU32 (fragmentA ! int 0)
                                                , input RegU32 (fragmentA ! int 1)
                                                , input RegU32 (fragmentA ! int 2)
                                                , input RegU32 (fragmentA ! int 3)
                                                ]
                                            )
                                            ( f16
                                                [ input RegU32 (fragmentB ! int 0)
                                                , input RegU32 (fragmentB ! int 1)
                                                ]
                                            )
                                    for_ U32 (int 0) (.+ int 1) (.< int 4) $ \element -> do
                                        row <- define (Const U32) (groupId .+ (element ./ int 2) .* int 8)
                                        column <- define (Const U32) (threadInGroup .* int 2 .+ element .% int 2)
                                        matrixD ! (row .* int 8 .+ column) .= fragmentD ! element
                            |]
                     )
        generated `shouldBe` expected
