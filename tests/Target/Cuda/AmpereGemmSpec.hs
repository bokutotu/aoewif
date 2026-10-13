{-# LANGUAGE TemplateHaskell #-}

module Target.Cuda.AmpereGemmSpec (spec) where

import qualified Aoewif.Target.Cuda.Codegen  as Codegen
import           Aoewif.Target.Cuda.DSL
import           Aoewif.Target.Cuda.External (external)
import           Aoewif.Target.Cuda.TH       (cuda)
import           Test.Hspec                  (Spec, describe, it, shouldBe)

genericToShared :: Expr -> Expr
genericToShared = external "__cvta_generic_to_shared"

spec :: Spec
spec = describe "reference kernels" $ do
    -- Row-major A[M, K], B[K, N], D[M, N]; D = A * B.
    -- M, N, K must be positive multiples of 32.
    -- Launch with grid (N / 8, M / 16) and block (32).
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
                                m <- parameter U32 "M"
                                n <- parameter U32 "N"
                                k <- parameter U32 "K"
                                let _ = m
                                body $ do
                                    lane <- define (Const U32) threadIdxX
                                    groupId <- define (Const U32) (lane ./ int 4)
                                    threadInGroup <- define (Const U32) (lane .% int 4)
                                    tileRow <- define (Const U32) (blockIdxY .* int 16)
                                    tileColumn <- define (Const U32) (blockIdxX .* int 8)
                                    sharedA <- shared Align16 F16 (int 16 .* int 16)
                                    sharedB <- shared Align16 F16 (int 16 .* int 8)
                                    fragmentD <- zeroArray F32 [int 4]
                                    -- A: top-left, bottom-left, top-right, bottom-right 8x8 matrices.
                                    addressA <-
                                        define
                                            (Const U32)
                                            (cast U32 (genericToShared (sharedA .+ ((lane .% int 16) .* int 16 .+ (lane ./ int 16) .* int 8))))
                                    -- B: top and bottom 8x8 matrices; upper lanes repeat valid addresses.
                                    addressB <-
                                        define
                                            (Const U32)
                                            (cast U32 (genericToShared (sharedB .+ ((lane .% int 16) .* int 8))))
                                    for_ U32 (int 0) (.+ int 16) (.< k) $ \tileK -> do
                                        for_ U32 lane (.+ int 32) (.< int 16 .* int 16) $ \index -> do
                                            row <- define (Const U32) (tileRow .+ index ./ int 16)
                                            column <- define (Const U32) (tileK .+ index .% int 16)
                                            sharedA ! index .= matrixA ! (cast USize row .* k .+ column)
                                        for_ U32 lane (.+ int 32) (.< int 16 .* int 8) $ \index -> do
                                            row <- define (Const U32) (tileK .+ index ./ int 8)
                                            column <- define (Const U32) (tileColumn .+ index .% int 8)
                                            sharedB ! index .= matrixB ! (cast USize row .* n .+ column)
                                        syncThreads
                                        fragmentA <- zeroArray U32 [int 4]
                                        fragmentB <- zeroArray U32 [int 2]
                                        ldmatrix $
                                            m8n8b16
                                                X4
                                                Normal
                                                Shared
                                                [ writeOnly RegU32 (fragmentA ! int 0)
                                                , writeOnly RegU32 (fragmentA ! int 1)
                                                , writeOnly RegU32 (fragmentA ! int 2)
                                                , writeOnly RegU32 (fragmentA ! int 3)
                                                ]
                                                (input RegU32 addressA)
                                        ldmatrix $
                                            m8n8b16
                                                X2
                                                Transposed
                                                Shared
                                                [ writeOnly RegU32 (fragmentB ! int 0)
                                                , writeOnly RegU32 (fragmentB ! int 1)
                                                ]
                                                (input RegU32 addressB)
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
                                        -- Finish shared-memory reads before storing the next tile.
                                        syncThreads
                                    for_ U32 (int 0) (.+ int 1) (.< int 4) $ \element -> do
                                        row <- define (Const U32) (tileRow .+ groupId .+ (element ./ int 2) .* int 8)
                                        column <- define (Const U32) (tileColumn .+ threadInGroup .* int 2 .+ element .% int 2)
                                        matrixD ! (cast USize row .* n .+ column) .= fragmentD ! element
                            |]
                     )
        generated `shouldBe` expected
