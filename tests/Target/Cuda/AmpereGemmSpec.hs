module Target.Cuda.AmpereGemmSpec (spec) where

import qualified Aoewif.Target.Cuda.Codegen as Codegen
import           Aoewif.Target.Cuda.DSL
import           Test.Hspec                 (Spec, describe, it, shouldBe)

spec :: Spec
spec = describe "reference kernels" $ do
    it "generates the Ampere MMA reference kernel" $ do
        expected <- readFile "reference_kernel/ampere_mma.cu"
        let generated = Codegen.generateWith (Codegen.Config [Codegen.CudaFp16Header]) $
                kernel "ampere_mma" $ do
                    matrixA <- parameter (Pointer (Const F16)) "matrix_a"
                    matrixB <- parameter (Pointer (Const F16)) "matrix_b"
                    matrixD <- parameter (Pointer F32) "matrix_d"
                    body $ do
                        groupId <- define (Const U32) "group_id" (threadIdxX ./ int 4)
                        threadInGroup <- define (Const U32) "thread_in_group" (threadIdxX .% int 4)
                        fragmentA <- zeroArray U32 "fragment_a" [int 4]
                        for_ U32 "packed_index" (int 0) (.+ int 1) (.< int 4) $ \packedIndex -> do
                            row <- define (Const U32) "row" (groupId .+ (packedIndex .% int 2) .* int 8)
                            column <- define (Const U32) "column" (threadInGroup .* int 2 .+ (packedIndex ./ int 2) .* int 8)
                            fragmentA
                                ! packedIndex
                                .= ( cast U32 (call (var "__half_as_ushort") [matrixA ! (row .* int 16 .+ column)])
                                        .|. shiftL
                                            (cast U32 (call (var "__half_as_ushort") [matrixA ! (row .* int 16 .+ column .+ int 1)]))
                                            (int 16)
                                   )
                        fragmentB <- zeroArray U32 "fragment_b" [int 2]
                        fragmentD <- zeroArray F32 "fragment_d" [int 4]
                        for_ U32 "packed_index" (int 0) (.+ int 1) (.< int 2) $ \packedIndex -> do
                            row <- define (Const U32) "row" (threadInGroup .* int 2 .+ packedIndex .* int 8)
                            column <- define (Const U32) "column" groupId
                            fragmentB
                                ! packedIndex
                                .= ( cast U32 (call (var "__half_as_ushort") [matrixB ! (row .* int 8 .+ column)])
                                        .|. shiftL
                                            (cast U32 (call (var "__half_as_ushort") [matrixB ! ((row .+ int 1) .* int 8 .+ column)]))
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
                        for_ U32 "element" (int 0) (.+ int 1) (.< int 4) $ \element -> do
                            row <- define (Const U32) "row" (groupId .+ (element ./ int 2) .* int 8)
                            column <- define (Const U32) "column" (threadInGroup .* int 2 .+ element .% int 2)
                            matrixD ! (row .* int 8 .+ column) .= fragmentD ! element
        generated `shouldBe` expected
