module Target.Cuda.AmpereSpec (spec) where

import           Aoewif.Target.Cuda.Ampere
import qualified Aoewif.Target.Cuda.Codegen as Codegen
import           Aoewif.Target.Cuda.DSL
import           Test.Hspec                 (Spec, describe, it, shouldBe)

spec :: Spec
spec =
    describe "Ampere tensor core instructions" $ do
        it "renders ldmatrix, fragment plumbing, and mma.m16n8k16.tf32" $ do
            Codegen.generate
                ( kernel "gemm" $ body $ do
                    smemA <- shared F32 "smemA" (int 64)
                    smemB <- shared F32 "smemB" (int 32)
                    aFragment <- ldMatrix "a" LdX4 (smemA ! int 16)
                    bFragment <- ldMatrix "b" LdX2 (smemB ! int 8)
                    accumulator <- declareFragment "c" 4
                    zeroFragment accumulator
                    mma M16N8K16Tf32 aFragment bFragment accumulator
                )
                `shouldBe` """
                           extern "C" __global__ void gemm() {
                               __shared__ float smemA[64];
                               __shared__ float smemB[32];
                               uint32_t a0;
                               uint32_t a1;
                               uint32_t a2;
                               uint32_t a3;
                               asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                   : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                                   : "r"(__cvta_generic_to_shared(&smemA[16]))
                               );
                               uint32_t b0;
                               uint32_t b1;
                               asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                                   : "=r"(b0), "=r"(b1)
                                   : "r"(__cvta_generic_to_shared(&smemB[8]))
                               );
                               uint32_t c0;
                               uint32_t c1;
                               uint32_t c2;
                               uint32_t c3;
                               (c0 = 0);
                               (c1 = 0);
                               (c2 = 0);
                               (c3 = 0);
                               asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                   : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                                   : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
                               );
                           }

                           """

        it "renders mma.m16n8k8.f16 from an fp16 shared tile" $ do
            Codegen.generateWith
                (Codegen.Config [Codegen.CudaFp16Header])
                ( kernel "f16gemm" $ body $ do
                    smemA <- shared F16 "smemA" (int 64)
                    smemB <- shared F16 "smemB" (int 32)
                    aFragment <- ldMatrix "a" LdX4 (smemA ! threadIdxX)
                    bFragment <- ldMatrix "b" LdX2 (smemB ! threadIdxX)
                    accumulator <- declareFragment "c" 4
                    mma M16N8K8F16 aFragment bFragment accumulator
                )
                `shouldBe` """
                           #include <cuda_fp16.h>

                           extern "C" __global__ void f16gemm() {
                               __shared__ __half smemA[64];
                               __shared__ __half smemB[32];
                               uint32_t a0;
                               uint32_t a1;
                               uint32_t a2;
                               uint32_t a3;
                               asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                   : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                                   : "r"(__cvta_generic_to_shared(&smemA[threadIdx.x]))
                               );
                               uint32_t b0;
                               uint32_t b1;
                               asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                                   : "=r"(b0), "=r"(b1)
                                   : "r"(__cvta_generic_to_shared(&smemB[threadIdx.x]))
                               );
                               uint32_t c0;
                               uint32_t c1;
                               uint32_t c2;
                               uint32_t c3;
                               asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                   : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                                   : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
                               );
                           }

                           """

        it "renders mma.m8n8k4.f64" $ do
            Codegen.generate
                ( kernel "f64gemm" $ body $ do
                    aFragment <- declareFragment "a" 1
                    bFragment <- declareFragment "b" 1
                    accumulator <- declareFragment "c" 2
                    mma M8N8K4F64 aFragment bFragment accumulator
                )
                `shouldBe` """
                           extern "C" __global__ void f64gemm() {
                               uint32_t a0;
                               uint32_t b0;
                               uint32_t c0;
                               uint32_t c1;
                               asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
                                   : "+r"(c0), "+r"(c1)
                                   : "r"(a0), "r"(b0)
                               );
                           }

                           """

        it "renders a cp.async K-loop" $ do
            Codegen.generate
                ( kernel "kloop" $ do
                    a <- parameter (Pointer (Const F32)) "A"
                    k <- parameter U32 "k"
                    body $ do
                        stage <- shared F32 "stage" (int 64)
                        for_ (define U32 "kk" (int 0)) (.< k) (\kk -> kk .+ int 16) $ \kk -> do
                            cpAsync CacheGlobal Bytes16 (stage ! int 0) (a ! kk)
                            commitGroup
                )
                `shouldBe` """
                           extern "C" __global__ void kloop(float const* A, uint32_t k) {
                               __shared__ float stage[64];
                               for (uint32_t kk = 0; (kk < k); (kk = (kk + 16))) {
                                   asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                                       :: "r"(__cvta_generic_to_shared(&stage[0])), "l"(&A[kk])
                                   );
                                   asm volatile("cp.async.commit_group;");
                               }
                           }

                           """

        it "renders a cp.async pipeline" $ do
            Codegen.generate
                ( kernel "pipeline" $ do
                    source <- parameter (Pointer (Const F32)) "source"
                    body $ do
                        stage <- shared F32 "stage" (int 64)
                        cpAsync CacheGlobal Bytes16 (stage ! int 0) (source ! int 0)
                        commitGroup
                        waitGroup (Just 2)
                        waitGroup Nothing
                )
                `shouldBe` """
                           extern "C" __global__ void pipeline(float const* source) {
                               __shared__ float stage[64];
                               asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                                   :: "r"(__cvta_generic_to_shared(&stage[0])), "l"(&source[0])
                               );
                               asm volatile("cp.async.commit_group;");
                               asm volatile("cp.async.wait_group 2;");
                               asm volatile("cp.async.wait_all;");
                           }

                           """
