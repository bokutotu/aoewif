module Aoewif.Cuda.Codegen (
    CudaSource,
    cudaSourceText,
    cudaKernelName,
    cudaLaunch,
    cudaSymbols,
    cudaBuffers,
    generateCuda,
) where

import qualified Aoewif.Cuda.IR as IR
import           Data.List      (intercalate)
import           Data.Maybe     (fromJust)
import           Data.Word      (Word64)

data CudaSource = CudaSource String IR.Kernel
    deriving stock (Eq, Show)

cudaSourceText :: CudaSource -> String
cudaSourceText (CudaSource sourceText _) = sourceText

cudaKernelName :: CudaSource -> String
cudaKernelName (CudaSource _ kernelIR) = sourceKernelName kernelIR

cudaLaunch :: CudaSource -> IR.Launch
cudaLaunch (CudaSource _ kernelIR) = IR.kernelLaunch kernelIR

cudaSymbols :: CudaSource -> [IR.Symbol]
cudaSymbols (CudaSource _ kernelIR) = IR.kernelSymbols kernelIR

cudaBuffers :: CudaSource -> [IR.BufferDecl]
cudaBuffers (CudaSource _ kernelIR) = IR.kernelBuffers kernelIR

generateCuda :: IR.Kernel -> CudaSource
generateCuda kernelIR = CudaSource (renderKernel kernelIR) kernelIR

renderKernel :: IR.Kernel -> String
renderKernel kernelIR =
    unlines
        [ "#include <cuda_runtime.h>"
        , "#include <math.h>"
        , "#include <stddef.h>"
        , ""
        ]
        ++ "extern \"C\" __global__ void "
        ++ sourceKernelName kernelIR
        ++ "("
        ++ intercalate ", " parameters
        ++ ") {\n"
        ++ renderStatements symbols buffers 1 (IR.kernelBody kernelIR)
        ++ "}\n"
  where
    symbols = IR.kernelSymbols kernelIR
    buffers = map (\buffer -> (IR.bufferId buffer, IR.bufferName buffer)) (IR.kernelBuffers kernelIR)
    parameters =
        map bufferParameter (IR.kernelBuffers kernelIR)
            ++ map symbolParameter (IR.kernelSymbols kernelIR)

bufferParameter :: IR.BufferDecl -> String
bufferParameter buffer = case IR.bufferAccess buffer of
    IR.ReadOnly  -> "const float* " ++ IR.bufferName buffer
    IR.ReadWrite -> "float* " ++ IR.bufferName buffer

symbolParameter :: IR.Symbol -> String
symbolParameter symbol = "size_t " ++ IR.symbolName symbol

renderStatements :: [IR.Symbol] -> [(IR.BufferId, String)] -> Int -> [IR.Statement] -> String
renderStatements symbols buffers indentation = concatMap (renderStatement symbols buffers indentation)

renderStatement :: [IR.Symbol] -> [(IR.BufferId, String)] -> Int -> IR.Statement -> String
renderStatement symbols buffers indentation statement = case statement of
    IR.LetF32 identifier expression ->
        indent indentation
            ++ "float "
            ++ valueName identifier
            ++ " = "
            ++ renderF32 symbols buffers expression
            ++ ";\n"
    IR.StoreF32 buffer address value ->
        indent indentation
            ++ bufferName buffers buffer
            ++ "["
            ++ renderIndex symbols address
            ++ "] = "
            ++ renderF32 symbols buffers value
            ++ ";\n"
    IR.SerialFor identifier extent body ->
        indent indentation
            ++ "for (size_t "
            ++ loopName identifier
            ++ " = 0; "
            ++ loopName identifier
            ++ " < "
            ++ renderExtent symbols extent
            ++ "; ++"
            ++ loopName identifier
            ++ ") {\n"
            ++ renderStatements symbols buffers (indentation + 1) body
            ++ indent indentation
            ++ "}\n"
    IR.IfThen predicate body ->
        indent indentation
            ++ "if ("
            ++ renderPredicate symbols predicate
            ++ ") {\n"
            ++ renderStatements symbols buffers (indentation + 1) body
            ++ indent indentation
            ++ "}\n"
    IR.AllocateShared declaration body ->
        indent indentation
            ++ "{\n"
            ++ indent (indentation + 1)
            ++ "__shared__ float "
            ++ IR.sharedName declaration
            ++ "["
            ++ show (staticElementCount (IR.sharedShape declaration))
            ++ "ull];\n"
            ++ renderStatements
                symbols
                ((IR.sharedBufferId declaration, IR.sharedName declaration) : buffers)
                (indentation + 1)
                body
            ++ indent indentation
            ++ "}\n"
    IR.SyncThreads -> indent indentation ++ "__syncthreads();\n"

renderF32 :: [IR.Symbol] -> [(IR.BufferId, String)] -> IR.F32Expr -> String
renderF32 symbols buffers expression = case expression of
    IR.F32Literal value -> floatLiteral value
    IR.F32Value identifier -> valueName identifier
    IR.LoadF32 buffer address ->
        bufferName buffers buffer ++ "[" ++ renderIndex symbols address ++ "]"
    IR.AddF32 lhs rhs -> binary "__fadd_rn" lhs rhs
    IR.SubF32 lhs rhs -> binary "__fsub_rn" lhs rhs
    IR.MulF32 lhs rhs -> binary "__fmul_rn" lhs rhs
    IR.DivF32 lhs rhs -> binary "__fdiv_rn" lhs rhs
  where
    binary function lhs rhs =
        function
            ++ "("
            ++ renderF32 symbols buffers lhs
            ++ ", "
            ++ renderF32 symbols buffers rhs
            ++ ")"

renderIndex :: [IR.Symbol] -> IR.IndexExpr -> String
renderIndex symbols expression = case expression of
    IR.BlockIndexX          -> "((size_t)blockIdx.x)"
    IR.ThreadIndexX         -> "((size_t)threadIdx.x)"
    IR.LoopIndex identifier -> loopName identifier
    IR.ConstantIndex value  -> show value ++ "ull"
    IR.ExtentIndex extent   -> renderExtent symbols extent
    IR.AddIndex lhs rhs     -> binary "+" lhs rhs
    IR.MulIndex lhs rhs     -> binary "*" lhs rhs
  where
    binary operator lhs rhs =
        "("
            ++ renderIndex symbols lhs
            ++ " "
            ++ operator
            ++ " "
            ++ renderIndex symbols rhs
            ++ ")"

renderPredicate :: [IR.Symbol] -> IR.Predicate -> String
renderPredicate symbols predicate = case predicate of
    IR.IndexLessThan lhs rhs ->
        renderIndex symbols lhs ++ " < " ++ renderIndex symbols rhs

renderExtent :: [IR.Symbol] -> IR.Extent -> String
renderExtent symbols extent = case extent of
    IR.StaticExtent value -> show value ++ "ull"
    IR.DynamicExtent identifier -> symbolName symbols identifier
    IR.CeilDivExtent dividend divisor ->
        "("
            ++ renderExtent symbols dividend
            ++ " / "
            ++ divisorText
            ++ " + ("
            ++ renderExtent symbols dividend
            ++ " % "
            ++ divisorText
            ++ " != 0ull))"
      where
        divisorText = show divisor ++ "ull"

floatLiteral :: Float -> String
floatLiteral value
    | isNaN value = "NAN"
    | isInfinite value && value > 0 = "INFINITY"
    | isInfinite value = "-INFINITY"
    | otherwise = show value ++ "f"

staticElementCount :: [Word64] -> Integer
staticElementCount = product . map toInteger

symbolName :: [IR.Symbol] -> IR.SymbolId -> String
symbolName symbols identifier = IR.symbolName (fromJust (lookup identifier symbolNames))
  where
    symbolNames = map (\symbol -> (IR.symbolId symbol, symbol)) symbols

bufferName :: [(IR.BufferId, String)] -> IR.BufferId -> String
bufferName buffers identifier = fromJust (lookup identifier buffers)

valueName :: IR.ValueId -> String
valueName (IR.ValueId identifier) = "value" ++ show identifier

loopName :: IR.LoopId -> String
loopName (IR.LoopId identifier) = "loop" ++ show identifier

sourceKernelName :: IR.Kernel -> String
sourceKernelName = IR.kernelName

indent :: Int -> String
indent level = replicate (level * 4) ' '
