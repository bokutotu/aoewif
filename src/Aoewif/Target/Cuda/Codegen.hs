module Aoewif.Target.Cuda.Codegen (
    Config (..),
    Include (..),
    generate,
    generateWith,
    indent,
    renderExpr,
)
where

import qualified Aoewif.Target.Cuda.Syntax       as Syntax
import           Aoewif.Target.Cuda.TensorCoreOp (RenderOp (renderOp))
import           Data.List                       (intercalate)

data Include
    = CudaFp16Header
    | CudaBf16Header
    deriving stock (Eq, Show)

newtype Config = Config
    { includes :: [Include]
    }
    deriving stock (Eq, Show)

generate :: Syntax.Kernel -> String
generate = generateWith (Config [])

generateWith :: Config -> Syntax.Kernel -> String
generateWith config (Syntax.Kernel name parameters body) =
    renderIncludes (includes config)
        ++ "extern \"C\" __global__ void "
        ++ renderName name
        ++ "("
        ++ intercalate ", " (map renderParameter parameters)
        ++ ") {\n"
        ++ renderStmts 1 body
        ++ "}\n"

renderIncludes :: [Include] -> String
renderIncludes [] = ""
renderIncludes configuredIncludes =
    unlines (map renderInclude configuredIncludes) ++ "\n"

renderInclude :: Include -> String
renderInclude CudaFp16Header = "#include <cuda_fp16.h>"
renderInclude CudaBf16Header = "#include <cuda_bf16.h>"

renderParameter :: Syntax.Parameter -> String
renderParameter (Syntax.Parameter parameterType name) =
    renderType parameterType ++ " " ++ renderName name

renderType :: Syntax.Type -> String
renderType Syntax.Bool = "bool"
renderType Syntax.U32 = "uint32_t"
renderType Syntax.USize = "size_t"
renderType Syntax.F16 = "__half"
renderType Syntax.BF16 = "__nv_bfloat16"
renderType Syntax.F32 = "float"
renderType (Syntax.Const valueType) =
    renderType valueType ++ " const"
renderType (Syntax.Pointer pointeeType) =
    renderType pointeeType ++ "*"

renderStmts :: Int -> [Syntax.Stmt] -> String
renderStmts indentation =
    concatMap (renderStmt indentation)

renderStmt :: Int -> Syntax.Stmt -> String
renderStmt indentation stmt =
    case stmt of
        Syntax.VarDecl variableType name initializer ->
            indent indentation
                ++ renderType variableType
                ++ " "
                ++ renderName name
                ++ renderInitializer initializer
                ++ ";\n"
        Syntax.SharedDecl elementType name extent ->
            indent indentation
                ++ "__shared__ "
                ++ renderType elementType
                ++ " "
                ++ renderName name
                ++ "["
                ++ renderExpr extent
                ++ "];\n"
        Syntax.ExprStmt expr ->
            indent indentation
                ++ renderExpr expr
                ++ ";\n"
        Syntax.If condition body alternative ->
            indent indentation
                ++ "if ("
                ++ renderExpr condition
                ++ ") {\n"
                ++ renderStmts (indentation + 1) body
                ++ renderAlternative indentation alternative
        Syntax.Op op ->
            renderOp indentation op

renderInitializer :: Maybe Syntax.Expr -> String
renderInitializer Nothing = ""
renderInitializer (Just expr) =
    " = " ++ renderExpr expr

renderAlternative :: Int -> Maybe [Syntax.Stmt] -> String
renderAlternative indentation Nothing =
    indent indentation ++ "}\n"
renderAlternative indentation (Just body) =
    indent indentation
        ++ "} else {\n"
        ++ renderStmts (indentation + 1) body
        ++ indent indentation
        ++ "}\n"

renderExpr :: Syntax.Expr -> String
renderExpr expr =
    case expr of
        Syntax.Var name ->
            renderName name
        Syntax.IntLit value ->
            show value
        Syntax.FloatLit value ->
            renderFloatLit value
        Syntax.ThreadIdx index ->
            renderThreadIdx index
        Syntax.BlockIdx index ->
            renderBlockIdx index
        Syntax.BlockDim dimension ->
            renderBlockDim dimension
        Syntax.GridDim dimension ->
            renderGridDim dimension
        Syntax.Unary (Syntax.StaticCast targetType) operand ->
            "static_cast<"
                ++ renderType targetType
                ++ ">("
                ++ renderExpr operand
                ++ ")"
        Syntax.Binary operator lhs rhs ->
            "("
                ++ renderExpr lhs
                ++ " "
                ++ renderBinaryOp operator
                ++ " "
                ++ renderExpr rhs
                ++ ")"
        Syntax.Subscript value index ->
            renderExpr value
                ++ "["
                ++ renderExpr index
                ++ "]"
        Syntax.Call function arguments ->
            renderExpr function
                ++ "("
                ++ intercalate ", " (map renderExpr arguments)
                ++ ")"

renderBinaryOp :: Syntax.BinaryOp -> String
renderBinaryOp Syntax.Assign   = "="
renderBinaryOp Syntax.Add      = "+"
renderBinaryOp Syntax.Multiply = "*"
renderBinaryOp Syntax.LessThan = "<"

renderFloatLit :: Float -> String
renderFloatLit value
    | isNaN value = "NAN"
    | isInfinite value && value > 0 = "INFINITY"
    | isInfinite value = "-INFINITY"
    | otherwise = show value ++ "f"

renderThreadIdx :: Syntax.ThreadIdx -> String
renderThreadIdx Syntax.ThreadIdxX = "threadIdx.x"
renderThreadIdx Syntax.ThreadIdxY = "threadIdx.y"
renderThreadIdx Syntax.ThreadIdxZ = "threadIdx.z"

renderBlockIdx :: Syntax.BlockIdx -> String
renderBlockIdx Syntax.BlockIdxX = "blockIdx.x"
renderBlockIdx Syntax.BlockIdxY = "blockIdx.y"
renderBlockIdx Syntax.BlockIdxZ = "blockIdx.z"

renderBlockDim :: Syntax.BlockDim -> String
renderBlockDim Syntax.BlockDimX = "blockDim.x"
renderBlockDim Syntax.BlockDimY = "blockDim.y"
renderBlockDim Syntax.BlockDimZ = "blockDim.z"

renderGridDim :: Syntax.GridDim -> String
renderGridDim Syntax.GridDimX = "gridDim.x"
renderGridDim Syntax.GridDimY = "gridDim.y"
renderGridDim Syntax.GridDimZ = "gridDim.z"

renderName :: Syntax.Name -> String
renderName (Syntax.Name name) = name

indent :: Int -> String
indent level = replicate (level * 4) ' '
