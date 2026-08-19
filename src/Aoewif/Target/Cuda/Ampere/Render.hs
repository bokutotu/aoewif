{-# OPTIONS_GHC -Wno-orphans #-}

module Aoewif.Target.Cuda.Ampere.Render (
    ldMatrixInfo,
) where

import           Aoewif.Target.Cuda.Ampere.Instruction (AmpereOp (..),
                                                        CacheOp (..),
                                                        CpAsyncSize (..),
                                                        LdMatrixForm (..),
                                                        MmaShape (..))
import           Aoewif.Target.Cuda.Codegen            (indent, renderExpr)
import           Aoewif.Target.Cuda.Syntax             (Expr)
import           Aoewif.Target.Cuda.TensorCoreOp       (RenderOp (..))
import           Data.List                             (intercalate)

-- The extension socket: AmpereOp is defined in Instruction and RenderOp in
-- TensorCoreOp, so the instance is intentionally orphaned here where the asm
-- rendering lives.

instance RenderOp AmpereOp where
    renderOp indentation op =
        case op of
            Mma shape aRegisters bRegisters dRegisters ->
                renderMma indentation shape aRegisters bRegisters dRegisters
            LdMatrix form registers address ->
                renderLdMatrix indentation form registers address
            CpAsync cache size destination source ->
                renderCpAsync indentation cache size destination source
            CommitGroup ->
                asmLine indentation "cp.async.commit_group;"
            WaitGroup groups ->
                asmLine indentation (waitGroupInstruction groups)

data MmaInfo = MmaInfo
    { mmaInfoAsmTag :: String
    , mmaInfoARegs  :: Int
    , mmaInfoBRegs  :: Int
    , mmaInfoDRegs  :: Int
    }

mmaInfo :: MmaShape -> MmaInfo
mmaInfo M16N8K8F16 =
    MmaInfo "m16n8k8.row.col.f32.f16.f16.f32" 4 2 4
mmaInfo M16N8K16Tf32 =
    MmaInfo "m16n8k16.row.col.f32.tf32.tf32.f32" 4 2 4
mmaInfo M16N8K16Bf16 =
    MmaInfo "m16n8k16.row.col.f32.bf16.bf16.f32" 4 2 4
mmaInfo M8N8K4F64 =
    MmaInfo "m8n8k4.row.col.f64.f64.f64.f64" 1 1 2

renderMma :: Int -> MmaShape -> [Expr] -> [Expr] -> [Expr] -> String
renderMma indentation shape aRegisters bRegisters dRegisters =
    unlines
        [ asmOpen indentation ("mma.sync.aligned." ++ asmTag ++ " " ++ operands ++ ";")
        , indent (indentation + 1) ++ ": " ++ constraints "+r" dRegisters
        , indent (indentation + 1) ++ ": " ++ constraints "r" (aRegisters ++ bRegisters)
        , indent indentation ++ ");"
        ]
  where
    MmaInfo
        { mmaInfoAsmTag = asmTag
        , mmaInfoARegs = aCount
        , mmaInfoBRegs = bCount
        , mmaInfoDRegs = dCount
        } = mmaInfo shape
    operands =
        "{"
            ++ placeholders 0 dCount
            ++ "}, "
            ++ "{"
            ++ placeholders dCount aCount
            ++ "}, "
            ++ "{"
            ++ placeholders (dCount + aCount) bCount
            ++ "}, "
            ++ "{"
            ++ placeholders 0 dCount
            ++ "}"

renderLdMatrix :: Int -> LdMatrixForm -> [Expr] -> Expr -> String
renderLdMatrix indentation form registers address =
    unlines
        [ asmOpen
            indentation
            ( "ldmatrix.sync.aligned.m8n8."
                ++ formTag
                ++ ".shared.b16 {"
                ++ placeholders 0 registerCount
                ++ "}, ["
                ++ "%"
                ++ show registerCount
                ++ "];"
            )
        , indent (indentation + 1) ++ ": " ++ constraints "=r" registers
        , indent (indentation + 1) ++ ": \"r\"(" ++ sharedAddress address ++ ")"
        , indent indentation ++ ");"
        ]
  where
    (formTag, registerCount) = ldMatrixInfo form

renderCpAsync :: Int -> CacheOp -> CpAsyncSize -> Expr -> Expr -> String
renderCpAsync indentation cache size destination source =
    unlines
        [ asmOpen
            indentation
            ( "cp.async."
                ++ cacheTag cache
                ++ ".shared.global [%0], [%1], "
                ++ show (bytes size)
                ++ ";"
            )
        , indent (indentation + 1)
            ++ ":: \"r\"("
            ++ sharedAddress destination
            ++ "), \"l\"(&"
            ++ renderExpr source
            ++ ")"
        , indent indentation ++ ");"
        ]

asmOpen :: Int -> String -> String
asmOpen indentation instruction =
    indent indentation ++ "asm volatile(\"" ++ instruction ++ "\""

asmLine :: Int -> String -> String
asmLine indentation instruction =
    indent indentation ++ "asm volatile(\"" ++ instruction ++ "\");\n"

constraints :: String -> [Expr] -> String
constraints constraint =
    intercalate ", " . map (registerConstraint constraint)

registerConstraint :: String -> Expr -> String
registerConstraint constraint register =
    "\"" ++ constraint ++ "\"(" ++ renderExpr register ++ ")"

placeholders :: Int -> Int -> String
placeholders first count =
    intercalate "," ["%" ++ show index | index <- [first .. first + count - 1]]

sharedAddress :: Expr -> String
sharedAddress address =
    "__cvta_generic_to_shared(&" ++ renderExpr address ++ ")"

ldMatrixInfo :: LdMatrixForm -> (String, Int)
ldMatrixInfo LdX1 = ("x1", 1)
ldMatrixInfo LdX2 = ("x2", 2)
ldMatrixInfo LdX4 = ("x4", 4)

cacheTag :: CacheOp -> String
cacheTag CacheAll    = "ca"
cacheTag CacheGlobal = "cg"

bytes :: CpAsyncSize -> Int
bytes Bytes4  = 4
bytes Bytes8  = 8
bytes Bytes16 = 16

waitGroupInstruction :: Maybe Int -> String
waitGroupInstruction Nothing = "cp.async.wait_all;"
waitGroupInstruction (Just groups) = "cp.async.wait_group " ++ show groups ++ ";"
