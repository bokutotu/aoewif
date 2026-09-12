{-# LANGUAGE QuasiQuotes #-}
{-# OPTIONS_GHC -Wno-orphans #-}

module Aoewif.Target.Cuda.Sm80.Render () where

import           Aoewif.Target.Cuda.Codegen          (indent, renderExpr)
import           Aoewif.Target.Cuda.Sm80.Instruction (CpAsyncShape (..),
                                                      LdMatrixForm (..),
                                                      LdMatrixMode (..),
                                                      MmaShape (..),
                                                      Sm80Op (..),
                                                      ldMatrixRegisterCount)
import           Aoewif.Target.Cuda.Syntax           (Expr)
import           Aoewif.Target.Cuda.TensorCoreOp     (RenderOp (..))
import           Data.List                           (intercalate)
import           Data.String.Interpolate             (i)

instance RenderOp Sm80Op where
    renderOp indentation op =
        case op of
            Mma shape aRegisters bRegisters dRegisters ->
                renderMma indentation shape aRegisters bRegisters dRegisters
            LdMatrix mode form registers address ->
                renderLdMatrix indentation mode form registers address
            MovMatrix register ->
                renderMovMatrix indentation register
            CpAsync shape sourceSize destination source ->
                renderCpAsync indentation shape sourceSize destination source
            CommitGroup ->
                [i|#{padding}asm volatile("cp.async.commit_group;");
|]
            WaitGroup Nothing ->
                [i|#{padding}asm volatile("cp.async.wait_all;");
|]
            WaitGroup (Just groups) ->
                [i|#{padding}asm volatile("cp.async.wait_group #{groups};");
|]
      where
        padding = indent indentation

data MmaInfo = MmaInfo
    { mmaInfoAsmTag :: String
    , mmaInfoARegs  :: Int
    , mmaInfoBRegs  :: Int
    , mmaInfoDRegs  :: Int
    }

mmaInfo :: MmaShape -> MmaInfo
mmaInfo M8N8K4F16    = MmaInfo "m8n8k4.row.col.f32.f16.f16.f32" 2 2 8
mmaInfo M16N8K8F16   = MmaInfo "m16n8k8.row.col.f32.f16.f16.f32" 2 1 4
mmaInfo M16N8K16F16  = MmaInfo "m16n8k16.row.col.f32.f16.f16.f32" 4 2 4
mmaInfo M16N8K8BF16  = MmaInfo "m16n8k8.row.col.f32.bf16.bf16.f32" 2 1 4
mmaInfo M16N8K16BF16 = MmaInfo "m16n8k16.row.col.f32.bf16.bf16.f32" 4 2 4

renderMma :: Int -> MmaShape -> [Expr] -> [Expr] -> [Expr] -> String
renderMma indentation shape aRegisters bRegisters dRegisters =
    [i|#{padding}asm volatile("mma.sync.aligned.#{asmTag} {#{dOperands}}, {#{aOperands}}, {#{bOperands}}, {#{dOperands}};"
#{operandPadding}: #{constraints "+f" dRegisters}
#{operandPadding}: #{constraints "r" (aRegisters ++ bRegisters)}
#{padding});
|]
  where
    MmaInfo
        { mmaInfoAsmTag = asmTag
        , mmaInfoARegs = aCount
        , mmaInfoBRegs = bCount
        , mmaInfoDRegs = dCount
        } = mmaInfo shape
    dOperands = placeholders 0 dCount
    aOperands = placeholders dCount aCount
    bOperands = placeholders (dCount + aCount) bCount
    padding = indent indentation
    operandPadding = indent (indentation + 1)

renderLdMatrix :: Int -> LdMatrixMode -> LdMatrixForm -> [Expr] -> Expr -> String
renderLdMatrix indentation mode form registers address =
    [i|#{padding}asm volatile("ldmatrix.sync.aligned.m8n8.#{formTag}#{modeTag}.shared.b16 {#{outputs}}, [%#{registerCount}];"
#{operandPadding}: #{constraints "=r" registers}
#{operandPadding}: "l"(#{sharedAddress address})
#{padding});
|]
  where
    formTag = ldMatrixFormTag form
    modeTag = ldMatrixModeTag mode
    registerCount = ldMatrixRegisterCount form
    outputs = placeholders 0 registerCount
    padding = indent indentation
    operandPadding = indent (indentation + 1)

renderMovMatrix :: Int -> Expr -> String
renderMovMatrix indentation register =
    [i|#{padding}asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %0;"
#{operandPadding}: "+r"(#{renderExpr register})
#{padding});
|]
  where
    padding = indent indentation
    operandPadding = indent (indentation + 1)

renderCpAsync :: Int -> CpAsyncShape -> Maybe Expr -> Expr -> Expr -> String
renderCpAsync indentation shape sourceSize destination source =
    case sourceSize of
        Nothing ->
            [i|#{padding}asm volatile("cp.async.#{cache}.shared.global [%0], [%1], #{size};"
#{operandPadding}:: "l"(#{sharedAddress destination}), "l"(&#{renderExpr source})
#{padding});
|]
        Just sourceSizeExpr ->
            [i|#{padding}asm volatile("cp.async.#{cache}.shared.global [%0], [%1], #{size}, %2;"
#{operandPadding}:: "l"(#{sharedAddress destination}), "l"(&#{renderExpr source}), "r"(#{renderExpr sourceSizeExpr})
#{padding});
|]
  where
    (cache, size) = cpAsyncInfo shape
    padding = indent indentation
    operandPadding = indent (indentation + 1)

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

ldMatrixFormTag :: LdMatrixForm -> String
ldMatrixFormTag LdX1 = "x1"
ldMatrixFormTag LdX2 = "x2"
ldMatrixFormTag LdX4 = "x4"

ldMatrixModeTag :: LdMatrixMode -> String
ldMatrixModeTag LdMatrixNormal    = ""
ldMatrixModeTag LdMatrixTranspose = ".trans"

cpAsyncInfo :: CpAsyncShape -> (String, Int)
cpAsyncInfo CacheAll4     = ("ca", 4)
cpAsyncInfo CacheAll8     = ("ca", 8)
cpAsyncInfo CacheAll16    = ("ca", 16)
cpAsyncInfo CacheGlobal16 = ("cg", 16)
