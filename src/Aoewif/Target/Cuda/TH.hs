{-# LANGUAGE TemplateHaskellQuotes #-}

module Aoewif.Target.Cuda.TH (cuda) where

import qualified Aoewif.Target.Cuda.DSL as DSL
import           Data.Data              (Data, cast, gmapM)
import           Language.Haskell.TH

cuda :: Q Exp -> Q Exp
cuda expression = expression >>= transform

transform :: (Data a) => a -> Q a
transform value = do
    descended <- gmapM transform value
    case cast descended of
        Just statement -> do
            rewritten <- rewriteStmt statement
            maybe (fail "cuda: internal error: rewritten statement type mismatch") pure (cast rewritten)
        Nothing -> case cast descended of
            Just expression ->
                maybe (fail "cuda: internal error: rewritten expression type mismatch") pure (cast (rewriteExpr expression))
            Nothing -> pure descended

rewriteStmt :: Stmt -> Q Stmt
rewriteStmt statement@(BindS pattern_ expression) =
    case expression of
        AppE (VarE parameter) type_ | parameter == 'DSL.parameter -> withName $ \name -> apply parameter [type_, name]
        AppE (VarE declare) type_ | declare == 'DSL.declare -> withName $ \name -> apply declare [type_, name]
        AppE (AppE (VarE define) type_) value | define == 'DSL.define -> withName $ \name -> apply define [type_, name, value]
        AppE (AppE (VarE zeroArray) type_) dimensions | zeroArray == 'DSL.zeroArray -> withName $ \name -> apply zeroArray [type_, name, dimensions]
        AppE (AppE (AppE (VarE shared) alignment) type_) size | shared == 'DSL.shared -> withName $ \name -> apply shared [alignment, type_, name, size]
        _ -> pure statement
  where
    withName build =
        case pattern_ of
            VarP name -> pure $ BindS pattern_ (build (LitE (StringL (nameBase name))))
            _ -> fail "cuda: a declaration without an explicit name requires a variable binding"

    apply function = foldl AppE (VarE function)
rewriteStmt statement = pure statement

rewriteExpr :: Exp -> Exp
rewriteExpr expression = case expression of
    InfixE (Just loop) (VarE operator) (Just body)
        | operator == '($) -> InfixE (Just (nameLoop loop body)) (VarE operator) (Just body)
    AppE loop body -> AppE (nameLoop loop body) body
    _ -> expression
  where
    nameLoop loop (LamE [VarP name] _)
        | AppE (AppE (AppE (AppE (VarE for) type_) initial) step) condition <- loop
        , for == 'DSL.for_ =
            foldl
                AppE
                (VarE for)
                [ type_
                , LitE (StringL (nameBase name))
                , initial
                , step
                , condition
                ]
    nameLoop loop _ = loop
