{-# LANGUAGE QuasiQuotes #-}

module Aoewif.Target.Cuda.Stmt (
    Stmt (..),
)
where

import           Aoewif.Target.Cuda.Alignment (Alignment)
import           Aoewif.Target.Cuda.Expr      (Expr)
import           Aoewif.Target.Cuda.Name      (Name)
import           Aoewif.Target.Cuda.Render    (Render (..))
import           Aoewif.Target.Cuda.Type      (Type)
import           Aoewif.Target.Template       (template)

data Stmt
    = VarDecl Type Name (Maybe Expr)
    | Array Type Name [Expr]
    | SharedDecl Alignment Type Name Expr
    | SyncThreads
    | SyncWarp
    | ExprStmt Expr
    | If Expr [Stmt] (Maybe [Stmt])
    | For (Maybe Stmt) Expr (Maybe Expr) [Stmt]
    deriving stock (Show)

instance Render Stmt where
    render stmt =
        case stmt of
            VarDecl variableType name initializer ->
                let typeText = render variableType
                    nameText = render name
                    initializerText = renderInitializer initializer
                 in [template|${typeText} ${nameText}${initializerText};|] ++ "\n"
            Array elementType name extents ->
                let typeText = render elementType
                    nameText = render name
                    extentsText = concatMap (\extent -> let extentText = render extent in [template|[${extentText}]|]) extents
                 in [template|${typeText} ${nameText}${extentsText} = {};|] ++ "\n"
            SharedDecl alignment elementType name extent ->
                let alignmentText = render alignment
                    typeText = render elementType
                    nameText = render name
                    extentText = render extent
                 in [template|__shared__ ${alignmentText}${typeText} ${nameText}[${extentText}];|] ++ "\n"
            SyncThreads ->
                "__syncthreads();\n"
            SyncWarp ->
                "__syncwarp();\n"
            ExprStmt expr ->
                let expressionText = render expr
                 in [template|${expressionText};|] ++ "\n"
            If condition body alternative ->
                let conditionText = render condition
                    bodyText = concatMap render body
                    alternativeText = renderAlternative alternative
                 in [template|
                    if (${conditionText}) {
                        ${bodyText}
                    ${alternativeText}
                    |]
                        ++ "\n"
            For initializer condition update body ->
                let initializerText = maybe "" renderForInitStmt initializer
                    conditionText = render condition
                    updateText = maybe "" render update
                    bodyText = concatMap render body
                 in [template|
                    for (${initializerText}; ${conditionText}; ${updateText}) {
                        ${bodyText}
                    }
                    |]
                        ++ "\n"

renderInitializer :: Maybe Expr -> String
renderInitializer Nothing     = ""
renderInitializer (Just expr) = " = " ++ render expr

renderAlternative :: Maybe [Stmt] -> String
renderAlternative Nothing = "}\n"
renderAlternative (Just body) =
    [template|
    } else {
        ${bodyText}
    }
    |]
        ++ "\n"
  where
    bodyText = concatMap render body

renderForInitStmt :: Stmt -> String
renderForInitStmt (VarDecl variableType name initializer) =
    [template|${typeText} ${nameText}${initializerText}|]
  where
    typeText = render variableType
    nameText = render name
    initializerText = renderInitializer initializer
renderForInitStmt (ExprStmt expr) = render expr
renderForInitStmt statement = render statement
