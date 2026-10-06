module Aoewif.Target.Cuda.Stmt (
    Stmt (..),
)
where

import           Aoewif.Target.Cuda.Alignment (Alignment)
import           Aoewif.Target.Cuda.Expr      (Expr)
import           Aoewif.Target.Cuda.Name      (Name)
import           Aoewif.Target.Cuda.Render    (Render (..), indent)
import           Aoewif.Target.Cuda.Type      (Type)

data Stmt
    = VarDecl Type Name (Maybe Expr)
    | Array Type Name [Expr]
    | SharedDecl Alignment Type Name Expr
    | SyncThreads
    | ExprStmt Expr
    | If Expr [Stmt] (Maybe [Stmt])
    | For (Maybe Stmt) Expr (Maybe Expr) [Stmt]
    deriving stock (Show)

instance Render Stmt where
    render stmt =
        case stmt of
            VarDecl variableType name initializer ->
                render variableType
                    ++ " "
                    ++ render name
                    ++ renderInitializer initializer
                    ++ ";\n"
            Array elementType name extents ->
                render elementType
                    ++ " "
                    ++ render name
                    ++ concatMap (\extent -> "[" ++ render extent ++ "]") extents
                    ++ " = {};\n"
            SharedDecl alignment elementType name extent ->
                "__shared__ "
                    ++ render alignment
                    ++ render elementType
                    ++ " "
                    ++ render name
                    ++ "["
                    ++ render extent
                    ++ "];\n"
            SyncThreads ->
                "__syncthreads();\n"
            ExprStmt expr ->
                render expr ++ ";\n"
            If condition body alternative ->
                "if ("
                    ++ render condition
                    ++ ") {\n"
                    ++ indent (concatMap render body)
                    ++ renderAlternative alternative
            For initializer condition update body ->
                "for ("
                    ++ maybe "" renderForInitStmt initializer
                    ++ "; "
                    ++ render condition
                    ++ "; "
                    ++ maybe "" render update
                    ++ ") {\n"
                    ++ indent (concatMap render body)
                    ++ "}\n"

renderInitializer :: Maybe Expr -> String
renderInitializer Nothing     = ""
renderInitializer (Just expr) = " = " ++ render expr

renderAlternative :: Maybe [Stmt] -> String
renderAlternative Nothing = "}\n"
renderAlternative (Just body) =
    "} else {\n"
        ++ indent (concatMap render body)
        ++ "}\n"

renderForInitStmt :: Stmt -> String
renderForInitStmt (VarDecl variableType name initializer) =
    render variableType ++ " " ++ render name ++ renderInitializer initializer
renderForInitStmt (ExprStmt expr) = render expr
renderForInitStmt statement = render statement
