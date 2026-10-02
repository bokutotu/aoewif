module Aoewif.Target.Cuda.Expr (
    BinaryOp (..),
    Expr (..),
    UnaryOp (..),
)
where

import           Aoewif.Target.Cuda.BlockDim  (BlockDim)
import           Aoewif.Target.Cuda.BlockIdx  (BlockIdx)
import           Aoewif.Target.Cuda.GridDim   (GridDim)
import           Aoewif.Target.Cuda.Name      (Name)
import           Aoewif.Target.Cuda.Render    (Render (..))
import           Aoewif.Target.Cuda.ThreadIdx (ThreadIdx)
import           Aoewif.Target.Cuda.Type      (Type)
import           Data.List                    (intercalate)

data UnaryOp
    = StaticCast Type
    | ReinterpretCast Type
    | LogicalNot
    | BitComplement
    deriving stock (Eq, Show)

data BinaryOp
    = Assign
    | Add
    | Subtract
    | Multiply
    | Divide
    | Modulo
    | Equal
    | NotEqual
    | LessThan
    | LessThanOrEqual
    | GreaterThanOrEqual
    | LogicalAnd
    | LogicalOr
    | ShiftLeft
    | ShiftRight
    | BitAnd
    | BitXor
    | BitOr
    deriving stock (Eq, Show)

data Expr
    = Var Name
    | IntLit Integer
    | FloatLit Float
    | BoolLit Bool
    | ThreadIdx ThreadIdx
    | BlockIdx BlockIdx
    | BlockDim BlockDim
    | GridDim GridDim
    | Unary UnaryOp Expr
    | Binary BinaryOp Expr Expr
    | Conditional Expr Expr Expr
    | Subscript Expr Expr
    | Call Expr [Expr]
    deriving stock (Eq, Show)

instance Render BinaryOp where
    render Assign             = "="
    render Add                = "+"
    render Subtract           = "-"
    render Multiply           = "*"
    render Divide             = "/"
    render Modulo             = "%"
    render Equal              = "=="
    render NotEqual           = "!="
    render LessThan           = "<"
    render LessThanOrEqual    = "<="
    render GreaterThanOrEqual = ">="
    render LogicalAnd         = "&&"
    render LogicalOr          = "||"
    render ShiftLeft          = "<<"
    render ShiftRight         = ">>"
    render BitAnd             = "&"
    render BitXor             = "^"
    render BitOr              = "|"

instance Render Expr where
    render expr =
        case expr of
            Var name ->
                render name
            IntLit value ->
                show value
            FloatLit value ->
                renderFloatLit value
            BoolLit value ->
                if value then "true" else "false"
            ThreadIdx index ->
                render index
            BlockIdx index ->
                render index
            BlockDim dimension ->
                render dimension
            GridDim dimension ->
                render dimension
            Unary (StaticCast targetType) operand ->
                "static_cast<"
                    ++ render targetType
                    ++ ">("
                    ++ render operand
                    ++ ")"
            Unary (ReinterpretCast targetType) operand ->
                "(*reinterpret_cast<"
                    ++ render targetType
                    ++ "*>(&"
                    ++ render operand
                    ++ "))"
            Unary LogicalNot operand ->
                "(!"
                    ++ render operand
                    ++ ")"
            Unary BitComplement operand ->
                "(~"
                    ++ render operand
                    ++ ")"
            Binary operator lhs rhs ->
                "("
                    ++ render lhs
                    ++ " "
                    ++ render operator
                    ++ " "
                    ++ render rhs
                    ++ ")"
            Conditional condition consequent alternative ->
                "("
                    ++ render condition
                    ++ " ? "
                    ++ render consequent
                    ++ " : "
                    ++ render alternative
                    ++ ")"
            Subscript value index ->
                render value
                    ++ "["
                    ++ render index
                    ++ "]"
            Call function arguments ->
                render function
                    ++ "("
                    ++ intercalate ", " (map render arguments)
                    ++ ")"

renderFloatLit :: Float -> String
renderFloatLit value
    | isNaN value = "NAN"
    | isInfinite value && value > 0 = "INFINITY"
    | isInfinite value = "-INFINITY"
    | otherwise = show value ++ "f"
