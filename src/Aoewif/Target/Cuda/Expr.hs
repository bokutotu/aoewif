module Aoewif.Target.Cuda.Expr (
    Axis (..),
    BinaryOp (..),
    Expr (..),
    UnaryOp (..),
)
where

import           Aoewif.Target.Cuda.Mma    (Mma)
import           Aoewif.Target.Cuda.Name   (Name)
import           Aoewif.Target.Cuda.Render (Render (..))
import           Aoewif.Target.Cuda.Type   (Type)
import           Data.List                 (intercalate)

data Axis = X | Y | Z
    deriving stock (Eq, Show)

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
    | ThreadIdx Axis
    | BlockIdx Axis
    | BlockDim Axis
    | GridDim Axis
    | Unary UnaryOp Expr
    | Binary BinaryOp Expr Expr
    | Conditional Expr Expr Expr
    | Subscript Expr Expr
    | Call Expr [Expr]
    | MmaExpr Mma
    deriving stock (Eq, Show)

instance Render Axis where
    render X = "x"
    render Y = "y"
    render Z = "z"

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
            ThreadIdx axis ->
                "threadIdx." ++ render axis
            BlockIdx axis ->
                "blockIdx." ++ render axis
            BlockDim axis ->
                "blockDim." ++ render axis
            GridDim axis ->
                "gridDim." ++ render axis
            Unary (StaticCast targetType) operand ->
                "static_cast<" ++ render targetType ++ ">(" ++ render operand ++ ")"
            Unary (ReinterpretCast targetType) operand ->
                "(*reinterpret_cast<" ++ render targetType ++ "*>(&" ++ render operand ++ "))"
            Unary LogicalNot operand ->
                "(!" ++ render operand ++ ")"
            Unary BitComplement operand ->
                "(~" ++ render operand ++ ")"
            Binary operator lhs rhs ->
                "(" ++ render lhs ++ " " ++ render operator ++ " " ++ render rhs ++ ")"
            Conditional condition consequent alternative ->
                "(" ++ render condition ++ " ? " ++ render consequent ++ " : " ++ render alternative ++ ")"
            Subscript value index ->
                render value ++ "[" ++ render index ++ "]"
            Call function arguments ->
                render function ++ "(" ++ intercalate ", " (map render arguments) ++ ")"
            MmaExpr instruction ->
                render instruction

renderFloatLit :: Float -> String
renderFloatLit value
    | isNaN value = "NAN"
    | isInfinite value && value > 0 = "INFINITY"
    | isInfinite value = "-INFINITY"
    | otherwise = show value ++ "f"
