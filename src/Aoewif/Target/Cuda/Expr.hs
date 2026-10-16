{-# LANGUAGE QuasiQuotes #-}

module Aoewif.Target.Cuda.Expr (
    Axis (..),
    BinaryOp (..),
    Expr (..),
    UnaryOp (..),
)
where

import           Aoewif.Target.Cuda.CpAsync  (CpAsync)
import           Aoewif.Target.Cuda.LdMatrix (LdMatrix)
import           Aoewif.Target.Cuda.Mma      (Mma)
import           Aoewif.Target.Cuda.Name     (Name)
import           Aoewif.Target.Cuda.Render   (Render (..))
import           Aoewif.Target.Cuda.Type     (Type)
import           Aoewif.Target.Template      (template)
import           Data.List                   (intercalate)

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
    | LdMatrixExpr LdMatrix
    | CpAsyncExpr CpAsync
    | CpAsyncCommitGroup
    | CpAsyncWaitGroup Integer
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
                let typeText = render targetType
                    operandText = render operand
                 in [template|static_cast<${typeText}>(${operandText})|]
            Unary (ReinterpretCast targetType) operand ->
                let typeText = render targetType
                    operandText = render operand
                 in [template|(*reinterpret_cast<${typeText}*>(&${operandText}))|]
            Unary LogicalNot operand ->
                let operandText = render operand
                 in [template|(!${operandText})|]
            Unary BitComplement operand ->
                let operandText = render operand
                 in [template|(~${operandText})|]
            Binary operator lhs rhs ->
                let lhsText = render lhs
                    operatorText = render operator
                    rhsText = render rhs
                 in [template|(${lhsText} ${operatorText} ${rhsText})|]
            Conditional condition consequent alternative ->
                let conditionText = render condition
                    consequentText = render consequent
                    alternativeText = render alternative
                 in [template|(${conditionText} ? ${consequentText} : ${alternativeText})|]
            Subscript value index ->
                let valueText = render value
                    indexText = render index
                 in [template|${valueText}[${indexText}]|]
            Call function arguments ->
                let functionText = render function
                    argumentsText = intercalate ", " (map render arguments)
                 in [template|${functionText}(${argumentsText})|]
            MmaExpr instruction ->
                render instruction
            LdMatrixExpr instruction ->
                render instruction
            CpAsyncExpr instruction ->
                render instruction
            CpAsyncCommitGroup ->
                "asm volatile(\"cp.async.commit_group;\\n\" ::: \"memory\")"
            CpAsyncWaitGroup groups ->
                let groupCount = show groups
                 in [template|
                    asm volatile(
                        "cp.async.wait_group ${groupCount};\n"
                        ::: "memory"
                    )
                    |]

renderFloatLit :: Float -> String
renderFloatLit value
    | isNaN value = "NAN"
    | isInfinite value && value > 0 = "INFINITY"
    | isInfinite value = "-INFINITY"
    | otherwise = show value ++ "f"
