module Aoewif.Target.Cuda.DSL.Expr (
    Expr,
    bitcast,
    blockDimX,
    blockDimY,
    blockDimZ,
    blockIdxX,
    blockIdxY,
    blockIdxZ,
    bool,
    call,
    cast,
    complement,
    float,
    gridDimX,
    gridDimY,
    gridDimZ,
    ifElse,
    int,
    not_,
    threadIdxX,
    threadIdxY,
    threadIdxZ,
    var,
    (!),
    (.*),
    (./),
    (.+),
    (.-),
    (.%),
    (.==),
    (./=),
    (.<),
    (.<=),
    (.>=),
    (.&&),
    (.||),
    (.&.),
    (.^),
    (.|.),
    (.<<),
    (.>>),
)
where

import           Aoewif.Target.Cuda.Expr (Axis (..), BinaryOp (..), Expr (..),
                                          UnaryOp (..))
import           Aoewif.Target.Cuda.Name (Name (..))
import           Aoewif.Target.Cuda.Type (Type)

var :: String -> Expr
var = Var . Name

int :: Integer -> Expr
int = IntLit

float :: Float -> Expr
float = FloatLit

bool :: Bool -> Expr
bool = BoolLit

cast :: Type -> Expr -> Expr
cast targetType = Unary (StaticCast targetType)

bitcast :: Type -> Expr -> Expr
bitcast targetType = Unary (ReinterpretCast targetType)

not_ :: Expr -> Expr
not_ = Unary LogicalNot

complement :: Expr -> Expr
complement = Unary BitComplement

ifElse :: Expr -> Expr -> Expr -> Expr
ifElse = Conditional

call :: Expr -> [Expr] -> Expr
call = Call

threadIdxX, threadIdxY, threadIdxZ :: Expr
threadIdxX = ThreadIdx X
threadIdxY = ThreadIdx Y
threadIdxZ = ThreadIdx Z

blockIdxX, blockIdxY, blockIdxZ :: Expr
blockIdxX = BlockIdx X
blockIdxY = BlockIdx Y
blockIdxZ = BlockIdx Z

blockDimX, blockDimY, blockDimZ :: Expr
blockDimX = BlockDim X
blockDimY = BlockDim Y
blockDimZ = BlockDim Z

gridDimX, gridDimY, gridDimZ :: Expr
gridDimX = GridDim X
gridDimY = GridDim Y
gridDimZ = GridDim Z

infixl 9 !

(!) :: Expr -> Expr -> Expr
(!) = Subscript

infixl 8 .<<, .>>

(.<<) :: Expr -> Expr -> Expr
(.<<) = Binary ShiftLeft

(.>>) :: Expr -> Expr -> Expr
(.>>) = Binary ShiftRight

infixl 7 .*

(.*) :: Expr -> Expr -> Expr
(.*) = Binary Multiply

infixl 7 ./

(./) :: Expr -> Expr -> Expr
(./) = Binary Divide

infixl 7 .%

(.%) :: Expr -> Expr -> Expr
(.%) = Binary Modulo

infixl 6 .+

(.+) :: Expr -> Expr -> Expr
(.+) = Binary Add

infixl 6 .-

(.-) :: Expr -> Expr -> Expr
(.-) = Binary Subtract

infix 4 .==, ./=, .<, .<=, .>=

(.==) :: Expr -> Expr -> Expr
(.==) = Binary Equal

(./=) :: Expr -> Expr -> Expr
(./=) = Binary NotEqual

(.<) :: Expr -> Expr -> Expr
(.<) = Binary LessThan

(.<=) :: Expr -> Expr -> Expr
(.<=) = Binary LessThanOrEqual

(.>=) :: Expr -> Expr -> Expr
(.>=) = Binary GreaterThanOrEqual

infixl 7 .&.

(.&.) :: Expr -> Expr -> Expr
(.&.) = Binary BitAnd

infixl 6 .^

(.^) :: Expr -> Expr -> Expr
(.^) = Binary BitXor

infixl 5 .|.

(.|.) :: Expr -> Expr -> Expr
(.|.) = Binary BitOr

infixr 3 .&&

(.&&) :: Expr -> Expr -> Expr
(.&&) = Binary LogicalAnd

infixr 2 .||

(.||) :: Expr -> Expr -> Expr
(.||) = Binary LogicalOr
