module Aoewif.Target.Cuda.DSL (
    Alignment (..),
    AsmInput,
    AsmOutput,
    Block,
    Expr,
    Kernel,
    KernelBuilder,
    Mma,
    MmaOperation,
    Operand,
    RegisterConstraint (..),
    Type (..),
    accumulateInto,
    body,
    bool,
    bitcast,
    blockDimX,
    blockDimY,
    blockDimZ,
    blockIdxX,
    blockIdxY,
    blockIdxZ,
    call,
    call_,
    cast,
    complement,
    declare,
    define,
    emit,
    expr_,
    f16,
    f32,
    float,
    for_,
    gridDimX,
    gridDimY,
    gridDimZ,
    ifElse,
    ifElse_,
    if_,
    int,
    input,
    kernel,
    m16n8k16,
    mma,
    multiplyAddInto,
    not_,
    parameter,
    readWrite,
    shared,
    shiftL,
    shiftR,
    syncThreads,
    threadIdxX,
    threadIdxY,
    threadIdxZ,
    var,
    writeOnly,
    xor,
    zeroArray,
    (!),
    (.*),
    (./),
    (.+),
    (.-),
    (.%),
    (.=),
    (.==),
    (./=),
    (.<),
    (.<=),
    (.>=),
    (.&&),
    (.||),
    (.&.),
    (.|.),
)
where

import           Aoewif.Target.Cuda.Alignment  (Alignment (..))
import           Aoewif.Target.Cuda.AsmOperand (AsmInput, AsmOutput,
                                                RegisterConstraint (..), input,
                                                readWrite, writeOnly)
import           Aoewif.Target.Cuda.Expr       (Axis (..), BinaryOp (..),
                                                Expr (..), UnaryOp (..))
import           Aoewif.Target.Cuda.Kernel     (Kernel (..))
import           Aoewif.Target.Cuda.Mma        (Mma (..), MmaOperation (..),
                                                Operand (..))
import           Aoewif.Target.Cuda.Name       (Name (..))
import           Aoewif.Target.Cuda.Parameter  (Parameter (..))
import           Aoewif.Target.Cuda.Stmt       (Stmt (..))
import           Aoewif.Target.Cuda.Type       (Type (..))

newtype KernelBuilder value = KernelBuilder ([Parameter], value)
    deriving newtype (Functor, Applicative, Monad)

newtype Block value = Block ([Stmt], value)
    deriving newtype (Functor, Applicative, Monad)

kernel :: String -> KernelBuilder (Block ()) -> Kernel
kernel name (KernelBuilder (parameters, block)) =
    Kernel (Name name) parameters (blockStatements block)

parameter :: Type -> String -> KernelBuilder Expr
parameter parameterType text =
    KernelBuilder ([Parameter parameterType name], Var name)
  where
    name = Name text

body :: Block () -> KernelBuilder (Block ())
body = pure

declare :: Type -> String -> Block Expr
declare variableType text = do
    emit (VarDecl variableType name Nothing)
    pure (Var name)
  where
    name = Name text

zeroArray :: Type -> String -> [Expr] -> Block Expr
zeroArray elementType text extents = do
    emit (Array elementType name extents)
    pure (Var name)
  where
    name = Name text

define :: Type -> String -> Expr -> Block Expr
define variableType text initializer = do
    emit (VarDecl variableType name (Just initializer))
    pure (Var name)
  where
    name = Name text

shared :: Alignment -> Type -> String -> Expr -> Block Expr
shared alignment elementType text extent = do
    emit (SharedDecl alignment elementType name extent)
    pure (Var name)
  where
    name = Name text

expr_ :: Expr -> Block ()
expr_ = emit . ExprStmt

if_ :: Expr -> Block () -> Block ()
if_ condition consequent =
    emit (If condition (blockStatements consequent) Nothing)

ifElse_ :: Expr -> Block () -> Block () -> Block ()
ifElse_ condition consequent alternative =
    emit
        ( If
            condition
            (blockStatements consequent)
            (Just (blockStatements alternative))
        )

for_ :: Maybe Stmt -> Expr -> Maybe Expr -> Block () -> Block ()
for_ initializer condition update loopBody =
    emit
        ( For
            initializer
            condition
            update
            (blockStatements loopBody)
        )

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

shiftL :: Expr -> Expr -> Expr
shiftL = Binary ShiftLeft

shiftR :: Expr -> Expr -> Expr
shiftR = Binary ShiftRight

xor :: Expr -> Expr -> Expr
xor = Binary BitXor

ifElse :: Expr -> Expr -> Expr -> Expr
ifElse = Conditional

call :: Expr -> [Expr] -> Expr
call = Call

call_ :: Expr -> [Expr] -> Block ()
call_ function arguments =
    expr_ (call function arguments)

mma :: (MmaOperation -> Mma) -> MmaOperation -> Block ()
mma shape = expr_ . MmaExpr . shape

m16n8k16 :: MmaOperation -> Mma
m16n8k16 = M16N8K16

accumulateInto :: Operand AsmOutput -> Operand AsmInput -> Operand AsmInput -> MmaOperation
accumulateInto = CABC

multiplyAddInto :: Operand AsmOutput -> Operand AsmInput -> Operand AsmInput -> Operand AsmInput -> MmaOperation
multiplyAddInto = DABC

f16 :: [parameter] -> Operand parameter
f16 = Operand F16

f32 :: [parameter] -> Operand parameter
f32 = Operand F32

syncThreads :: Block ()
syncThreads = emit SyncThreads

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

infixl 5 .|.

(.|.) :: Expr -> Expr -> Expr
(.|.) = Binary BitOr

infixr 3 .&&

(.&&) :: Expr -> Expr -> Expr
(.&&) = Binary LogicalAnd

infixr 2 .||

(.||) :: Expr -> Expr -> Expr
(.||) = Binary LogicalOr

infix 1 .=

(.=) :: Expr -> Expr -> Block ()
lhs .= rhs =
    expr_ (Binary Assign lhs rhs)

emit :: Stmt -> Block ()
emit statement = Block ([statement], ())

blockStatements :: Block () -> [Stmt]
blockStatements (Block (result, ())) = result
