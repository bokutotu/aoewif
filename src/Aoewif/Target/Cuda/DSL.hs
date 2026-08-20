module Aoewif.Target.Cuda.DSL (
    Block,
    Expr,
    Kernel,
    KernelBuilder,
    Type (..),
    body,
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
    declare,
    define,
    emit,
    expr_,
    float,
    for_,
    gridDimX,
    gridDimY,
    gridDimZ,
    ifElse_,
    if_,
    int,
    kernel,
    parameter,
    shared,
    syncThreads,
    threadIdxX,
    threadIdxY,
    threadIdxZ,
    var,
    (!),
    (.*),
    (.+),
    (.-),
    (.%),
    (.=),
    (.<),
    (.>>),
    (.&),
    (.^),
)
where

import           Aoewif.Target.Cuda.Syntax

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

define :: Type -> String -> Expr -> Block Expr
define variableType text initializer = do
    emit (VarDecl variableType name (Just initializer))
    pure (Var name)
  where
    name = Name text

shared :: Type -> String -> Expr -> Block Expr
shared elementType text extent = do
    emit (SharedDecl elementType name extent)
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

for_ :: Block Expr -> (Expr -> Expr) -> (Expr -> Expr) -> (Expr -> Block ()) -> Block ()
for_ initBlock condition update loopBody = do
    let Block (initStmts, loopVar) = initBlock
    emit
        ( For
            initStmts
            (condition loopVar)
            (Just (Binary Assign loopVar (update loopVar)))
            (blockStatements (loopBody loopVar))
        )

var :: String -> Expr
var = Var . Name

int :: Integer -> Expr
int = IntLit

float :: Float -> Expr
float = FloatLit

cast :: Type -> Expr -> Expr
cast targetType = Unary (StaticCast targetType)

bitcast :: Type -> Expr -> Expr
bitcast targetType = Unary (ReinterpretCast targetType)

call :: Expr -> [Expr] -> Expr
call = Call

call_ :: Expr -> [Expr] -> Block ()
call_ function arguments =
    expr_ (call function arguments)

syncThreads :: Block ()
syncThreads =
    call_ (var "__syncthreads") []

threadIdxX, threadIdxY, threadIdxZ :: Expr
threadIdxX = ThreadIdx ThreadIdxX
threadIdxY = ThreadIdx ThreadIdxY
threadIdxZ = ThreadIdx ThreadIdxZ

blockIdxX, blockIdxY, blockIdxZ :: Expr
blockIdxX = BlockIdx BlockIdxX
blockIdxY = BlockIdx BlockIdxY
blockIdxZ = BlockIdx BlockIdxZ

blockDimX, blockDimY, blockDimZ :: Expr
blockDimX = BlockDim BlockDimX
blockDimY = BlockDim BlockDimY
blockDimZ = BlockDim BlockDimZ

gridDimX, gridDimY, gridDimZ :: Expr
gridDimX = GridDim GridDimX
gridDimY = GridDim GridDimY
gridDimZ = GridDim GridDimZ

infixl 9 !

(!) :: Expr -> Expr -> Expr
(!) = Subscript

infixl 7 .*

(.*) :: Expr -> Expr -> Expr
(.*) = Binary Multiply

infixl 7 .%

(.%) :: Expr -> Expr -> Expr
(.%) = Binary Modulo

infixl 6 .+

(.+) :: Expr -> Expr -> Expr
(.+) = Binary Add

infixl 6 .-

(.-) :: Expr -> Expr -> Expr
(.-) = Binary Subtract

infix 4 .<

(.<) :: Expr -> Expr -> Expr
(.<) = Binary LessThan

infixl 5 .>>

(.>>) :: Expr -> Expr -> Expr
(.>>) = Binary ShiftRight

infixl 3 .&

(.&) :: Expr -> Expr -> Expr
(.&) = Binary BitAnd

infixl 2 .^

(.^) :: Expr -> Expr -> Expr
(.^) = Binary BitXor

infix 1 .=

(.=) :: Expr -> Expr -> Block ()
lhs .= rhs =
    expr_ (Binary Assign lhs rhs)

emit :: Stmt -> Block ()
emit statement =
    Block ([statement], ())

blockStatements :: Block () -> [Stmt]
blockStatements (Block (result, ())) = result
