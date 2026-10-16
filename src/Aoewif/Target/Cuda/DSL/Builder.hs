module Aoewif.Target.Cuda.DSL.Builder (
    Block,
    Kernel,
    KernelBuilder,
    body,
    call_,
    declare,
    define,
    emit,
    expr_,
    for_,
    ifElse_,
    if_,
    kernel,
    parameter,
    shared,
    syncThreads,
    syncWarp,
    zeroArray,
    (.=),
)
where

import           Aoewif.Target.Cuda.Alignment (Alignment)
import           Aoewif.Target.Cuda.Expr      (BinaryOp (..), Expr (..))
import           Aoewif.Target.Cuda.Kernel    (Kernel (..))
import           Aoewif.Target.Cuda.Name      (Name (..))
import           Aoewif.Target.Cuda.Parameter (Parameter (..))
import           Aoewif.Target.Cuda.Stmt      (Stmt (..))
import           Aoewif.Target.Cuda.Type      (Type)

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

for_ :: Type -> String -> Expr -> (Expr -> Expr) -> (Expr -> Expr) -> (Expr -> Block ()) -> Block ()
for_ variableType text initializer step condition loopBody =
    emit
        ( For
            (Just (VarDecl variableType name (Just initializer)))
            (condition index)
            (Just (Binary Assign index (step index)))
            (blockStatements (loopBody index))
        )
  where
    name = Name text
    index = Var name

call_ :: Expr -> [Expr] -> Block ()
call_ function arguments =
    expr_ (Call function arguments)

syncThreads :: Block ()
syncThreads = emit SyncThreads

syncWarp :: Block ()
syncWarp = emit SyncWarp

infix 1 .=

(.=) :: Expr -> Expr -> Block ()
lhs .= rhs =
    expr_ (Binary Assign lhs rhs)

emit :: Stmt -> Block ()
emit statement = Block ([statement], ())

blockStatements :: Block () -> [Stmt]
blockStatements (Block (result, ())) = result
