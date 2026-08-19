module Aoewif.Target.Cuda.Syntax (
    BinaryOp (..),
    BlockDim (..),
    BlockIdx (..),
    Expr (..),
    GridDim (..),
    Kernel (..),
    Name (..),
    Parameter (..),
    Stmt (..),
    ThreadIdx (..),
    Type (..),
    UnaryOp (..),
)
where

newtype Name = Name String
    deriving stock (Eq, Ord, Show)

data Type
    = Bool
    | U32
    | USize
    | F32
    | Const Type
    | Pointer Type
    deriving stock (Eq, Show)

data Parameter = Parameter Type Name
    deriving stock (Eq, Show)

data ThreadIdx
    = ThreadIdxX
    | ThreadIdxY
    | ThreadIdxZ
    deriving stock (Eq, Show)

data BlockIdx
    = BlockIdxX
    | BlockIdxY
    | BlockIdxZ
    deriving stock (Eq, Show)

data BlockDim
    = BlockDimX
    | BlockDimY
    | BlockDimZ
    deriving stock (Eq, Show)

data GridDim
    = GridDimX
    | GridDimY
    | GridDimZ
    deriving stock (Eq, Show)

newtype UnaryOp
    = StaticCast Type
    deriving stock (Eq, Show)

data BinaryOp
    = Assign
    | Add
    | Multiply
    | LessThan
    deriving stock (Eq, Show)

data Expr
    = Var Name
    | IntLit Integer
    | FloatLit Float
    | ThreadIdx ThreadIdx
    | BlockIdx BlockIdx
    | BlockDim BlockDim
    | GridDim GridDim
    | Unary UnaryOp Expr
    | Binary BinaryOp Expr Expr
    | Subscript Expr Expr
    | Call Expr [Expr]
    deriving stock (Eq, Show)

data Stmt
    = VarDecl Type Name (Maybe Expr)
    | ExprStmt Expr
    | If Expr [Stmt] (Maybe [Stmt])
    deriving stock (Eq, Show)

data Kernel = Kernel
    { kernelName       :: Name
    , kernelParameters :: [Parameter]
    , kernelBody       :: [Stmt]
    }
    deriving stock (Eq, Show)
