module Aoewif.Target.Cuda.Syntax (
    BinaryOp (..),
    BlockDim (..),
    BlockIdx (..),
    Expression (..),
    GridDim (..),
    Kernel (..),
    Name (..),
    Parameter (..),
    Statement (..),
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

data Expression
    = Variable Name
    | IntegerLiteral Integer
    | FloatLiteral Float
    | ThreadIdx ThreadIdx
    | BlockIdx BlockIdx
    | BlockDim BlockDim
    | GridDim GridDim
    | Unary UnaryOp Expression
    | Binary BinaryOp Expression Expression
    | Subscript Expression Expression
    | Call Expression [Expression]
    deriving stock (Eq, Show)

data Statement
    = VariableDeclaration Type Name (Maybe Expression)
    | ExpressionStatement Expression
    | If Expression [Statement] (Maybe [Statement])
    deriving stock (Eq, Show)

data Kernel = Kernel
    { kernelName       :: Name
    , kernelParameters :: [Parameter]
    , kernelBody       :: [Statement]
    }
    deriving stock (Eq, Show)
