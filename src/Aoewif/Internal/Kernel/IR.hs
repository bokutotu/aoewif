module Aoewif.Internal.Kernel.IR (
    ValueId,
    ValueRole (..),
    valueIndex,
    valueRole,
    Buffer (..),
    LoopId (..),
    BuiltinIndex (..),
    LoopExecution (..),
    IndexExpression (..),
    IndexPredicate (..),
    Origin (..),
    ScalarOperation (..),
    Value (..),
    Statement (..),
    Input (..),
    Output (..),
    Kernel (..),
    newValueId,
) where

import qualified Aoewif.Internal.Compute.IR as Compute
import           Data.Word                  (Word64)

data ValueRole
    = TemporaryValue
    | AccumulatorValue Int
    deriving stock (Eq, Show)

data ValueId = ValueId Int ValueRole
    deriving stock (Eq, Show)

newValueId :: Int -> ValueRole -> ValueId
newValueId = ValueId

valueIndex :: ValueId -> Int
valueIndex (ValueId identifier _) = identifier

valueRole :: ValueId -> ValueRole
valueRole (ValueId _ role) = role

data Buffer
    = InputBuffer Int
    | OutputBuffer Int
    deriving stock (Eq, Ord, Show)

newtype LoopId = LoopId Int
    deriving stock (Eq, Ord, Show)

data BuiltinIndex
    = BlockX
    | BlockY
    | BlockZ
    | ThreadX
    | ThreadY
    | ThreadZ
    deriving stock (Eq, Ord, Show)

data LoopExecution
    = SerialExecution
    | ParallelExecution
    deriving stock (Eq, Show)

data IndexExpression
    = LoopValue LoopId
    | BuiltinValue BuiltinIndex
    | ConstantValue Word64
    | DimensionValue Word64
    | SymbolValue Compute.SymbolId
    | AddValue IndexExpression IndexExpression
    | MulValue IndexExpression IndexExpression
    | CeilDivValue IndexExpression Word64
    | FlattenedValue IndexExpression IndexExpression IndexExpression
    deriving stock (Eq, Ord, Show)

data IndexPredicate
    = IndexLessThan IndexExpression IndexExpression
    | IndexEqual IndexExpression IndexExpression
    deriving stock (Eq, Ord, Show)

data Origin
    = BlockOrigin Compute.BlockId Compute.Name
    | NodeOrigin Compute.ComputeNodeId Compute.Name
    deriving stock (Eq, Show)

data ScalarOperation
    = LiteralOperation Compute.ScalarLiteral
    | IndexOperation IndexExpression
    | LoadOperation Buffer IndexExpression
    | AddOperation ValueId ValueId
    | SubOperation ValueId ValueId
    | MulOperation ValueId ValueId
    | DivOperation ValueId ValueId
    | FmaOperation ValueId ValueId ValueId
    | MinOperation ValueId ValueId
    | MaxOperation ValueId ValueId
    | ExpOperation ValueId
    | LogOperation ValueId
    | CompareOperation Compute.ComparePredicate ValueId ValueId
    | SelectOperation ValueId ValueId ValueId
    deriving stock (Eq, Show)

data Value = Value
    { valueId        :: ValueId
    , valueType      :: Compute.DType
    , valueOrigin    :: Origin
    , valueOperation :: ScalarOperation
    }
    deriving stock (Eq, Show)

data Statement
    = DefineValue Value
    | ForLoop
        LoopId
        IndexExpression
        IndexExpression
        LoopExecution
        (Maybe Word64)
        (Maybe BuiltinIndex)
        [Statement]
    | Conditional [IndexPredicate] [Statement]
    | StoreBuffer Buffer IndexExpression ValueId Origin
    deriving stock (Eq, Show)

data Input = Input
    { inputIndex  :: Int
    , inputTensor :: Compute.TensorDecl
    }
    deriving stock (Eq, Show)

data Output = Output
    { outputIndex  :: Int
    , outputTensor :: Compute.TensorDecl
    }
    deriving stock (Eq, Show)

data Kernel = Kernel
    { kernelName    :: String
    , kernelSymbols :: [Compute.Symbol]
    , kernelInputs  :: [Input]
    , kernelOutputs :: [Output]
    , kernelBody    :: [Statement]
    }
    deriving stock (Eq, Show)
