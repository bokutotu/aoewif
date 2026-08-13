module Aoewif.Internal.Kernel.IR (
    ValueId,
    ValueRole (..),
    valueIndex,
    valueRole,
    LoopId (..),
    BuiltinIndex (..),
    IndexExpression (..),
    IndexPredicate (..),
    Origin (..),
    ScalarOperation (..),
    Value (..),
    Statement (..),
    Input (..),
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

data IndexPredicate = IndexLessThan IndexExpression IndexExpression
    deriving stock (Eq, Ord, Show)

data Origin
    = ComputeOrigin Compute.ComputeId String
    | NodeOrigin Compute.ComputeNodeId String
    deriving stock (Eq, Show)

data ScalarOperation
    = LiteralOperation Compute.ScalarLiteral
    | IndexOperation IndexExpression
    | LoadOperation Int IndexExpression
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
    , valueType      :: Compute.ScalarType
    , valueOrigin    :: Origin
    , valueOperation :: ScalarOperation
    }
    deriving stock (Eq, Show)

data Statement
    = DefineValue Value
    | InitializeAccumulator ValueId Compute.ScalarLiteral Origin
    | AssignValue ValueId ValueId
    | ForLoop LoopId IndexExpression [Statement]
    | Conditional [IndexPredicate] [Statement]
    | StoreOutput IndexExpression ValueId Origin
    deriving stock (Eq, Show)

data Input = Input
    { inputIndex  :: Int
    , inputTensor :: Compute.Tensor
    }
    deriving stock (Eq, Show)

data Kernel = Kernel
    { kernelName    :: String
    , kernelSymbols :: [Compute.Symbol]
    , kernelInputs  :: [Input]
    , kernelOutput  :: Compute.Tensor
    , kernelBody    :: [Statement]
    }
    deriving stock (Eq, Show)
