module Aoewif.Internal.Kernel.Lower (
    lower,
) where

import qualified Aoewif.Internal.Compute.Operation as Compute
import qualified Aoewif.Internal.IR                as IR
import qualified Aoewif.Internal.Kernel.Operation  as Kernel
import           Data.Maybe                        (fromJust)

data LowerContext = LowerContext
    { contextIR       :: IR.IR Compute.ComputeBlock
    , contextBindings :: [(IR.IndexId, IR.IndexExpr)]
    , contextValues   :: [(Compute.ComputeValueId, Kernel.ValueId)]
    }

newtype LowerState = LowerState
    { stateNextValue :: Int
    }

newtype Lower value = Lower
    { runLower :: LowerState -> (value, LowerState)
    }

instance Functor Lower where
    fmap transform (Lower action) = Lower $ \state ->
        let (value, nextState) = action state
         in (transform value, nextState)

instance Applicative Lower where
    pure value = Lower (value,)
    Lower functionAction <*> Lower valueAction = Lower $ \state ->
        let (function, functionState) = functionAction state
            (value, valueState) = valueAction functionState
         in (function value, valueState)

instance Monad Lower where
    Lower action >>= next = Lower $ \state ->
        let (value, nextState) = action state
         in runLower (next value) nextState

lower :: IR.IR Compute.ComputeBlock -> IR.IR Kernel.KernelBlock
lower computeIR =
    IR.IR
        { IR.irName = IR.irName computeIR
        , IR.irSymbols = IR.irSymbols computeIR
        , IR.irTensors = IR.irTensors computeIR
        , IR.irBody = body
        }
  where
    initialContext = LowerContext computeIR [] []
    (body, _) = runLower (lowerLoopIR initialContext (IR.irBody computeIR)) (LowerState 0)

lowerLoopIR :: LowerContext -> IR.LoopIR Compute.ComputeBlock -> Lower (IR.LoopIR Kernel.KernelBlock)
lowerLoopIR context (IR.LoopIR statements) =
    IR.LoopIR <$> mapM (lowerStatement context) statements

lowerStatement :: LowerContext -> IR.Statement Compute.ComputeBlock -> Lower (IR.Statement Kernel.KernelBlock)
lowerStatement context statement = case statement of
    IR.For loop body        -> IR.For loop <$> lowerLoopIR context body
    IR.Guard predicate body -> IR.Guard predicate <$> lowerLoopIR context body
    IR.Execute computeBlock -> IR.Execute <$> lowerBlock context computeBlock

lowerBlock :: LowerContext -> IR.Block Compute.ComputeBlock -> Lower (IR.Block Kernel.KernelBlock)
lowerBlock context computeBlock = do
    let blockContext =
            context
                { contextBindings =
                    map
                        (\binding -> (IR.bindingIndex binding, IR.bindingExpression binding))
                        (IR.blockBindings computeBlock)
                , contextValues = []
                }
    statements <- lowerComputeStatements blockContext (Compute.computeStatements (IR.blockOperation computeBlock))
    pure
        IR.Block
            { IR.blockId = IR.blockId computeBlock
            , IR.blockName = IR.blockName computeBlock
            , IR.blockBindings = IR.blockBindings computeBlock
            , IR.blockOperation = Kernel.KernelBlock statements
            }

lowerComputeStatements :: LowerContext -> [Compute.ComputeStatement] -> Lower [Kernel.KernelStatement]
lowerComputeStatements _ [] = pure []
lowerComputeStatements context (statement : statements) = case statement of
    Compute.Load result source indices -> do
        let tensor = IR.tensorAt source (contextIR context)
            address = flattenAddress (IR.tensorShape tensor) (map (lowerComputeIndex context) indices)
        (loadStatements, value) <-
            define (IR.tensorType tensor) (Kernel.LoadOperation (tensorBuffer tensor) address)
        rest <-
            lowerComputeStatements
                context{contextValues = (result, value) : contextValues context}
                statements
        pure (loadStatements ++ rest)
    Compute.Store target indices value -> do
        (valueStatements, result) <- lowerScalar context value
        rest <- lowerComputeStatements context statements
        pure (valueStatements ++ [Kernel.StoreBuffer targetBuffer address result] ++ rest)
      where
        tensor = IR.tensorAt target (contextIR context)
        targetBuffer = tensorBuffer tensor
        address = flattenAddress (IR.tensorShape tensor) (map (lowerComputeIndex context) indices)
    Compute.Update reducer target indices value -> do
        (valueStatements, valueResult) <- lowerScalar context value
        (loadStatements, currentValue) <-
            define IR.F32Type (Kernel.LoadOperation targetBuffer address)
        (reducerStatements, reducedValue) <-
            define IR.F32Type (reducerOperation reducer currentValue valueResult)
        rest <- lowerComputeStatements context statements
        pure
            ( valueStatements
                ++ loadStatements
                ++ reducerStatements
                ++ [Kernel.StoreBuffer targetBuffer address reducedValue]
                ++ rest
            )
      where
        tensor = IR.tensorAt target (contextIR context)
        targetBuffer = tensorBuffer tensor
        address = flattenAddress (IR.tensorShape tensor) (map (lowerComputeIndex context) indices)

lowerScalar :: LowerContext -> Compute.ScalarExpr -> Lower ([Kernel.KernelStatement], Kernel.ValueId)
lowerScalar context expression = case expression of
    Compute.LiteralExpr literal ->
        define (IR.scalarLiteralType literal) (Kernel.LiteralOperation literal)
    Compute.IndexValueExpr identifier ->
        define IR.IndexType (Kernel.IndexOperation (indexValue context identifier))
    Compute.ValueExpr identifier -> pure ([], computeValue context identifier)
    Compute.AddExpr lhs rhs -> lowerBinary context Kernel.AddOperation lhs rhs
    Compute.SubExpr lhs rhs -> lowerBinary context Kernel.SubOperation lhs rhs
    Compute.MulExpr lhs rhs -> lowerBinary context Kernel.MulOperation lhs rhs
    Compute.DivExpr lhs rhs -> lowerBinary context Kernel.DivOperation lhs rhs
    Compute.FmaExpr lhs rhs accumulator -> do
        (lhsStatements, lhsValue) <- lowerScalar context lhs
        (rhsStatements, rhsValue) <- lowerScalar context rhs
        (accumulatorStatements, accumulatorValue) <- lowerScalar context accumulator
        (operationStatements, result) <-
            define IR.F32Type (Kernel.FmaOperation lhsValue rhsValue accumulatorValue)
        pure
            ( lhsStatements ++ rhsStatements ++ accumulatorStatements ++ operationStatements
            , result
            )
    Compute.MinExpr lhs rhs -> lowerBinary context Kernel.MinOperation lhs rhs
    Compute.MaxExpr lhs rhs -> lowerBinary context Kernel.MaxOperation lhs rhs
    Compute.ExpExpr value -> lowerUnary context Kernel.ExpOperation value
    Compute.LogExpr value -> lowerUnary context Kernel.LogOperation value
    Compute.CompareExpr predicate lhs rhs -> do
        (lhsStatements, lhsValue) <- lowerScalar context lhs
        (rhsStatements, rhsValue) <- lowerScalar context rhs
        (operationStatements, result) <-
            define IR.BoolType (Kernel.CompareOperation predicate lhsValue rhsValue)
        pure (lhsStatements ++ rhsStatements ++ operationStatements, result)
    Compute.SelectExpr condition trueValue falseValue -> do
        (conditionStatements, conditionValue) <- lowerScalar context condition
        (trueStatements, loweredTrueValue) <- lowerScalar context trueValue
        (falseStatements, loweredFalseValue) <- lowerScalar context falseValue
        (operationStatements, result) <-
            define
                (Compute.exprType trueValue)
                (Kernel.SelectOperation conditionValue loweredTrueValue loweredFalseValue)
        pure
            ( conditionStatements ++ trueStatements ++ falseStatements ++ operationStatements
            , result
            )

lowerBinary :: LowerContext -> (Kernel.ValueId -> Kernel.ValueId -> Kernel.ScalarOperation) -> Compute.ScalarExpr -> Compute.ScalarExpr -> Lower ([Kernel.KernelStatement], Kernel.ValueId)
lowerBinary context constructor lhs rhs = do
    (lhsStatements, lhsValue) <- lowerScalar context lhs
    (rhsStatements, rhsValue) <- lowerScalar context rhs
    (operationStatements, result) <-
        define IR.F32Type (constructor lhsValue rhsValue)
    pure (lhsStatements ++ rhsStatements ++ operationStatements, result)

lowerUnary :: LowerContext -> (Kernel.ValueId -> Kernel.ScalarOperation) -> Compute.ScalarExpr -> Lower ([Kernel.KernelStatement], Kernel.ValueId)
lowerUnary context constructor value = do
    (valueStatements, valueResult) <- lowerScalar context value
    (operationStatements, result) <-
        define IR.F32Type (constructor valueResult)
    pure (valueStatements ++ operationStatements, result)

define :: IR.DType -> Kernel.ScalarOperation -> Lower ([Kernel.KernelStatement], Kernel.ValueId)
define scalarType operation = Lower $ \state ->
    let identifier = Kernel.ValueId (stateNextValue state)
        value = Kernel.Value identifier scalarType operation
     in ( ([Kernel.DefineValue value], identifier)
        , state{stateNextValue = stateNextValue state + 1}
        )

reducerOperation :: Compute.ReducerKind -> Kernel.ValueId -> Kernel.ValueId -> Kernel.ScalarOperation
reducerOperation reducer = case reducer of
    Compute.AddReducer -> Kernel.AddOperation
    Compute.MulReducer -> Kernel.MulOperation
    Compute.MinReducer -> Kernel.MinOperation
    Compute.MaxReducer -> Kernel.MaxOperation

lowerComputeIndex :: LowerContext -> Compute.ComputeIndexExpr -> IR.IndexExpr
lowerComputeIndex context expression = case expression of
    Compute.IterationIndex identifier -> indexValue context identifier
    Compute.ConstantComputeIndex value -> IR.ConstantIndex value
    Compute.AddComputeIndex lhs rhs ->
        IR.AddIndex (lowerComputeIndex context lhs) (lowerComputeIndex context rhs)
    Compute.MulComputeIndex lhs rhs ->
        IR.MulIndex (lowerComputeIndex context lhs) (lowerComputeIndex context rhs)
    Compute.CeilDivComputeIndex dividend divisor ->
        IR.CeilDivIndex (lowerComputeIndex context dividend) divisor

indexValue :: LowerContext -> IR.IndexId -> IR.IndexExpr
indexValue context identifier = fromJust (lookup identifier (contextBindings context))

computeValue :: LowerContext -> Compute.ComputeValueId -> Kernel.ValueId
computeValue context identifier = fromJust (lookup identifier (contextValues context))

flattenAddress :: [IR.DimExpr] -> [IR.IndexExpr] -> IR.IndexExpr
flattenAddress _ [] = IR.ConstantIndex 0
flattenAddress dimensions (firstIndex : remainingIndices) =
    foldl flatten firstIndex (zip (drop 1 dimensions) remainingIndices)
  where
    flatten address (dimension, indexExpression) =
        IR.AddIndex
            (IR.MulIndex address (IR.DimensionIndex dimension))
            indexExpression

tensorBuffer :: IR.TensorDecl -> Kernel.Buffer
tensorBuffer tensor = case IR.tensorKind tensor of
    IR.InputTensor index  -> Kernel.InputBuffer index
    IR.OutputTensor index -> Kernel.OutputBuffer index
