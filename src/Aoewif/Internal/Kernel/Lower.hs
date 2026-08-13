module Aoewif.Internal.Kernel.Lower (
    lower,
) where

import qualified Aoewif.Internal.Compute.Operation as Compute
import qualified Aoewif.Internal.IR                as IR
import qualified Aoewif.Internal.Kernel.Operation  as Kernel
import qualified Aoewif.Internal.Primitive         as Primitive
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
            define
                (Primitive.DataValueType (IR.tensorDataType tensor))
                (Kernel.LoadOperation (tensorBuffer tensor) address)
        rest <-
            lowerComputeStatements
                context{contextValues = (result, value) : contextValues context}
                statements
        pure (loadStatements ++ rest)
    Compute.Store target indices value -> do
        (valueStatements, result) <- lowerData context value
        rest <- lowerComputeStatements context statements
        pure (valueStatements ++ [Kernel.StoreBuffer targetBuffer address result] ++ rest)
      where
        tensor = IR.tensorAt target (contextIR context)
        targetBuffer = tensorBuffer tensor
        address = flattenAddress (IR.tensorShape tensor) (map (lowerComputeIndex context) indices)
    Compute.Update reducer target indices value -> do
        (valueStatements, valueResult) <- lowerData context value
        (loadStatements, currentValue) <-
            define tensorValueType (Kernel.LoadOperation targetBuffer address)
        (reducerStatements, reducedValue) <-
            define tensorValueType (reducerOperation reducer currentValue valueResult)
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
        tensorValueType = Primitive.DataValueType (IR.tensorDataType tensor)

lowerData :: LowerContext -> Compute.DataExpr -> Lower ([Kernel.KernelStatement], Kernel.ValueId)
lowerData context expression = case expression of
    Compute.DataLiteralExpr value ->
        define f32ValueType (Kernel.DataLiteralOperation value)
    Compute.ValueExpr identifier -> pure ([], computeValue context identifier)
    Compute.AddExpr lhs rhs -> lowerDataBinary context Kernel.AddOperation lhs rhs
    Compute.SubExpr lhs rhs -> lowerDataBinary context Kernel.SubOperation lhs rhs
    Compute.MulExpr lhs rhs -> lowerDataBinary context Kernel.MulOperation lhs rhs
    Compute.DivExpr lhs rhs -> lowerDataBinary context Kernel.DivOperation lhs rhs
    Compute.FmaExpr lhs rhs accumulator -> do
        (lhsStatements, lhsValue) <- lowerData context lhs
        (rhsStatements, rhsValue) <- lowerData context rhs
        (accumulatorStatements, accumulatorValue) <- lowerData context accumulator
        (operationStatements, result) <-
            define f32ValueType (Kernel.FmaOperation lhsValue rhsValue accumulatorValue)
        pure
            ( lhsStatements ++ rhsStatements ++ accumulatorStatements ++ operationStatements
            , result
            )
    Compute.MinExpr lhs rhs -> lowerDataBinary context Kernel.MinOperation lhs rhs
    Compute.MaxExpr lhs rhs -> lowerDataBinary context Kernel.MaxOperation lhs rhs
    Compute.ExpExpr value -> lowerDataUnary context Kernel.ExpOperation value
    Compute.LogExpr value -> lowerDataUnary context Kernel.LogOperation value
    Compute.SelectDataExpr condition trueValue falseValue -> do
        (conditionStatements, conditionValue) <- lowerPredicate context condition
        (trueStatements, loweredTrueValue) <- lowerData context trueValue
        (falseStatements, loweredFalseValue) <- lowerData context falseValue
        (operationStatements, result) <-
            define
                f32ValueType
                (Kernel.SelectOperation conditionValue loweredTrueValue loweredFalseValue)
        pure
            ( conditionStatements ++ trueStatements ++ falseStatements ++ operationStatements
            , result
            )

lowerPredicate :: LowerContext -> Compute.PredicateExpr -> Lower ([Kernel.KernelStatement], Kernel.ValueId)
lowerPredicate context expression = case expression of
    Compute.PredicateLiteralExpr value ->
        define Primitive.PredicateValueType (Kernel.PredicateLiteralOperation value)
    Compute.CompareDataExpr predicate lhs rhs ->
        lowerComparison (lowerData context) predicate lhs rhs
    Compute.CompareBooleanExpr predicate lhs rhs ->
        lowerComparison (lowerPredicate context) predicate lhs rhs
    Compute.CompareIndexExpr predicate lhs rhs ->
        lowerComparison (lowerIndexValue context) predicate lhs rhs
    Compute.SelectPredicateExpr condition trueValue falseValue -> do
        (conditionStatements, conditionValue) <- lowerPredicate context condition
        (trueStatements, loweredTrueValue) <- lowerPredicate context trueValue
        (falseStatements, loweredFalseValue) <- lowerPredicate context falseValue
        (operationStatements, result) <-
            define
                Primitive.PredicateValueType
                (Kernel.SelectOperation conditionValue loweredTrueValue loweredFalseValue)
        pure
            ( conditionStatements ++ trueStatements ++ falseStatements ++ operationStatements
            , result
            )

lowerIndexValue :: LowerContext -> Compute.IndexValueExpr -> Lower ([Kernel.KernelStatement], Kernel.ValueId)
lowerIndexValue context expression = case expression of
    Compute.ComputeIndexValueExpr indexExpression ->
        define
            Primitive.IndexValueType
            (Kernel.IndexOperation (lowerComputeIndex context indexExpression))
    Compute.SelectIndexValueExpr condition trueValue falseValue -> do
        (conditionStatements, conditionValue) <- lowerPredicate context condition
        (trueStatements, loweredTrueValue) <- lowerIndexValue context trueValue
        (falseStatements, loweredFalseValue) <- lowerIndexValue context falseValue
        (operationStatements, result) <-
            define
                Primitive.IndexValueType
                (Kernel.SelectOperation conditionValue loweredTrueValue loweredFalseValue)
        pure
            ( conditionStatements ++ trueStatements ++ falseStatements ++ operationStatements
            , result
            )

lowerComparison :: (expression -> Lower ([Kernel.KernelStatement], Kernel.ValueId)) -> Primitive.ComparePredicate -> expression -> expression -> Lower ([Kernel.KernelStatement], Kernel.ValueId)
lowerComparison lowerValue predicate lhs rhs = do
    (lhsStatements, lhsValue) <- lowerValue lhs
    (rhsStatements, rhsValue) <- lowerValue rhs
    (operationStatements, result) <-
        define Primitive.PredicateValueType (Kernel.CompareOperation predicate lhsValue rhsValue)
    pure (lhsStatements ++ rhsStatements ++ operationStatements, result)

lowerDataBinary :: LowerContext -> (Kernel.ValueId -> Kernel.ValueId -> Kernel.ScalarOperation) -> Compute.DataExpr -> Compute.DataExpr -> Lower ([Kernel.KernelStatement], Kernel.ValueId)
lowerDataBinary context constructor lhs rhs = do
    (lhsStatements, lhsValue) <- lowerData context lhs
    (rhsStatements, rhsValue) <- lowerData context rhs
    (operationStatements, result) <-
        define f32ValueType (constructor lhsValue rhsValue)
    pure (lhsStatements ++ rhsStatements ++ operationStatements, result)

lowerDataUnary :: LowerContext -> (Kernel.ValueId -> Kernel.ScalarOperation) -> Compute.DataExpr -> Lower ([Kernel.KernelStatement], Kernel.ValueId)
lowerDataUnary context constructor value = do
    (valueStatements, valueResult) <- lowerData context value
    (operationStatements, result) <-
        define f32ValueType (constructor valueResult)
    pure (valueStatements ++ operationStatements, result)

define :: Primitive.ValueType -> Kernel.ScalarOperation -> Lower ([Kernel.KernelStatement], Kernel.ValueId)
define valueType operation = Lower $ \state ->
    let identifier = Kernel.ValueId (stateNextValue state)
        value = Kernel.Value identifier valueType operation
     in ( ([Kernel.DefineValue value], identifier)
        , state{stateNextValue = stateNextValue state + 1}
        )

f32ValueType :: Primitive.ValueType
f32ValueType = Primitive.DataValueType Primitive.F32Type

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
