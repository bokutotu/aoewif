module Aoewif.Internal.Kernel.Lower (
    lowerCpuSchedule,
    lowerCudaSchedule,
) where

import qualified Aoewif.Internal.Compute.IR    as Compute
import qualified Aoewif.Internal.Kernel.IR     as Kernel
import qualified Aoewif.Internal.Schedule.Cpu  as Cpu
import qualified Aoewif.Internal.Schedule.Cuda as Cuda
import qualified Aoewif.Internal.Schedule.IR   as Schedule
import           Data.Maybe                    (fromJust)

data LowerContext = LowerContext
    { contextCompute :: Compute.ComputeIR
    , contextAxes    :: [(Compute.AxisId, Kernel.IndexExpression)]
    , contextLoops   :: [(Schedule.LoopId, Kernel.IndexExpression)]
    , contextOrigin  :: Kernel.Origin
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

lowerCpuSchedule :: Cpu.CpuSchedule -> Kernel.Kernel
lowerCpuSchedule schedule =
    lowerKernel (Cpu.cpuScheduleCompute schedule) (Cpu.cpuScheduleIR schedule)

lowerCudaSchedule :: Cuda.CudaSchedule -> Kernel.Kernel
lowerCudaSchedule schedule =
    lowerKernel (Cuda.cudaScheduleCompute schedule) (Cuda.cudaScheduleIR schedule)

lowerKernel :: Compute.ComputeIR -> Schedule.ScheduleIR -> Kernel.Kernel
lowerKernel computeIR scheduleIR =
    Kernel.Kernel
        { Kernel.kernelName = nameText (Compute.computeName computeIR)
        , Kernel.kernelSymbols = Compute.computeSymbols computeIR
        , Kernel.kernelInputs = map inputFor inputTensors
        , Kernel.kernelOutputs = map outputFor outputTensors
        , Kernel.kernelBody = body
        }
  where
    inputTensors = filter isInput (Compute.computeTensors computeIR)
    outputTensors = filter isOutput (Compute.computeTensors computeIR)
    inputFor tensor = case Compute.tensorKind tensor of
        Compute.InputTensor index -> Kernel.Input index tensor
        Compute.OutputTensor _    -> error "inputTensors contains an output"
    outputFor tensor = case Compute.tensorKind tensor of
        Compute.OutputTensor index -> Kernel.Output index tensor
        Compute.InputTensor _      -> error "outputTensors contains an input"
    isInput tensor = case Compute.tensorKind tensor of
        Compute.InputTensor _  -> True
        Compute.OutputTensor _ -> False
    isOutput = not . isInput
    initialContext =
        LowerContext
            { contextCompute = computeIR
            , contextAxes = []
            , contextLoops = []
            , contextOrigin = Kernel.BlockOrigin (Compute.BlockId 0) (Compute.Name "")
            }
    (body, _) = runLower (lowerNode initialContext (Schedule.scheduleRoot scheduleIR)) (LowerState 0)

lowerNode :: LowerContext -> Schedule.ScheduleNode -> Lower [Kernel.Statement]
lowerNode context node = case node of
    Schedule.Band dimensions child -> do
        let loopContext = foldl addLoop context dimensions
        childStatements <- lowerNode loopContext child
        pure (foldr wrapLoop childStatements dimensions)
    Schedule.Sequence children -> concat <$> mapM (lowerNode context) children
    Schedule.Guard predicate child -> do
        body <- lowerNode context child
        pure [Kernel.Conditional [lowerPredicate context predicate] body]
    Schedule.Leaf blockIdentifier bindings -> lowerLeaf context blockIdentifier bindings
  where
    addLoop current dimension =
        current
            { contextLoops =
                (Schedule.loopId dimension, loopIndexValue dimension)
                    : contextLoops current
            }
    wrapLoop dimension body =
        [ Kernel.ForLoop
            (lowerLoopId (Schedule.loopId dimension))
            (lowerDim (Schedule.loopLowerBound dimension))
            (lowerDim (Schedule.loopExtent dimension))
            (lowerExecution (Schedule.loopExecution dimension))
            (Schedule.loopUnrollFactor dimension)
            (lowerCudaBinding <$> Schedule.loopCudaBinding dimension)
            body
        ]
    loopIndexValue dimension =
        case Schedule.loopCudaBinding dimension of
            Just binding ->
                addLowerBound
                    (lowerDim (Schedule.loopLowerBound dimension))
                    (Kernel.BuiltinValue (lowerCudaBinding binding))
            Nothing -> Kernel.LoopValue (lowerLoopId (Schedule.loopId dimension))

lowerLeaf :: LowerContext -> Compute.BlockId -> [Schedule.AxisBinding] -> Lower [Kernel.Statement]
lowerLeaf context blockIdentifier bindings = do
    let computeBlock = Compute.blockAt blockIdentifier (contextCompute context)
        leafContext =
            context
                { contextAxes =
                    map
                        (\binding -> (Schedule.bindingAxis binding, lowerScheduleIndex context (Schedule.bindingIndex binding)))
                        bindings
                , contextOrigin = Kernel.BlockOrigin blockIdentifier (Compute.blockName computeBlock)
                }
    concat <$> mapM (lowerDefinition leafContext computeBlock) (Compute.blockDefinitions computeBlock)

lowerDefinition :: LowerContext -> Compute.ComputeBlock -> Compute.Definition -> Lower [Kernel.Statement]
lowerDefinition context computeBlock definition = case definition of
    Compute.PointwiseDef target indices value -> do
        (valueStatements, result) <- lowerScalar context value
        pure
            ( valueStatements
                ++ [Kernel.StoreBuffer targetBuffer (targetAddress indices) result (contextOrigin context)]
            )
      where
        targetBuffer = tensorBuffer (Compute.tensorAt target (contextCompute context))
    Compute.ReductionDef target indices reducer identity reduceAxes value -> do
        (identityStatements, identityValue) <- lowerScalar context identity
        (valueStatements, valueResult) <- lowerScalar context value
        currentValue <- defineValue context Compute.F32Type (Kernel.LoadOperation targetBuffer address)
        reducedValue <- defineValue context Compute.F32Type (reducerOperation reducer (Kernel.valueId currentValue) valueResult)
        pure
            ( [ Kernel.Conditional
                    (map initializationPredicate reduceAxes)
                    ( identityStatements
                        ++ [Kernel.StoreBuffer targetBuffer address identityValue (contextOrigin context)]
                    )
              ]
                ++ valueStatements
                ++ [ Kernel.DefineValue currentValue
                   , Kernel.DefineValue reducedValue
                   , Kernel.StoreBuffer targetBuffer address (Kernel.valueId reducedValue) (contextOrigin context)
                   ]
            )
      where
        targetBuffer = tensorBuffer (Compute.tensorAt target (contextCompute context))
        address = targetAddress indices
        initializationPredicate axisIdentifier =
            let axisDecl = Compute.axisAt axisIdentifier computeBlock
             in Kernel.IndexEqual (axisValue context axisIdentifier) (lowerDim (Compute.axisLower axisDecl))
  where
    targetAddress indices =
        flattenAddress
            (Compute.tensorShape (Compute.tensorAt (Compute.definitionTarget definition) (contextCompute context)))
            (map (lowerComputeIndex context) indices)

lowerScalar :: LowerContext -> Compute.ScalarExpr -> Lower ([Kernel.Statement], Kernel.ValueId)
lowerScalar context expression = case expression of
    Compute.LiteralExpr literal -> define context (Compute.scalarLiteralType literal) (Kernel.LiteralOperation literal)
    Compute.IndexValueExpr axisIdentifier -> define context Compute.IndexType (Kernel.IndexOperation (axisValue context axisIdentifier))
    Compute.LoadExpr tensorIdentifier indices ->
        let tensor = Compute.tensorAt tensorIdentifier (contextCompute context)
            address = flattenAddress (Compute.tensorShape tensor) (map (lowerComputeIndex context) indices)
         in define context (Compute.tensorType tensor) (Kernel.LoadOperation (tensorBuffer tensor) address)
    Compute.AddExpr lhs rhs -> lowerBinary context Kernel.AddOperation lhs rhs
    Compute.SubExpr lhs rhs -> lowerBinary context Kernel.SubOperation lhs rhs
    Compute.MulExpr lhs rhs -> lowerBinary context Kernel.MulOperation lhs rhs
    Compute.DivExpr lhs rhs -> lowerBinary context Kernel.DivOperation lhs rhs
    Compute.FmaExpr lhs rhs accumulator -> do
        (lhsStatements, lhsValue) <- lowerScalar context lhs
        (rhsStatements, rhsValue) <- lowerScalar context rhs
        (accumulatorStatements, accumulatorValue) <- lowerScalar context accumulator
        (operationStatements, result) <-
            define context Compute.F32Type (Kernel.FmaOperation lhsValue rhsValue accumulatorValue)
        pure (lhsStatements ++ rhsStatements ++ accumulatorStatements ++ operationStatements, result)
    Compute.MinExpr lhs rhs -> lowerBinary context Kernel.MinOperation lhs rhs
    Compute.MaxExpr lhs rhs -> lowerBinary context Kernel.MaxOperation lhs rhs
    Compute.ExpExpr value -> lowerUnary context Kernel.ExpOperation value
    Compute.LogExpr value -> lowerUnary context Kernel.LogOperation value
    Compute.CompareExpr predicate lhs rhs -> do
        (lhsStatements, lhsValue) <- lowerScalar context lhs
        (rhsStatements, rhsValue) <- lowerScalar context rhs
        (operationStatements, result) <-
            define context Compute.BoolType (Kernel.CompareOperation predicate lhsValue rhsValue)
        pure (lhsStatements ++ rhsStatements ++ operationStatements, result)
    Compute.SelectExpr condition trueValue falseValue -> do
        (conditionStatements, conditionValue) <- lowerScalar context condition
        (trueStatements, loweredTrueValue) <- lowerScalar context trueValue
        (falseStatements, loweredFalseValue) <- lowerScalar context falseValue
        (operationStatements, result) <-
            define context (Compute.exprType trueValue) (Kernel.SelectOperation conditionValue loweredTrueValue loweredFalseValue)
        pure (conditionStatements ++ trueStatements ++ falseStatements ++ operationStatements, result)
    Compute.NamedExpr identifier name value ->
        lowerScalar context{contextOrigin = Kernel.NodeOrigin identifier name} value

lowerBinary :: LowerContext -> (Kernel.ValueId -> Kernel.ValueId -> Kernel.ScalarOperation) -> Compute.ScalarExpr -> Compute.ScalarExpr -> Lower ([Kernel.Statement], Kernel.ValueId)
lowerBinary context constructor lhs rhs = do
    (lhsStatements, lhsValue) <- lowerScalar context lhs
    (rhsStatements, rhsValue) <- lowerScalar context rhs
    (operationStatements, result) <- define context Compute.F32Type (constructor lhsValue rhsValue)
    pure (lhsStatements ++ rhsStatements ++ operationStatements, result)

lowerUnary :: LowerContext -> (Kernel.ValueId -> Kernel.ScalarOperation) -> Compute.ScalarExpr -> Lower ([Kernel.Statement], Kernel.ValueId)
lowerUnary context constructor value = do
    (valueStatements, valueResult) <- lowerScalar context value
    (operationStatements, result) <- define context Compute.F32Type (constructor valueResult)
    pure (valueStatements ++ operationStatements, result)

define :: LowerContext -> Compute.DType -> Kernel.ScalarOperation -> Lower ([Kernel.Statement], Kernel.ValueId)
define context scalarType operation = do
    value <- defineValue context scalarType operation
    pure ([Kernel.DefineValue value], Kernel.valueId value)

defineValue :: LowerContext -> Compute.DType -> Kernel.ScalarOperation -> Lower Kernel.Value
defineValue context scalarType operation = Lower $ \state ->
    let identifier = Kernel.newValueId (stateNextValue state) Kernel.TemporaryValue
        value = Kernel.Value identifier scalarType (contextOrigin context) operation
     in (value, state{stateNextValue = stateNextValue state + 1})

reducerOperation :: Compute.ReducerKind -> Kernel.ValueId -> Kernel.ValueId -> Kernel.ScalarOperation
reducerOperation reducer = case reducer of
    Compute.AddReducer -> Kernel.AddOperation
    Compute.MulReducer -> Kernel.MulOperation
    Compute.MinReducer -> Kernel.MinOperation
    Compute.MaxReducer -> Kernel.MaxOperation

lowerPredicate :: LowerContext -> Schedule.Predicate -> Kernel.IndexPredicate
lowerPredicate context (Schedule.IndexLessThan indexExpression extent) =
    Kernel.IndexLessThan (lowerScheduleIndex context indexExpression) (lowerDim extent)

lowerScheduleIndex :: LowerContext -> Schedule.IndexExpr -> Kernel.IndexExpression
lowerScheduleIndex context expression = case expression of
    Schedule.LoopIndex identifier -> loopValue context identifier
    Schedule.ConstantIndex value -> Kernel.ConstantValue value
    Schedule.AddIndex lhs rhs -> Kernel.AddValue (lowerScheduleIndex context lhs) (lowerScheduleIndex context rhs)
    Schedule.MulIndex lhs rhs -> Kernel.MulValue (lowerScheduleIndex context lhs) (lowerScheduleIndex context rhs)

lowerComputeIndex :: LowerContext -> Compute.IndexExpr -> Kernel.IndexExpression
lowerComputeIndex context expression = case expression of
    Compute.AxisIndex identifier -> axisValue context identifier
    Compute.ConstantIndex value -> Kernel.ConstantValue value
    Compute.AddIndex lhs rhs -> Kernel.AddValue (lowerComputeIndex context lhs) (lowerComputeIndex context rhs)
    Compute.MulIndex lhs rhs -> Kernel.MulValue (lowerComputeIndex context lhs) (lowerComputeIndex context rhs)
    Compute.CeilDivIndex dividend divisor -> Kernel.CeilDivValue (lowerComputeIndex context dividend) divisor

loopValue :: LowerContext -> Schedule.LoopId -> Kernel.IndexExpression
loopValue context identifier = fromJust (lookup identifier (contextLoops context))

axisValue :: LowerContext -> Compute.AxisId -> Kernel.IndexExpression
axisValue context identifier = fromJust (lookup identifier (contextAxes context))

lowerDim :: Compute.DimExpr -> Kernel.IndexExpression
lowerDim dimension = case dimension of
    Compute.StaticDim value -> Kernel.DimensionValue value
    Compute.SymbolDim symbol -> Kernel.SymbolValue symbol
    Compute.CeilDivDim dividend divisor -> Kernel.CeilDivValue (lowerDim dividend) divisor

addLowerBound :: Kernel.IndexExpression -> Kernel.IndexExpression -> Kernel.IndexExpression
addLowerBound lower value = case lower of
    Kernel.DimensionValue 0 -> value
    _                       -> Kernel.AddValue lower value

flattenAddress :: [Compute.DimExpr] -> [Kernel.IndexExpression] -> Kernel.IndexExpression
flattenAddress _ [] = Kernel.ConstantValue 0
flattenAddress dimensions (firstIndex : remainingIndices) =
    foldl flatten firstIndex (zip (drop 1 dimensions) remainingIndices)
  where
    flatten address (dimension, indexExpression) =
        Kernel.FlattenedValue address (lowerDim dimension) indexExpression

tensorBuffer :: Compute.TensorDecl -> Kernel.Buffer
tensorBuffer tensor = case Compute.tensorKind tensor of
    Compute.InputTensor index  -> Kernel.InputBuffer index
    Compute.OutputTensor index -> Kernel.OutputBuffer index

lowerLoopId :: Schedule.LoopId -> Kernel.LoopId
lowerLoopId (Schedule.LoopId identifier) = Kernel.LoopId identifier

lowerExecution :: Schedule.ExecutionKind -> Kernel.LoopExecution
lowerExecution execution = case execution of
    Schedule.Serial   -> Kernel.SerialExecution
    Schedule.Parallel -> Kernel.ParallelExecution

lowerCudaBinding :: Schedule.CudaBinding -> Kernel.BuiltinIndex
lowerCudaBinding binding = case binding of
    Schedule.BlockX  -> Kernel.BlockX
    Schedule.BlockY  -> Kernel.BlockY
    Schedule.BlockZ  -> Kernel.BlockZ
    Schedule.ThreadX -> Kernel.ThreadX
    Schedule.ThreadY -> Kernel.ThreadY
    Schedule.ThreadZ -> Kernel.ThreadZ

nameText :: Compute.Name -> String
nameText (Compute.Name name) = name
