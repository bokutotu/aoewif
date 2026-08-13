module Aoewif.Internal.Kernel.Lower (
    lowerCpuSchedule,
    lowerCudaSchedule,
) where

import qualified Aoewif.Internal.Compute.IR    as Compute
import qualified Aoewif.Internal.Kernel.IR     as Kernel
import qualified Aoewif.Internal.Schedule.Cpu  as Cpu
import qualified Aoewif.Internal.Schedule.Cuda as Cuda
import qualified Aoewif.Internal.Schedule.IR   as Schedule
import           Data.List                     (nub, partition, sort)
import           Data.Maybe                    (fromJust)

data LowerContext = LowerContext
    { contextProgram      :: Compute.Program
    , contextPlan         :: Schedule.LoopPlan
    , contextBuiltin      :: Schedule.CudaBinding -> Maybe Kernel.BuiltinIndex
    , contextIndices      :: [(Compute.IndexId, Kernel.IndexExpression)]
    , contextAccumulators :: [(Compute.AccumulatorId, Kernel.ValueId)]
    , contextOrigin       :: Kernel.Origin
    }

data LowerState = LowerState
    { stateNextValue       :: !Int
    , stateNextAccumulator :: !Int
    , stateNextLoop        :: !Int
    }

newtype Lower value = Lower
    { runLower :: LowerState -> (value, LowerState)
    }

instance Functor Lower where
    fmap transform (Lower action) = Lower $ \state ->
        let (value, nextState) = action state
         in (transform value, nextState)

instance Applicative Lower where
    pure value = Lower $ \state -> (value, state)
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
    lowerKernel
        (const Nothing)
        (Cpu.cpuScheduleProgram schedule)
        (Cpu.cpuScheduleCompute schedule)
        (Cpu.cpuSchedulePlan schedule)

lowerCudaSchedule :: Cuda.CudaSchedule -> Kernel.Kernel
lowerCudaSchedule schedule =
    lowerKernel
        (Just . lowerCudaBinding)
        (Cuda.cudaScheduleProgram schedule)
        (Cuda.cudaScheduleCompute schedule)
        (Cuda.cudaSchedulePlan schedule)

lowerKernel ::
    (Schedule.CudaBinding -> Maybe Kernel.BuiltinIndex) ->
    Compute.Program ->
    Compute.Compute ->
    Schedule.LoopPlan ->
    Kernel.Kernel
lowerKernel builtin program computation plan =
    Kernel.Kernel
        { Kernel.kernelName = Compute.programName program
        , Kernel.kernelSymbols = Compute.programSymbols program
        , Kernel.kernelInputs = zipWith Kernel.Input [0 ..] (Compute.programInputs program)
        , Kernel.kernelOutput = Compute.computeResult computation
        , Kernel.kernelBody = wrappedBody
        }
  where
    initialContext =
        LowerContext
            { contextProgram = program
            , contextPlan = plan
            , contextBuiltin = builtin
            , contextIndices = map rootIndex (Compute.computeIndices computation)
            , contextAccumulators = []
            , contextOrigin = Kernel.ComputeOrigin (Compute.computeId computation) (Compute.computeName computation)
            }
    rootIndex index =
        ( Compute.indexId index
        , lowerLogicalIndex initialContext (Schedule.logicalIndexFor (Compute.indexId index) plan)
        )
    initialState =
        LowerState
            { stateNextValue = 0
            , stateNextAccumulator = 0
            , stateNextLoop = nextLoopIndex plan
            }
    ((expressionStatements, result), _) = runLower (lowerExpression initialContext (Compute.computeBody computation)) initialState
    outputAddress =
        flattenAddress
            (Compute.tensorShape (Compute.computeResult computation))
            [indexValue initialContext (Compute.indexId index) | index <- Compute.computeIndices computation]
    resultStatements =
        hoistLoads
            ( expressionStatements
                ++ [Kernel.StoreOutput outputAddress result (contextOrigin initialContext)]
            )
    predicates = sort (nub (concatMap (lowerTailPredicates initialContext) (Schedule.planLogicalIndices plan)))
    guardedBody = case predicates of
        [] -> resultStatements
        _  -> [Kernel.Conditional predicates resultStatements]
    spatialLoops = filter ((== Schedule.SpatialLoop) . Schedule.loopKind) (Schedule.planLoops plan)
    wrappedBody = foldr (wrapSpatialLoop initialContext) guardedBody spatialLoops

lowerExpression :: LowerContext -> Compute.Expression -> Lower ([Kernel.Statement], Kernel.ValueId)
lowerExpression context expression = case expression of
    Compute.LiteralExpression literal ->
        defineValue context (Compute.scalarLiteralType literal) (Kernel.LiteralOperation literal)
    Compute.IndexValueExpression index ->
        defineValue context Compute.IndexType (Kernel.IndexOperation (indexValue context (Compute.indexId index)))
    Compute.ReadExpression tensor indices -> lowerRead context tensor indices
    Compute.AccumulatorExpression identifier ->
        pure ([], fromJust (lookup identifier (contextAccumulators context)))
    Compute.AddExpression lhs rhs -> lowerBinary context Kernel.AddOperation lhs rhs
    Compute.SubExpression lhs rhs -> lowerBinary context Kernel.SubOperation lhs rhs
    Compute.MulExpression lhs rhs -> lowerBinary context Kernel.MulOperation lhs rhs
    Compute.DivExpression lhs rhs -> lowerBinary context Kernel.DivOperation lhs rhs
    Compute.FmaExpression lhs rhs accumulator -> do
        (lhsStatements, lhsValue) <- lowerExpression context lhs
        (rhsStatements, rhsValue) <- lowerExpression context rhs
        (accumulatorStatements, accumulatorValue) <- lowerExpression context accumulator
        (operationStatements, result) <-
            defineValue context Compute.F32Type (Kernel.FmaOperation lhsValue rhsValue accumulatorValue)
        pure (lhsStatements ++ rhsStatements ++ accumulatorStatements ++ operationStatements, result)
    Compute.MinExpression lhs rhs -> lowerBinary context Kernel.MinOperation lhs rhs
    Compute.MaxExpression lhs rhs -> lowerBinary context Kernel.MaxOperation lhs rhs
    Compute.ExpExpression value -> lowerUnary context Kernel.ExpOperation value
    Compute.LogExpression value -> lowerUnary context Kernel.LogOperation value
    Compute.CompareExpression predicate lhs rhs -> do
        (lhsStatements, lhsValue) <- lowerExpression context lhs
        (rhsStatements, rhsValue) <- lowerExpression context rhs
        (operationStatements, result) <-
            defineValue context Compute.BoolType (Kernel.CompareOperation predicate lhsValue rhsValue)
        pure (lhsStatements ++ rhsStatements ++ operationStatements, result)
    Compute.SelectExpression condition trueValue falseValue -> do
        (conditionStatements, conditionValue) <- lowerExpression context condition
        (trueStatements, loweredTrueValue) <- lowerExpression context trueValue
        (falseStatements, loweredFalseValue) <- lowerExpression context falseValue
        (operationStatements, result) <-
            defineValue
                context
                (Compute.expressionType trueValue)
                (Kernel.SelectOperation conditionValue loweredTrueValue loweredFalseValue)
        pure
            ( conditionStatements ++ trueStatements ++ falseStatements ++ operationStatements
            , result
            )
    Compute.FoldExpression reductionIndex initialValue accumulator body ->
        lowerFold context reductionIndex initialValue accumulator body
    Compute.NamedExpression identifier name namedCompute ->
        lowerExpression context{contextOrigin = Kernel.NodeOrigin identifier name} namedCompute

lowerRead ::
    LowerContext ->
    Compute.Tensor ->
    [Compute.IndexExpression] ->
    Lower ([Kernel.Statement], Kernel.ValueId)
lowerRead context tensor indices = case Compute.tensorKind tensor of
    Compute.InputTensor inputIndex ->
        defineValue
            context
            Compute.F32Type
            (Kernel.LoadOperation inputIndex (flattenAddress (Compute.tensorShape tensor) loweredIndices))
    Compute.ResultTensor computationId ->
        let producer = Compute.computeAt computationId (contextProgram context)
            producerIndices = zip (map Compute.indexId (Compute.computeIndices producer)) loweredIndices
            producerContext =
                context
                    { contextIndices = producerIndices ++ contextIndices context
                    , contextOrigin = Kernel.ComputeOrigin computationId (Compute.computeName producer)
                    }
         in lowerExpression producerContext (Compute.computeBody producer)
  where
    loweredIndices = map (lowerIndexExpression context) indices

lowerFold ::
    LowerContext ->
    Compute.IndexVar ->
    Float ->
    Compute.AccumulatorId ->
    Compute.Expression ->
    Lower ([Kernel.Statement], Kernel.ValueId)
lowerFold context reductionIndex initialValue accumulator body = do
    accumulatorValue <- freshAccumulator
    (loopIdentifier, loopIndex, loopExtent) <- reductionLoop context reductionIndex
    let bodyContext =
            context
                { contextIndices = (Compute.indexId reductionIndex, loopIndex) : contextIndices context
                , contextAccumulators = (accumulator, accumulatorValue) : contextAccumulators context
                }
    (bodyStatements, bodyResult) <- lowerExpression bodyContext body
    pure
        (
            [ Kernel.InitializeAccumulator
                accumulatorValue
                (Compute.F32Literal initialValue)
                (contextOrigin context)
            , Kernel.ForLoop
                loopIdentifier
                loopExtent
                (bodyStatements ++ [Kernel.AssignValue accumulatorValue bodyResult])
            ]
        , accumulatorValue
        )

reductionLoop ::
    LowerContext ->
    Compute.IndexVar ->
    Lower (Kernel.LoopId, Kernel.IndexExpression, Kernel.IndexExpression)
reductionLoop context reductionIndex = case Schedule.lookupLogicalIndex (Compute.indexId reductionIndex) (contextPlan context) of
    Just logicalIndex ->
        let scheduleLoop = Schedule.loopFor (Compute.indexId reductionIndex) (contextPlan context)
            loopAxis = Schedule.loopAxisFor scheduleLoop (contextPlan context)
         in pure
                ( lowerLoopId scheduleLoop
                , lowerLogicalIndex context logicalIndex
                , lowerLoopExtent (Schedule.loopExtent loopAxis)
                )
    Nothing -> do
        loopIdentifier <- freshLoop
        pure
            ( loopIdentifier
            , Kernel.LoopValue loopIdentifier
            , lowerDim (Compute.indexExtent reductionIndex)
            )

lowerBinary ::
    LowerContext ->
    (Kernel.ValueId -> Kernel.ValueId -> Kernel.ScalarOperation) ->
    Compute.Expression ->
    Compute.Expression ->
    Lower ([Kernel.Statement], Kernel.ValueId)
lowerBinary context constructor lhs rhs = do
    (lhsStatements, lhsValue) <- lowerExpression context lhs
    (rhsStatements, rhsValue) <- lowerExpression context rhs
    (operationStatements, result) <- defineValue context Compute.F32Type (constructor lhsValue rhsValue)
    pure (lhsStatements ++ rhsStatements ++ operationStatements, result)

lowerUnary ::
    LowerContext ->
    (Kernel.ValueId -> Kernel.ScalarOperation) ->
    Compute.Expression ->
    Lower ([Kernel.Statement], Kernel.ValueId)
lowerUnary context constructor input = do
    (inputStatements, inputValue) <- lowerExpression context input
    (operationStatements, result) <- defineValue context Compute.F32Type (constructor inputValue)
    pure (inputStatements ++ operationStatements, result)

defineValue ::
    LowerContext ->
    Compute.ScalarType ->
    Kernel.ScalarOperation ->
    Lower ([Kernel.Statement], Kernel.ValueId)
defineValue context scalarType operation = do
    identifier <- freshTemporary
    let value =
            Kernel.Value
                { Kernel.valueId = identifier
                , Kernel.valueType = scalarType
                , Kernel.valueOrigin = contextOrigin context
                , Kernel.valueOperation = operation
                }
    pure ([Kernel.DefineValue value], identifier)

freshTemporary :: Lower Kernel.ValueId
freshTemporary = Lower $ \state ->
    let identifier = Kernel.newValueId (stateNextValue state) Kernel.TemporaryValue
     in (identifier, state{stateNextValue = stateNextValue state + 1})

freshAccumulator :: Lower Kernel.ValueId
freshAccumulator = Lower $ \state ->
    let identifier =
            Kernel.newValueId
                (stateNextValue state)
                (Kernel.AccumulatorValue (stateNextAccumulator state))
        nextState =
            state
                { stateNextValue = stateNextValue state + 1
                , stateNextAccumulator = stateNextAccumulator state + 1
                }
     in (identifier, nextState)

freshLoop :: Lower Kernel.LoopId
freshLoop = Lower $ \state ->
    let identifier = Kernel.LoopId (stateNextLoop state)
     in (identifier, state{stateNextLoop = stateNextLoop state + 1})

indexValue :: LowerContext -> Compute.IndexId -> Kernel.IndexExpression
indexValue context identifier = fromJust (lookup identifier (contextIndices context))

lowerIndexExpression :: LowerContext -> Compute.IndexExpression -> Kernel.IndexExpression
lowerIndexExpression context expression = case expression of
    Compute.VariableIndex index -> indexValue context (Compute.indexId index)
    Compute.ConstantIndex value -> Kernel.ConstantValue value

lowerLogicalIndex :: LowerContext -> Schedule.LogicalIndex -> Kernel.IndexExpression
lowerLogicalIndex context logicalIndex = lowerLoopExpression context (Schedule.logicalExpression logicalIndex)

lowerLoopExpression :: LowerContext -> Schedule.LoopIndexExpression -> Kernel.IndexExpression
lowerLoopExpression context expression = case expression of
    Schedule.LoopIndex loopIdentifier -> loopValue context loopIdentifier
    Schedule.LoopConstant value -> Kernel.ConstantValue value
    Schedule.AddIndex lhs rhs -> Kernel.AddValue (lowerLoopExpression context lhs) (lowerLoopExpression context rhs)
    Schedule.MulIndex lhs rhs -> Kernel.MulValue (lowerLoopExpression context lhs) (lowerLoopExpression context rhs)

loopValue :: LowerContext -> Schedule.LoopId -> Kernel.IndexExpression
loopValue context loopIdentifier =
    case Schedule.loopBinding (Schedule.loopAxisFor loopIdentifier (contextPlan context)) >>= contextBuiltin context of
        Just builtin -> Kernel.BuiltinValue builtin
        Nothing      -> Kernel.LoopValue (lowerLoopId loopIdentifier)

lowerLoopId :: Schedule.LoopId -> Kernel.LoopId
lowerLoopId = Kernel.LoopId . Schedule.loopIdIndex

lowerLoopExtent :: Schedule.LoopExtent -> Kernel.IndexExpression
lowerLoopExtent extent = case extent of
    Schedule.StaticExtent value -> Kernel.DimensionValue value
    Schedule.SymbolExtent symbol -> Kernel.SymbolValue symbol
    Schedule.CeilDivExtent dividend divisor -> Kernel.CeilDivValue (lowerLoopExtent dividend) divisor

lowerDim :: Compute.Dim -> Kernel.IndexExpression
lowerDim dimension = case dimension of
    Compute.StaticDim value  -> Kernel.DimensionValue value
    Compute.SymbolDim symbol -> Kernel.SymbolValue symbol

flattenAddress :: [Compute.Dim] -> [Kernel.IndexExpression] -> Kernel.IndexExpression
flattenAddress _ [] = Kernel.ConstantValue 0
flattenAddress dimensions (firstIndex : remainingIndices) =
    foldl flatten firstIndex (zip (drop 1 dimensions) remainingIndices)
  where
    flatten address (dimension, index) = Kernel.FlattenedValue address (lowerDim dimension) index

hoistLoads :: [Kernel.Statement] -> [Kernel.Statement]
hoistLoads statements = loads ++ remaining
  where
    (loads, remaining) = partition isLoad (map descend statements)
    isLoad statement = case statement of
        Kernel.DefineValue value -> case Kernel.valueOperation value of
            Kernel.LoadOperation _ _ -> True
            _                        -> False
        _ -> False
    descend statement = case statement of
        Kernel.ForLoop identifier extent body -> Kernel.ForLoop identifier extent (hoistLoads body)
        Kernel.Conditional predicates body -> Kernel.Conditional predicates (hoistLoads body)
        _ -> statement

lowerTailPredicates :: LowerContext -> Schedule.LogicalIndex -> [Kernel.IndexPredicate]
lowerTailPredicates context logicalIndex = map lowerPredicate (Schedule.logicalTailPredicates logicalIndex)
  where
    lowerPredicate predicate =
        Kernel.IndexLessThan
            (lowerLoopExpression context (Schedule.tailPredicateIndex predicate))
            (lowerLoopExtent (Schedule.tailPredicateExtent predicate))

wrapSpatialLoop :: LowerContext -> Schedule.LoopAxis -> [Kernel.Statement] -> [Kernel.Statement]
wrapSpatialLoop context loopAxis body =
    case Schedule.loopBinding loopAxis >>= contextBuiltin context of
        Just _ -> body
        Nothing ->
            [ Kernel.ForLoop
                (lowerLoopId (Schedule.loopAxisId loopAxis))
                (lowerLoopExtent (Schedule.loopExtent loopAxis))
                body
            ]

nextLoopIndex :: Schedule.LoopPlan -> Int
nextLoopIndex plan = 1 + maximum (-1 : map (Schedule.loopIdIndex . Schedule.loopAxisId) (Schedule.planLoops plan))

lowerCudaBinding :: Schedule.CudaBinding -> Kernel.BuiltinIndex
lowerCudaBinding binding = case binding of
    Schedule.BlockX  -> Kernel.BlockX
    Schedule.BlockY  -> Kernel.BlockY
    Schedule.BlockZ  -> Kernel.BlockZ
    Schedule.ThreadX -> Kernel.ThreadX
    Schedule.ThreadY -> Kernel.ThreadY
    Schedule.ThreadZ -> Kernel.ThreadZ
