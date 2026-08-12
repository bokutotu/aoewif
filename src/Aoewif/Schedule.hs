{-# OPTIONS_GHC -fno-cse -fno-full-laziness #-}

module Aoewif.Schedule (
    LoopId,
    loopIdIndex,
    LoopExtent (..),
    staticLoopExtent,
    LoopIndexExpr (..),
    TailPredicate,
    tailPredicateIndex,
    tailPredicateExtent,
    LogicalIndex,
    logicalIterator,
    logicalExpression,
    logicalTailPredicates,
    CudaBinding (..),
    CudaDimension (..),
    isBlockBinding,
    isThreadBinding,
    bindingDimension,
    LoopAxis,
    loopAxisId,
    loopSourceIterator,
    loopName,
    loopExtent,
    loopKind,
    loopBinding,
    LoopPlan,
    planLoops,
    planLogicalIndices,
    lookupLogicalIndex,
    lookupLoopAxis,
    loopFor,
    ScheduleError (..),
    CpuSchedule,
    newCpuSchedule,
    cpuScheduleFunction,
    cpuScheduleOperation,
    cpuSchedulePlan,
    cpuLoopFor,
    splitCpuSchedule,
    reorderCpuSchedule,
    verifyCpuSchedule,
    VerifiedCpuSchedule,
    verifiedCpuFunction,
    verifiedCpuOperation,
    verifiedCpuPlan,
    CudaDim3,
    newCudaDim3,
    cudaDimX,
    cudaDimY,
    cudaDimZ,
    cudaDimGet,
    CudaTarget,
    newCudaTarget,
    cudaMaxThreadsPerBlock,
    cudaMaxBlockDimensions,
    cudaMaxGridDimensions,
    defaultCudaTarget,
    CudaSchedule,
    newCudaSchedule,
    cudaScheduleFunction,
    cudaScheduleOperation,
    cudaSchedulePlan,
    cudaScheduleTarget,
    cudaLoopFor,
    splitCudaSchedule,
    reorderCudaSchedule,
    bindCudaSchedule,
    verifyCudaSchedule,
    VerifiedCudaSchedule,
    verifiedCudaFunction,
    verifiedCudaOperation,
    verifiedCudaPlan,
    verifiedCudaTarget,
)
where

import           Aoewif.IR         (ComputeOp, ComputeOpId, Dim (..),
                                    IteratorId, IteratorKind (..), SymbolId,
                                    VerifiedComputeFunction, computeIterators,
                                    computeOpIdIndex, iteratorExtent,
                                    iteratorId, iteratorKind, iteratorName,
                                    lookupOperation, verifiedFunction)
import           Control.Exception (Exception)
import           Data.IORef        (IORef, atomicModifyIORef', newIORef)
import           Data.List         (find, findIndex, partition)
import           Data.Maybe        (fromMaybe)
import           Data.Word         (Word64)
import           System.IO.Unsafe  (unsafePerformIO)

data LoopId = LoopId !Word64 !Int
    deriving (Eq, Ord)

instance Show LoopId where
    show loopId = "LoopId " ++ show (loopIdIndex loopId)

loopIdIndex :: LoopId -> Int
loopIdIndex (LoopId _ index) = index

data LoopExtent
    = StaticExtent Word64
    | SymbolExtent SymbolId
    | CeilDivExtent LoopExtent Word64
    deriving (Eq, Show)

staticLoopExtent :: LoopExtent -> Maybe Word64
staticLoopExtent extent = case extent of
    StaticExtent value -> Just value
    SymbolExtent _ -> Nothing
    CeilDivExtent dividend divisor -> ceilDivWord64 <$> staticLoopExtent dividend <*> pure divisor

data LoopIndexExpr
    = LoopIndex LoopId
    | LoopConstant Word64
    | AddIndex LoopIndexExpr LoopIndexExpr
    | MulIndex LoopIndexExpr LoopIndexExpr
    deriving (Eq, Show)

data TailPredicate = TailPredicate
    { internalTailPredicateIndex  :: LoopIndexExpr
    , internalTailPredicateExtent :: LoopExtent
    }
    deriving (Eq, Show)

tailPredicateIndex :: TailPredicate -> LoopIndexExpr
tailPredicateIndex = internalTailPredicateIndex

tailPredicateExtent :: TailPredicate -> LoopExtent
tailPredicateExtent = internalTailPredicateExtent

data LogicalIndex = LogicalIndex
    { internalLogicalIterator       :: IteratorId
    , internalLogicalExpression     :: LoopIndexExpr
    , internalLogicalTailPredicates :: [TailPredicate]
    }
    deriving (Eq, Show)

logicalIterator :: LogicalIndex -> IteratorId
logicalIterator = internalLogicalIterator

logicalExpression :: LogicalIndex -> LoopIndexExpr
logicalExpression = internalLogicalExpression

logicalTailPredicates :: LogicalIndex -> [TailPredicate]
logicalTailPredicates = internalLogicalTailPredicates

data CudaBinding
    = BlockX
    | BlockY
    | BlockZ
    | ThreadX
    | ThreadY
    | ThreadZ
    deriving (Eq, Ord, Show)

data CudaDimension = DimensionX | DimensionY | DimensionZ
    deriving (Eq, Ord, Show)

isBlockBinding :: CudaBinding -> Bool
isBlockBinding binding = case binding of
    BlockX  -> True
    BlockY  -> True
    BlockZ  -> True
    ThreadX -> False
    ThreadY -> False
    ThreadZ -> False

isThreadBinding :: CudaBinding -> Bool
isThreadBinding = not . isBlockBinding

bindingDimension :: CudaBinding -> CudaDimension
bindingDimension binding = case binding of
    BlockX  -> DimensionX
    ThreadX -> DimensionX
    BlockY  -> DimensionY
    ThreadY -> DimensionY
    BlockZ  -> DimensionZ
    ThreadZ -> DimensionZ

data LoopAxis = LoopAxis
    { internalLoopAxisId         :: LoopId
    , internalLoopSourceIterator :: IteratorId
    , internalLoopName           :: String
    , internalLoopExtent         :: LoopExtent
    , internalLoopKind           :: IteratorKind
    , internalLoopBinding        :: Maybe CudaBinding
    }
    deriving (Eq, Show)

loopAxisId :: LoopAxis -> LoopId
loopAxisId = internalLoopAxisId

loopSourceIterator :: LoopAxis -> IteratorId
loopSourceIterator = internalLoopSourceIterator

loopName :: LoopAxis -> String
loopName = internalLoopName

loopExtent :: LoopAxis -> LoopExtent
loopExtent = internalLoopExtent

loopKind :: LoopAxis -> IteratorKind
loopKind = internalLoopKind

loopBinding :: LoopAxis -> Maybe CudaBinding
loopBinding = internalLoopBinding

data LoopPlan = LoopPlan
    { internalLoopPlanOwner      :: !Word64
    , internalLoopPlanNextIndex  :: !Int
    , internalPlanLoops          :: [LoopAxis]
    , internalPlanLogicalIndices :: [LogicalIndex]
    }
    deriving (Eq, Show)

planLoops :: LoopPlan -> [LoopAxis]
planLoops = internalPlanLoops

planLogicalIndices :: LoopPlan -> [LogicalIndex]
planLogicalIndices = internalPlanLogicalIndices

lookupLogicalIndex :: IteratorId -> LoopPlan -> Maybe LogicalIndex
lookupLogicalIndex iterator = find ((== iterator) . logicalIterator) . planLogicalIndices

lookupLoopAxis :: LoopId -> LoopPlan -> Maybe LoopAxis
lookupLoopAxis loopId plan
    | loopOwner loopId /= internalLoopPlanOwner plan = Nothing
    | otherwise = find ((== loopId) . loopAxisId) (planLoops plan)

loopFor :: IteratorId -> LoopPlan -> Maybe LoopId
loopFor iterator plan = case map loopAxisId matching of
    [loopId] -> Just loopId
    _        -> Nothing
  where
    matching = filter ((== iterator) . loopSourceIterator) (planLoops plan)

data ScheduleError
    = UnknownOperation Int
    | ForeignLoop LoopId
    | UnknownLoop LoopId
    | ZeroSplitFactor
    | ReductionSplitUnsupported String
    | BoundLoopSplitUnsupported String
    | IncompleteLoopOrder Int Int
    | DuplicateLoop LoopId
    | ReductionReorderUnsupported
    | ParallelLoopInsideReduction
    | ReductionBindUnsupported String
    | LoopAlreadyBound String CudaBinding
    | BindingAlreadyUsed CudaBinding String
    | CpuBindingUnsupported String CudaBinding
    | InvalidCudaTargetLimit String
    | DynamicCudaThreadExtent String
    | ZeroCudaLaunchDimension CudaBinding
    | CudaDimensionExceeded CudaBinding Word64 Word64
    | CudaThreadsPerBlockExceeded Word64 Word64
    | ArithmeticOverflow String
    deriving (Eq)

instance Show ScheduleError where
    show scheduleError = case scheduleError of
        UnknownOperation index -> "unknown compute operation " ++ show index
        ForeignLoop loopId -> "foreign loop " ++ show (loopIdIndex loopId)
        UnknownLoop loopId -> "unknown loop " ++ show (loopIdIndex loopId)
        ZeroSplitFactor -> "loop split factor must be greater than zero"
        ReductionSplitUnsupported name -> "splitting reduction loop `" ++ name ++ "` is unsupported"
        BoundLoopSplitUnsupported name -> "bound loop `" ++ name ++ "` must be split before CUDA binding"
        IncompleteLoopOrder expected actual ->
            "loop order must contain all " ++ show expected ++ " loops exactly once, got " ++ show actual
        DuplicateLoop loopId -> "loop " ++ show (loopIdIndex loopId) ++ " appears more than once"
        ReductionReorderUnsupported -> "reordering reduction loops is unsupported"
        ParallelLoopInsideReduction -> "strict reductions require all parallel loops outside reduction loops"
        ReductionBindUnsupported name -> "binding reduction loop `" ++ name ++ "` to CUDA is unsupported"
        LoopAlreadyBound name binding -> "loop `" ++ name ++ "` is already bound to " ++ show binding
        BindingAlreadyUsed binding name -> "CUDA binding " ++ show binding ++ " is already used by loop `" ++ name ++ "`"
        CpuBindingUnsupported name binding -> "CPU loop `" ++ name ++ "` cannot have CUDA binding " ++ show binding
        InvalidCudaTargetLimit field -> "CUDA target limit `" ++ field ++ "` must be greater than zero"
        DynamicCudaThreadExtent name -> "CUDA thread loop `" ++ name ++ "` must have a static extent"
        ZeroCudaLaunchDimension binding -> "CUDA binding " ++ show binding ++ " has zero extent"
        CudaDimensionExceeded binding requested limit ->
            "CUDA binding " ++ show binding ++ " requests extent " ++ show requested ++ ", exceeding target limit " ++ show limit
        CudaThreadsPerBlockExceeded requested limit ->
            "CUDA schedule requests " ++ show requested ++ " threads per block, exceeding target limit " ++ show limit
        ArithmeticOverflow context -> "arithmetic overflow while computing " ++ context

instance Exception ScheduleError

data CpuSchedule = CpuSchedule
    { internalCpuScheduleFunction    :: VerifiedComputeFunction
    , internalCpuScheduleOperationId :: ComputeOpId
    , internalCpuSchedulePlan        :: LoopPlan
    }
    deriving (Eq, Show)

cpuScheduleFunction :: CpuSchedule -> VerifiedComputeFunction
cpuScheduleFunction = internalCpuScheduleFunction

cpuSchedulePlan :: CpuSchedule -> LoopPlan
cpuSchedulePlan = internalCpuSchedulePlan

newCpuSchedule :: VerifiedComputeFunction -> ComputeOpId -> Either ScheduleError CpuSchedule
newCpuSchedule function operation = do
    plan <- createLoopPlan function operation
    pure
        CpuSchedule
            { internalCpuScheduleFunction = function
            , internalCpuScheduleOperationId = operation
            , internalCpuSchedulePlan = plan
            }

cpuScheduleOperation :: CpuSchedule -> ComputeOp
cpuScheduleOperation schedule = retainedOperation (cpuScheduleFunction schedule) (internalCpuScheduleOperationId schedule)

cpuLoopFor :: IteratorId -> CpuSchedule -> Maybe LoopId
cpuLoopFor iterator = loopFor iterator . cpuSchedulePlan

splitCpuSchedule :: LoopId -> Word64 -> CpuSchedule -> Either ScheduleError (LoopId, LoopId, CpuSchedule)
splitCpuSchedule loopId factor schedule = do
    (outer, inner, plan) <- splitLoopPlan loopId factor (cpuSchedulePlan schedule)
    pure (outer, inner, schedule{internalCpuSchedulePlan = plan})

reorderCpuSchedule :: [LoopId] -> CpuSchedule -> Either ScheduleError CpuSchedule
reorderCpuSchedule order schedule = do
    plan <- reorderLoopPlan order (cpuSchedulePlan schedule)
    pure schedule{internalCpuSchedulePlan = plan}

verifyCpuSchedule :: CpuSchedule -> Either ScheduleError VerifiedCpuSchedule
verifyCpuSchedule schedule = do
    verifyLoopPlan (cpuSchedulePlan schedule)
    verifyCpuBindings (planLoops (cpuSchedulePlan schedule))
    pure (VerifiedCpuSchedule schedule)

newtype VerifiedCpuSchedule = VerifiedCpuSchedule CpuSchedule
    deriving (Eq, Show)

verifiedCpuFunction :: VerifiedCpuSchedule -> VerifiedComputeFunction
verifiedCpuFunction (VerifiedCpuSchedule schedule) = cpuScheduleFunction schedule

verifiedCpuOperation :: VerifiedCpuSchedule -> ComputeOp
verifiedCpuOperation (VerifiedCpuSchedule schedule) = cpuScheduleOperation schedule

verifiedCpuPlan :: VerifiedCpuSchedule -> LoopPlan
verifiedCpuPlan (VerifiedCpuSchedule schedule) = cpuSchedulePlan schedule

data CudaDim3 = CudaDim3
    { internalCudaDimX :: Word64
    , internalCudaDimY :: Word64
    , internalCudaDimZ :: Word64
    }
    deriving (Eq, Show)

newCudaDim3 :: Word64 -> Word64 -> Word64 -> CudaDim3
newCudaDim3 = CudaDim3

cudaDimX :: CudaDim3 -> Word64
cudaDimX = internalCudaDimX

cudaDimY :: CudaDim3 -> Word64
cudaDimY = internalCudaDimY

cudaDimZ :: CudaDim3 -> Word64
cudaDimZ = internalCudaDimZ

cudaDimGet :: CudaDim3 -> CudaDimension -> Word64
cudaDimGet dimensions dimension = case dimension of
    DimensionX -> cudaDimX dimensions
    DimensionY -> cudaDimY dimensions
    DimensionZ -> cudaDimZ dimensions

data CudaTarget = CudaTarget
    { internalCudaMaxThreadsPerBlock :: Word64
    , internalCudaMaxBlockDimensions :: CudaDim3
    , internalCudaMaxGridDimensions  :: CudaDim3
    }
    deriving (Eq, Show)

newCudaTarget :: Word64 -> CudaDim3 -> CudaDim3 -> CudaTarget
newCudaTarget = CudaTarget

cudaMaxThreadsPerBlock :: CudaTarget -> Word64
cudaMaxThreadsPerBlock = internalCudaMaxThreadsPerBlock

cudaMaxBlockDimensions :: CudaTarget -> CudaDim3
cudaMaxBlockDimensions = internalCudaMaxBlockDimensions

cudaMaxGridDimensions :: CudaTarget -> CudaDim3
cudaMaxGridDimensions = internalCudaMaxGridDimensions

defaultCudaTarget :: CudaTarget
defaultCudaTarget = newCudaTarget 1024 (newCudaDim3 1024 1024 64) (newCudaDim3 2147483647 65535 65535)

data CudaSchedule = CudaSchedule
    { internalCudaScheduleFunction    :: VerifiedComputeFunction
    , internalCudaScheduleOperationId :: ComputeOpId
    , internalCudaSchedulePlan        :: LoopPlan
    , internalCudaScheduleTarget      :: CudaTarget
    }
    deriving (Eq, Show)

cudaScheduleFunction :: CudaSchedule -> VerifiedComputeFunction
cudaScheduleFunction = internalCudaScheduleFunction

cudaSchedulePlan :: CudaSchedule -> LoopPlan
cudaSchedulePlan = internalCudaSchedulePlan

cudaScheduleTarget :: CudaSchedule -> CudaTarget
cudaScheduleTarget = internalCudaScheduleTarget

newCudaSchedule :: VerifiedComputeFunction -> ComputeOpId -> CudaTarget -> Either ScheduleError CudaSchedule
newCudaSchedule function operation target = do
    validateCudaTarget target
    plan <- createLoopPlan function operation
    pure
        CudaSchedule
            { internalCudaScheduleFunction = function
            , internalCudaScheduleOperationId = operation
            , internalCudaSchedulePlan = plan
            , internalCudaScheduleTarget = target
            }

cudaScheduleOperation :: CudaSchedule -> ComputeOp
cudaScheduleOperation schedule = retainedOperation (cudaScheduleFunction schedule) (internalCudaScheduleOperationId schedule)

cudaLoopFor :: IteratorId -> CudaSchedule -> Maybe LoopId
cudaLoopFor iterator = loopFor iterator . cudaSchedulePlan

splitCudaSchedule :: LoopId -> Word64 -> CudaSchedule -> Either ScheduleError (LoopId, LoopId, CudaSchedule)
splitCudaSchedule loopId factor schedule = do
    (outer, inner, plan) <- splitLoopPlan loopId factor (cudaSchedulePlan schedule)
    pure (outer, inner, schedule{internalCudaSchedulePlan = plan})

reorderCudaSchedule :: [LoopId] -> CudaSchedule -> Either ScheduleError CudaSchedule
reorderCudaSchedule order schedule = do
    plan <- reorderLoopPlan order (cudaSchedulePlan schedule)
    pure schedule{internalCudaSchedulePlan = plan}

bindCudaSchedule :: LoopId -> CudaBinding -> CudaSchedule -> Either ScheduleError CudaSchedule
bindCudaSchedule loopId binding schedule = do
    plan <- bindLoopPlan loopId binding (cudaSchedulePlan schedule)
    pure schedule{internalCudaSchedulePlan = plan}

verifyCudaSchedule :: CudaSchedule -> Either ScheduleError VerifiedCudaSchedule
verifyCudaSchedule schedule = do
    verifyLoopPlan (cudaSchedulePlan schedule)
    validateCudaTarget (cudaScheduleTarget schedule)
    validateCudaBindings (cudaSchedulePlan schedule) (cudaScheduleTarget schedule)
    pure (VerifiedCudaSchedule schedule)

newtype VerifiedCudaSchedule = VerifiedCudaSchedule CudaSchedule
    deriving (Eq, Show)

verifiedCudaFunction :: VerifiedCudaSchedule -> VerifiedComputeFunction
verifiedCudaFunction (VerifiedCudaSchedule schedule) = cudaScheduleFunction schedule

verifiedCudaOperation :: VerifiedCudaSchedule -> ComputeOp
verifiedCudaOperation (VerifiedCudaSchedule schedule) = cudaScheduleOperation schedule

verifiedCudaPlan :: VerifiedCudaSchedule -> LoopPlan
verifiedCudaPlan (VerifiedCudaSchedule schedule) = cudaSchedulePlan schedule

verifiedCudaTarget :: VerifiedCudaSchedule -> CudaTarget
verifiedCudaTarget (VerifiedCudaSchedule schedule) = cudaScheduleTarget schedule

createLoopPlan :: VerifiedComputeFunction -> ComputeOpId -> Either ScheduleError LoopPlan
createLoopPlan function operationId = do
    operation <- operationFor function operationId
    let owner = freshLoopPlanOwner operation
        iterators = computeIterators operation
        (parallelIterators, reductionIterators) = partition ((== Parallel) . iteratorKind) iterators
        normalized = parallelIterators ++ reductionIterators
        axes = zipWith (newAxis owner) [0 ..] normalized
        logicalIndices = map (newLogicalIndex axes) iterators
    pure
        LoopPlan
            { internalLoopPlanOwner = owner
            , internalLoopPlanNextIndex = length axes
            , internalPlanLoops = axes
            , internalPlanLogicalIndices = logicalIndices
            }
  where
    newAxis owner index iterator =
        LoopAxis
            { internalLoopAxisId = LoopId owner index
            , internalLoopSourceIterator = iteratorId iterator
            , internalLoopName = iteratorName iterator
            , internalLoopExtent = extentFromDim (iteratorExtent iterator)
            , internalLoopKind = iteratorKind iterator
            , internalLoopBinding = Nothing
            }
    newLogicalIndex axes iterator =
        let loopId = loopAxisId (fromMaybe impossible (find ((== iteratorId iterator) . loopSourceIterator) axes))
         in LogicalIndex
                { internalLogicalIterator = iteratorId iterator
                , internalLogicalExpression = LoopIndex loopId
                , internalLogicalTailPredicates = []
                }
    impossible = error "every compute iterator must have a normalized loop"

splitLoopPlan :: LoopId -> Word64 -> LoopPlan -> Either ScheduleError (LoopId, LoopId, LoopPlan)
splitLoopPlan loopId factor plan
    | factor == 0 = Left ZeroSplitFactor
    | otherwise = do
        position <- loopPosition loopId plan
        let original = planLoops plan !! position
        case loopKind original of
            Reduction -> Left (ReductionSplitUnsupported (loopName original))
            Parallel  -> pure ()
        case loopBinding original of
            Just _  -> Left (BoundLoopSplitUnsupported (loopName original))
            Nothing -> pure ()
        outerExtent <- ceilDivExtent factor (loopExtent original)
        let outerId = LoopId (internalLoopPlanOwner plan) (internalLoopPlanNextIndex plan)
            innerId = LoopId (internalLoopPlanOwner plan) (internalLoopPlanNextIndex plan + 1)
            outer =
                original
                    { internalLoopAxisId = outerId
                    , internalLoopName = loopName original ++ "_outer"
                    , internalLoopExtent = outerExtent
                    }
            inner =
                original
                    { internalLoopAxisId = innerId
                    , internalLoopName = loopName original ++ "_inner"
                    , internalLoopExtent = StaticExtent factor
                    }
            replacement = splitIndex outerId innerId factor
            before = take position (planLoops plan)
            after = drop (position + 1) (planLoops plan)
            rewritten = map (rewriteLogicalIndex original loopId replacement factor) (planLogicalIndices plan)
            newPlan =
                plan
                    { internalLoopPlanNextIndex = internalLoopPlanNextIndex plan + 2
                    , internalPlanLoops = before ++ [outer, inner] ++ after
                    , internalPlanLogicalIndices = rewritten
                    }
        pure (outerId, innerId, newPlan)

rewriteLogicalIndex :: LoopAxis -> LoopId -> LoopIndexExpr -> Word64 -> LogicalIndex -> LogicalIndex
rewriteLogicalIndex original target replacement factor logicalIndex
    | logicalIterator logicalIndex /= loopSourceIterator original = logicalIndex
    | otherwise =
        logicalIndex
            { internalLogicalExpression = replaceLoop target replacement (logicalExpression logicalIndex)
            , internalLogicalTailPredicates = rewrittenPredicates ++ newPredicate
            }
  where
    rewrittenPredicates =
        map
            (\predicate -> predicate{internalTailPredicateIndex = replaceLoop target replacement (tailPredicateIndex predicate)})
            (logicalTailPredicates logicalIndex)
    newPredicate
        | divisibleBy factor (loopExtent original) = []
        | otherwise = [TailPredicate replacement (loopExtent original)]

reorderLoopPlan :: [LoopId] -> LoopPlan -> Either ScheduleError LoopPlan
reorderLoopPlan order plan
    | length order /= length (planLoops plan) = Left (IncompleteLoopOrder (length (planLoops plan)) (length order))
    | otherwise = do
        reordered <- collect [] order
        let currentReductions = map loopAxisId (filter ((== Reduction) . loopKind) (planLoops plan))
            reorderedReductions = map loopAxisId (filter ((== Reduction) . loopKind) reordered)
        if currentReductions /= reorderedReductions
            then Left ReductionReorderUnsupported
            else
                if hasParallelInsideReduction reordered
                    then Left ParallelLoopInsideReduction
                    else pure plan{internalPlanLoops = reordered}
  where
    collect _ [] = pure []
    collect seen (loopId : rest)
        | loopId `elem` seen = Left (DuplicateLoop loopId)
        | otherwise = do
            axis <- loopAxisChecked loopId plan
            remaining <- collect (seen ++ [loopId]) rest
            pure (axis : remaining)

bindLoopPlan :: LoopId -> CudaBinding -> LoopPlan -> Either ScheduleError LoopPlan
bindLoopPlan loopId binding plan = do
    position <- loopPosition loopId plan
    let selected = planLoops plan !! position
    case loopKind selected of
        Reduction -> Left (ReductionBindUnsupported (loopName selected))
        Parallel  -> pure ()
    case loopBinding selected of
        Just existing -> Left (LoopAlreadyBound (loopName selected) existing)
        Nothing       -> pure ()
    case find ((== Just binding) . loopBinding) (planLoops plan) of
        Just existing -> Left (BindingAlreadyUsed binding (loopName existing))
        Nothing       -> pure ()
    let rebound = selected{internalLoopBinding = Just binding}
        loops = take position (planLoops plan) ++ [rebound] ++ drop (position + 1) (planLoops plan)
    pure plan{internalPlanLoops = loops}

verifyLoopPlan :: LoopPlan -> Either ScheduleError ()
verifyLoopPlan plan
    | hasParallelInsideReduction (planLoops plan) = Left ParallelLoopInsideReduction
    | otherwise = Right ()

verifyCpuBindings :: [LoopAxis] -> Either ScheduleError ()
verifyCpuBindings [] = Right ()
verifyCpuBindings (axis : rest) = case loopBinding axis of
    Just binding -> Left (CpuBindingUnsupported (loopName axis) binding)
    Nothing      -> verifyCpuBindings rest

validateCudaTarget :: CudaTarget -> Either ScheduleError ()
validateCudaTarget target = validate limits
  where
    block = cudaMaxBlockDimensions target
    grid = cudaMaxGridDimensions target
    limits =
        [ ("max threads per block", cudaMaxThreadsPerBlock target)
        , ("max block dimension x", cudaDimX block)
        , ("max block dimension y", cudaDimY block)
        , ("max block dimension z", cudaDimZ block)
        , ("max grid dimension x", cudaDimX grid)
        , ("max grid dimension y", cudaDimY grid)
        , ("max grid dimension z", cudaDimZ grid)
        ]
    validate [] = Right ()
    validate ((field, value) : rest)
        | value == 0 = Left (InvalidCudaTargetLimit field)
        | otherwise = validate rest

validateCudaBindings :: LoopPlan -> CudaTarget -> Either ScheduleError ()
validateCudaBindings plan target = do
    threads <- validate 1 (planLoops plan)
    if threads > cudaMaxThreadsPerBlock target
        then Left (CudaThreadsPerBlockExceeded threads (cudaMaxThreadsPerBlock target))
        else Right ()
  where
    validate threads [] = Right threads
    validate threads (axis : rest) = case loopBinding axis of
        Nothing -> validate threads rest
        Just binding -> case staticLoopExtent (loopExtent axis) of
            Nothing
                | isThreadBinding binding -> Left (DynamicCudaThreadExtent (loopName axis))
                | otherwise -> validate threads rest
            Just extent
                | extent == 0 -> Left (ZeroCudaLaunchDimension binding)
                | extent > dimensionLimit target binding ->
                    Left (CudaDimensionExceeded binding extent (dimensionLimit target binding))
                | isThreadBinding binding -> do
                    product' <- checkedMultiply "CUDA threads per block" threads extent
                    validate product' rest
                | otherwise -> validate threads rest

dimensionLimit :: CudaTarget -> CudaBinding -> Word64
dimensionLimit target binding
    | isThreadBinding binding = cudaDimGet (cudaMaxBlockDimensions target) (bindingDimension binding)
    | otherwise = cudaDimGet (cudaMaxGridDimensions target) (bindingDimension binding)

operationFor :: VerifiedComputeFunction -> ComputeOpId -> Either ScheduleError ComputeOp
operationFor function operation = case lookupOperation operation (verifiedFunction function) of
    Just found -> Right found
    Nothing    -> Left (UnknownOperation (computeOpIdIndex operation))

retainedOperation :: VerifiedComputeFunction -> ComputeOpId -> ComputeOp
retainedOperation function operation = case operationFor function operation of
    Right found -> found
    Left _      -> error "a schedule must retain its compute operation"

loopPosition :: LoopId -> LoopPlan -> Either ScheduleError Int
loopPosition loopId plan
    | loopOwner loopId /= internalLoopPlanOwner plan = Left (ForeignLoop loopId)
    | otherwise = maybe (Left (UnknownLoop loopId)) Right (findIndex ((== loopId) . loopAxisId) (planLoops plan))

loopAxisChecked :: LoopId -> LoopPlan -> Either ScheduleError LoopAxis
loopAxisChecked loopId plan = do
    position <- loopPosition loopId plan
    pure (planLoops plan !! position)

loopOwner :: LoopId -> Word64
loopOwner (LoopId owner _) = owner

extentFromDim :: Dim -> LoopExtent
extentFromDim dimension = case dimension of
    StaticDim value  -> StaticExtent value
    SymbolDim symbol -> SymbolExtent symbol

ceilDivExtent :: Word64 -> LoopExtent -> Either ScheduleError LoopExtent
ceilDivExtent divisor extent = case extent of
    StaticExtent value -> Right (StaticExtent (ceilDivWord64 value divisor))
    SymbolExtent _ -> Right (CeilDivExtent extent divisor)
    CeilDivExtent dividend innerDivisor -> do
        combined <- checkedMultiply "split loop divisor" innerDivisor divisor
        Right (CeilDivExtent dividend combined)

divisibleBy :: Word64 -> LoopExtent -> Bool
divisibleBy divisor extent = case staticLoopExtent extent of
    Just value -> value `mod` divisor == 0
    Nothing    -> False

splitIndex :: LoopId -> LoopId -> Word64 -> LoopIndexExpr
splitIndex outer inner factor = AddIndex (MulIndex (LoopIndex outer) (LoopConstant factor)) (LoopIndex inner)

replaceLoop :: LoopId -> LoopIndexExpr -> LoopIndexExpr -> LoopIndexExpr
replaceLoop target replacement expression = case expression of
    LoopIndex loopId
        | loopId == target -> replacement
        | otherwise -> expression
    LoopConstant _ -> expression
    AddIndex lhs rhs -> AddIndex (replaceLoop target replacement lhs) (replaceLoop target replacement rhs)
    MulIndex lhs rhs -> MulIndex (replaceLoop target replacement lhs) (replaceLoop target replacement rhs)

hasParallelInsideReduction :: [LoopAxis] -> Bool
hasParallelInsideReduction = go False
  where
    go _ [] = False
    go sawReduction (axis : rest) = case loopKind axis of
        Parallel | sawReduction -> True
        Parallel                -> go sawReduction rest
        Reduction               -> go True rest

ceilDivWord64 :: Word64 -> Word64 -> Word64
ceilDivWord64 value divisor = value `div` divisor + if value `mod` divisor == 0 then 0 else 1

checkedMultiply :: String -> Word64 -> Word64 -> Either ScheduleError Word64
checkedMultiply context lhs rhs
    | lhs /= 0 && rhs > maxBound `div` lhs = Left (ArithmeticOverflow context)
    | otherwise = Right (lhs * rhs)

{-# NOINLINE loopPlanOwnerCounter #-}
loopPlanOwnerCounter :: IORef Word64
loopPlanOwnerCounter = unsafePerformIO (newIORef 1)

{-# NOINLINE freshLoopPlanOwner #-}
freshLoopPlanOwner :: ComputeOp -> Word64
freshLoopPlanOwner operation = operation `seq` unsafePerformIO (atomicModifyIORef' loopPlanOwnerCounter next)
  where
    next owner = (owner + 1, owner)
