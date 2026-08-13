module Aoewif.Internal.Schedule.Base (
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
    operationFor,
    createLoopPlan,
    splitLoopPlan,
    reorderLoopPlan,
    bindLoopPlan,
    checkedMultiply,
)
where

import           Aoewif.Internal.IR (ComputeFunction, ComputeOp, ComputeOpId,
                                     Dim (..), IteratorId, IteratorKind (..),
                                     SymbolId, computeIterators,
                                     computeOpIdIndex, iteratorExtent,
                                     iteratorId, iteratorKind, iteratorName,
                                     lookupOperation)
import           Data.List          (find, findIndex, partition)
import           Data.Maybe         (fromMaybe)
import           Data.Word          (Word64)

newtype LoopId = LoopId Int
    deriving (Eq, Ord)

instance Show LoopId where
    show loopId = "LoopId " ++ show (loopIdIndex loopId)

loopIdIndex :: LoopId -> Int
loopIdIndex (LoopId index) = index

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
    { internalLoopPlanNextIndex  :: !Int
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
lookupLoopAxis loopId = find ((== loopId) . loopAxisId) . planLoops

loopFor :: IteratorId -> LoopPlan -> Maybe LoopId
loopFor iterator plan = case map loopAxisId matching of
    [loopId] -> Just loopId
    _        -> Nothing
  where
    matching = filter ((== iterator) . loopSourceIterator) (planLoops plan)

data ScheduleError
    = UnknownOperation Int
    | UnknownIterator Int
    | UnknownLoop Int
    | ZeroSplitFactor
    | ReductionSplitUnsupported String
    | BoundLoopSplitUnsupported String
    | IncompleteLoopOrder Int Int
    | DuplicateLoop Int
    | ReductionReorderUnsupported
    | ParallelLoopInsideReduction
    | ReductionBindUnsupported String
    | LoopAlreadyBound String CudaBinding
    | BindingAlreadyUsed CudaBinding String
    | InvalidCudaTargetLimit String
    | DynamicCudaThreadExtent String
    | ZeroCudaLaunchDimension CudaBinding
    | CudaDimensionExceeded CudaBinding Word64 Word64
    | CudaThreadsPerBlockExceeded Word64 Word64
    | ArithmeticOverflow String
    deriving (Eq, Show)

createLoopPlan :: ComputeOp -> LoopPlan
createLoopPlan operation =
    let iterators = computeIterators operation
        (parallelIterators, reductionIterators) = partition ((== Parallel) . iteratorKind) iterators
        normalized = parallelIterators ++ reductionIterators
        axes = zipWith newAxis [0 ..] normalized
        logicalIndices = map (newLogicalIndex axes) iterators
     in LoopPlan
            { internalLoopPlanNextIndex = length axes
            , internalPlanLoops = axes
            , internalPlanLogicalIndices = logicalIndices
            }
  where
    newAxis index iterator =
        LoopAxis
            { internalLoopAxisId = LoopId index
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
        let outerId = LoopId (internalLoopPlanNextIndex plan)
            innerId = LoopId (internalLoopPlanNextIndex plan + 1)
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
        | loopId `elem` seen = Left (DuplicateLoop (loopIdIndex loopId))
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

operationFor :: ComputeFunction -> ComputeOpId -> Either ScheduleError ComputeOp
operationFor function operation = case lookupOperation operation function of
    Just found -> Right found
    Nothing    -> Left (UnknownOperation (computeOpIdIndex operation))

loopPosition :: LoopId -> LoopPlan -> Either ScheduleError Int
loopPosition loopId plan = maybe (Left (UnknownLoop (loopIdIndex loopId))) Right (findIndex ((== loopId) . loopAxisId) (planLoops plan))

loopAxisChecked :: LoopId -> LoopPlan -> Either ScheduleError LoopAxis
loopAxisChecked loopId plan = do
    position <- loopPosition loopId plan
    pure (planLoops plan !! position)

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
