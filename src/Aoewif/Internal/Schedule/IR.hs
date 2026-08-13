module Aoewif.Internal.Schedule.IR (
    LoopId,
    loopIdIndex,
    LoopExtent (..),
    staticLoopExtent,
    LoopIndexExpression (..),
    TailPredicate,
    tailPredicateIndex,
    tailPredicateExtent,
    LogicalIndex,
    logicalIndexVariable,
    logicalExpression,
    logicalTailPredicates,
    CudaBinding (..),
    CudaDimension (..),
    isBlockBinding,
    isThreadBinding,
    bindingDimension,
    LoopKind (..),
    LoopAxis,
    loopAxisId,
    loopSourceIndex,
    loopName,
    loopExtent,
    loopKind,
    loopBinding,
    LoopPlan,
    planLoops,
    planLogicalIndices,
    lookupLogicalIndex,
    logicalIndexFor,
    lookupLoopAxis,
    loopAxisFor,
    loopFor,
    ScheduleError (..),
    createLoopPlan,
    splitLoopPlan,
    reorderLoopPlan,
    bindLoopPlan,
    checkedMultiply,
)
where

import qualified Aoewif.Internal.Compute.IR as Compute
import           Data.List                  (find, findIndex)
import           Data.Maybe                 (fromJust)
import           Data.Word                  (Word64)

newtype LoopId = LoopId Int
    deriving stock (Eq, Ord)

instance Show LoopId where
    show loopId = "LoopId " ++ show (loopIdIndex loopId)

loopIdIndex :: LoopId -> Int
loopIdIndex (LoopId index) = index

data LoopExtent
    = StaticExtent Word64
    | SymbolExtent Compute.SymbolId
    | CeilDivExtent LoopExtent Word64
    deriving stock (Eq, Show)

staticLoopExtent :: LoopExtent -> Maybe Word64
staticLoopExtent extent = case extent of
    StaticExtent value -> Just value
    SymbolExtent _ -> Nothing
    CeilDivExtent dividend divisor -> ceilDivWord64 <$> staticLoopExtent dividend <*> pure divisor

data LoopIndexExpression
    = LoopIndex LoopId
    | LoopConstant Word64
    | AddIndex LoopIndexExpression LoopIndexExpression
    | MulIndex LoopIndexExpression LoopIndexExpression
    deriving stock (Eq, Show)

data TailPredicate = TailPredicate
    { internalTailPredicateIndex  :: LoopIndexExpression
    , internalTailPredicateExtent :: LoopExtent
    }
    deriving stock (Eq, Show)

tailPredicateIndex :: TailPredicate -> LoopIndexExpression
tailPredicateIndex = internalTailPredicateIndex

tailPredicateExtent :: TailPredicate -> LoopExtent
tailPredicateExtent = internalTailPredicateExtent

data LogicalIndex = LogicalIndex
    { internalLogicalIndexVariable :: Compute.IndexId
    , internalLogicalExpression    :: LoopIndexExpression
    , internalTailPredicates       :: [TailPredicate]
    }
    deriving stock (Eq, Show)

logicalIndexVariable :: LogicalIndex -> Compute.IndexId
logicalIndexVariable = internalLogicalIndexVariable

logicalExpression :: LogicalIndex -> LoopIndexExpression
logicalExpression = internalLogicalExpression

logicalTailPredicates :: LogicalIndex -> [TailPredicate]
logicalTailPredicates = internalTailPredicates

data CudaBinding
    = BlockX
    | BlockY
    | BlockZ
    | ThreadX
    | ThreadY
    | ThreadZ
    deriving stock (Eq, Ord, Show)

data CudaDimension = DimensionX | DimensionY | DimensionZ
    deriving stock (Eq, Ord, Show)

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

data LoopKind
    = SpatialLoop
    | ReductionLoop
    deriving stock (Eq, Show)

data LoopAxis = LoopAxis
    { internalLoopAxisId  :: LoopId
    , internalLoopSource  :: Compute.IndexId
    , internalLoopName    :: String
    , internalLoopExtent  :: LoopExtent
    , internalLoopKind    :: LoopKind
    , internalLoopBinding :: Maybe CudaBinding
    }
    deriving stock (Eq, Show)

loopAxisId :: LoopAxis -> LoopId
loopAxisId = internalLoopAxisId

loopSourceIndex :: LoopAxis -> Compute.IndexId
loopSourceIndex = internalLoopSource

loopName :: LoopAxis -> String
loopName = internalLoopName

loopExtent :: LoopAxis -> LoopExtent
loopExtent = internalLoopExtent

loopKind :: LoopAxis -> LoopKind
loopKind = internalLoopKind

loopBinding :: LoopAxis -> Maybe CudaBinding
loopBinding = internalLoopBinding

data LoopPlan = LoopPlan
    { internalLoopPlanNextIndex  :: !Int
    , internalPlanLoops          :: [LoopAxis]
    , internalPlanLogicalIndices :: [LogicalIndex]
    }
    deriving stock (Eq, Show)

planLoops :: LoopPlan -> [LoopAxis]
planLoops = internalPlanLoops

planLogicalIndices :: LoopPlan -> [LogicalIndex]
planLogicalIndices = internalPlanLogicalIndices

lookupLogicalIndex :: Compute.IndexId -> LoopPlan -> Maybe LogicalIndex
lookupLogicalIndex index = find ((== index) . logicalIndexVariable) . planLogicalIndices

logicalIndexFor :: Compute.IndexId -> LoopPlan -> LogicalIndex
logicalIndexFor index = fromJust . lookupLogicalIndex index

lookupLoopAxis :: LoopId -> LoopPlan -> Maybe LoopAxis
lookupLoopAxis loopId = find ((== loopId) . loopAxisId) . planLoops

loopAxisFor :: LoopId -> LoopPlan -> LoopAxis
loopAxisFor loopId = fromJust . lookupLoopAxis loopId

loopFor :: Compute.IndexId -> LoopPlan -> LoopId
loopFor index = loopAxisId . fromJust . find ((== index) . loopSourceIndex) . planLoops

data ScheduleError
    = UnknownLoop Int
    | ZeroSplitFactor
    | ReductionSplitUnsupported String
    | BoundLoopSplitUnsupported String
    | IncompleteLoopOrder Int Int
    | DuplicateLoop Int
    | ReductionReorderUnsupported
    | SpatialLoopInsideReduction
    | ReductionBindUnsupported String
    | LoopAlreadyBound String CudaBinding
    | BindingAlreadyUsed CudaBinding String
    | InvalidCudaTargetLimit String
    | DynamicCudaThreadExtent String
    | ZeroCudaLaunchDimension CudaBinding
    | CudaDimensionExceeded CudaBinding Word64 Word64
    | CudaThreadsPerBlockExceeded Word64 Word64
    | ArithmeticOverflow String
    deriving stock (Eq, Show)

createLoopPlan :: Compute.Compute -> LoopPlan
createLoopPlan computation =
    LoopPlan
        { internalLoopPlanNextIndex = length axes
        , internalPlanLoops = axes
        , internalPlanLogicalIndices = zipWith newLogicalIndex indices axes
        }
  where
    spatial = [(index, SpatialLoop) | index <- Compute.computeIndices computation]
    reductions = [(index, ReductionLoop) | index <- Compute.reductionIndices (Compute.computeBody computation)]
    indexed = spatial ++ reductions
    indices = map fst indexed
    axes = zipWith newAxis [0 ..] indexed
    newAxis identifier (index, kind) =
        LoopAxis
            { internalLoopAxisId = LoopId identifier
            , internalLoopSource = Compute.indexId index
            , internalLoopName = Compute.indexName index
            , internalLoopExtent = extentFromDim (Compute.indexExtent index)
            , internalLoopKind = kind
            , internalLoopBinding = Nothing
            }
    newLogicalIndex index axis =
        LogicalIndex
            { internalLogicalIndexVariable = Compute.indexId index
            , internalLogicalExpression = LoopIndex (loopAxisId axis)
            , internalTailPredicates = []
            }

splitLoopPlan :: LoopId -> Word64 -> LoopPlan -> Either ScheduleError (LoopId, LoopId, LoopPlan)
splitLoopPlan loopId factor plan
    | factor == 0 = Left ZeroSplitFactor
    | otherwise = do
        position <- loopPosition loopId plan
        let original = planLoops plan !! position
        case loopKind original of
            ReductionLoop -> Left (ReductionSplitUnsupported (loopName original))
            SpatialLoop -> pure ()
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
            nextPlan =
                plan
                    { internalLoopPlanNextIndex = internalLoopPlanNextIndex plan + 2
                    , internalPlanLoops = before ++ [outer, inner] ++ after
                    , internalPlanLogicalIndices = rewritten
                    }
        pure (outerId, innerId, nextPlan)

rewriteLogicalIndex :: LoopAxis -> LoopId -> LoopIndexExpression -> Word64 -> LogicalIndex -> LogicalIndex
rewriteLogicalIndex original target replacement factor logicalIndex
    | logicalIndexVariable logicalIndex /= loopSourceIndex original = logicalIndex
    | otherwise =
        logicalIndex
            { internalLogicalExpression = replaceLoop target replacement (logicalExpression logicalIndex)
            , internalTailPredicates = rewrittenPredicates ++ newPredicate
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
        let currentReductions = map loopAxisId (filter ((== ReductionLoop) . loopKind) (planLoops plan))
            reorderedReductions = map loopAxisId (filter ((== ReductionLoop) . loopKind) reordered)
        if currentReductions /= reorderedReductions
            then Left ReductionReorderUnsupported
            else
                if hasSpatialInsideReduction reordered
                    then Left SpatialLoopInsideReduction
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
        ReductionLoop -> Left (ReductionBindUnsupported (loopName selected))
        SpatialLoop   -> pure ()
    case loopBinding selected of
        Just existing -> Left (LoopAlreadyBound (loopName selected) existing)
        Nothing       -> pure ()
    case find ((== Just binding) . loopBinding) (planLoops plan) of
        Just existing -> Left (BindingAlreadyUsed binding (loopName existing))
        Nothing       -> pure ()
    let rebound = selected{internalLoopBinding = Just binding}
        loops = take position (planLoops plan) ++ [rebound] ++ drop (position + 1) (planLoops plan)
    pure plan{internalPlanLoops = loops}

loopPosition :: LoopId -> LoopPlan -> Either ScheduleError Int
loopPosition loopId plan = maybe (Left (UnknownLoop (loopIdIndex loopId))) Right (findIndex ((== loopId) . loopAxisId) (planLoops plan))

loopAxisChecked :: LoopId -> LoopPlan -> Either ScheduleError LoopAxis
loopAxisChecked loopId plan = do
    position <- loopPosition loopId plan
    pure (planLoops plan !! position)

extentFromDim :: Compute.Dim -> LoopExtent
extentFromDim dimension = case dimension of
    Compute.StaticDim value  -> StaticExtent value
    Compute.SymbolDim symbol -> SymbolExtent symbol

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

splitIndex :: LoopId -> LoopId -> Word64 -> LoopIndexExpression
splitIndex outer inner factor = AddIndex (MulIndex (LoopIndex outer) (LoopConstant factor)) (LoopIndex inner)

replaceLoop :: LoopId -> LoopIndexExpression -> LoopIndexExpression -> LoopIndexExpression
replaceLoop target replacement expression = case expression of
    LoopIndex loopId
        | loopId == target -> replacement
        | otherwise -> expression
    LoopConstant _ -> expression
    AddIndex lhs rhs -> AddIndex (replaceLoop target replacement lhs) (replaceLoop target replacement rhs)
    MulIndex lhs rhs -> MulIndex (replaceLoop target replacement lhs) (replaceLoop target replacement rhs)

hasSpatialInsideReduction :: [LoopAxis] -> Bool
hasSpatialInsideReduction = go False
  where
    go _ [] = False
    go sawReduction (axis : rest) = case loopKind axis of
        SpatialLoop | sawReduction -> True
        SpatialLoop                -> go sawReduction rest
        ReductionLoop              -> go True rest

ceilDivWord64 :: Word64 -> Word64 -> Word64
ceilDivWord64 value divisor = value `div` divisor + if value `mod` divisor == 0 then 0 else 1

checkedMultiply :: String -> Word64 -> Word64 -> Either ScheduleError Word64
checkedMultiply context lhs rhs
    | lhs /= 0 && rhs > maxBound `div` lhs = Left (ArithmeticOverflow context)
    | otherwise = Right (lhs * rhs)
