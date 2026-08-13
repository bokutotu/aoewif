module Aoewif.Internal.Schedule.Cuda (
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
    splitCudaSchedule,
    reorderCudaSchedule,
    bindCudaSchedule,
)
where

import           Aoewif.Internal.IR            (ComputeFunction, ComputeOp,
                                                ComputeOpId)
import           Aoewif.Internal.Schedule.Base (CudaBinding, CudaDimension (..),
                                                LoopAxis, LoopId, LoopPlan,
                                                ScheduleError (..),
                                                bindLoopPlan, bindingDimension,
                                                checkedMultiply, createLoopPlan,
                                                isThreadBinding, lookupLoopAxis,
                                                loopBinding, loopExtent,
                                                loopIdIndex, loopName,
                                                operationFor, planLoops,
                                                reorderLoopPlan, splitLoopPlan,
                                                staticLoopExtent)
import           Data.Word                     (Word64)

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
    { internalCudaScheduleFunction  :: ComputeFunction
    , internalCudaScheduleOperation :: ComputeOp
    , internalCudaSchedulePlan      :: LoopPlan
    , internalCudaScheduleTarget    :: CudaTarget
    }
    deriving (Eq, Show)

cudaScheduleFunction :: CudaSchedule -> ComputeFunction
cudaScheduleFunction = internalCudaScheduleFunction

cudaSchedulePlan :: CudaSchedule -> LoopPlan
cudaSchedulePlan = internalCudaSchedulePlan

cudaScheduleTarget :: CudaSchedule -> CudaTarget
cudaScheduleTarget = internalCudaScheduleTarget

newCudaSchedule :: ComputeFunction -> ComputeOpId -> CudaTarget -> Either ScheduleError CudaSchedule
newCudaSchedule function operationId target = do
    checkCudaTarget target
    operation <- operationFor function operationId
    pure
        CudaSchedule
            { internalCudaScheduleFunction = function
            , internalCudaScheduleOperation = operation
            , internalCudaSchedulePlan = createLoopPlan operation
            , internalCudaScheduleTarget = target
            }

cudaScheduleOperation :: CudaSchedule -> ComputeOp
cudaScheduleOperation = internalCudaScheduleOperation

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
    axis <- maybe (Left (UnknownLoop (loopIdIndex loopId))) Right (lookupLoopAxis loopId plan)
    checkCudaBinding binding axis plan (cudaScheduleTarget schedule)
    pure schedule{internalCudaSchedulePlan = plan}

checkCudaTarget :: CudaTarget -> Either ScheduleError ()
checkCudaTarget target = check limits
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
    check [] = Right ()
    check ((field, value) : rest)
        | value == 0 = Left (InvalidCudaTargetLimit field)
        | otherwise = check rest

checkCudaBinding :: CudaBinding -> LoopAxis -> LoopPlan -> CudaTarget -> Either ScheduleError ()
checkCudaBinding binding axis plan target = do
    case staticLoopExtent (loopExtent axis) of
        Nothing
            | isThreadBinding binding -> Left (DynamicCudaThreadExtent (loopName axis))
            | otherwise -> Right ()
        Just extent
            | extent == 0 -> Left (ZeroCudaLaunchDimension binding)
            | extent > dimensionLimit target binding ->
                Left (CudaDimensionExceeded binding extent (dimensionLimit target binding))
            | otherwise -> Right ()
    if isThreadBinding binding
        then do
            threads <- threadCount 1 (planLoops plan)
            if threads > cudaMaxThreadsPerBlock target
                then Left (CudaThreadsPerBlockExceeded threads (cudaMaxThreadsPerBlock target))
                else Right ()
        else Right ()

threadCount :: Word64 -> [LoopAxis] -> Either ScheduleError Word64
threadCount threads [] = Right threads
threadCount threads (axis : rest) = case loopBinding axis of
    Just binding
        | isThreadBinding binding -> case staticLoopExtent (loopExtent axis) of
            Nothing -> Left (DynamicCudaThreadExtent (loopName axis))
            Just extent -> do
                product' <- checkedMultiply "CUDA threads per block" threads extent
                threadCount product' rest
    _ -> threadCount threads rest

dimensionLimit :: CudaTarget -> CudaBinding -> Word64
dimensionLimit target binding
    | isThreadBinding binding = cudaDimGet (cudaMaxBlockDimensions target) (bindingDimension binding)
    | otherwise = cudaDimGet (cudaMaxGridDimensions target) (bindingDimension binding)
