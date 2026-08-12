module ScheduleSpec (spec) where

import Aoewif.IR
import Aoewif.Schedule
import Data.Word (Word64)
import Test.Hspec hiding (parallel)

data ParallelFixture = ParallelFixture
  { parallelFunction :: VerifiedComputeFunction,
    parallelOperation :: ComputeOpId,
    parallelFirst :: IteratorId,
    parallelSecond :: IteratorId
  }

data DynamicFixture = DynamicFixture
  { dynamicFunction :: VerifiedComputeFunction,
    dynamicOperation :: ComputeOpId,
    dynamicIterator :: IteratorId,
    dynamicExtent :: SymbolId
  }

data ReductionFixture = ReductionFixture
  { reductionFunction :: VerifiedComputeFunction,
    reductionOperation :: ComputeOpId,
    reductionRow :: IteratorId,
    firstReduction :: IteratorId,
    secondReduction :: IteratorId
  }

parallel2d :: Word64 -> Word64 -> ParallelFixture
parallel2d firstExtent secondExtent =
  ParallelFixture
    { parallelFunction = function,
      parallelOperation = computeOpId operation,
      parallelFirst = iteratorId first,
      parallelSecond = iteratorId second
    }
  where
    function = mustSucceed $ buildComputeFunction "parallel_2d" $ do
      inputTensor <- input "input" (tensorTypeF32 [StaticDim firstExtent, StaticDim secondExtent])
      output <- compute "output" $ do
        firstIterator <- parallel "first" (StaticDim firstExtent)
        secondIterator <- parallel "second" (StaticDim secondExtent)
        readTensor inputTensor [IteratorIndex firstIterator, IteratorIndex secondIterator]
      markOutput output
    operation = onlyOperation function
    (first, second) = twoIterators operation

parallel1d :: Word64 -> ParallelFixture
parallel1d extent =
  ParallelFixture
    { parallelFunction = function,
      parallelOperation = computeOpId operation,
      parallelFirst = iteratorId element,
      parallelSecond = iteratorId element
    }
  where
    function = mustSucceed $ buildComputeFunction "parallel_1d" $ do
      inputTensor <- input "input" (tensorTypeF32 [StaticDim extent])
      output <- compute "output" $ do
        elementIterator <- parallel "element" (StaticDim extent)
        readTensor inputTensor [IteratorIndex elementIterator]
      markOutput output
    operation = onlyOperation function
    element = oneIterator operation

dynamicParallel1d :: DynamicFixture
dynamicParallel1d =
  DynamicFixture
    { dynamicFunction = function,
      dynamicOperation = computeOpId operation,
      dynamicIterator = iteratorId element,
      dynamicExtent = symbolId extentSymbol
    }
  where
    function = mustSucceed $ buildComputeFunction "dynamic_parallel_1d" $ do
      extent <- symbol "extent"
      inputTensor <- input "input" (tensorTypeF32 [SymbolDim extent])
      output <- compute "output" $ do
        elementIterator <- parallel "element" (SymbolDim extent)
        readTensor inputTensor [IteratorIndex elementIterator]
      markOutput output
    operation = onlyOperation function
    element = oneIterator operation
    extentSymbol = onlySymbol function

mixedReduction :: ReductionFixture
mixedReduction =
  ReductionFixture
    { reductionFunction = function,
      reductionOperation = computeOpId operation,
      reductionRow = iteratorId row,
      firstReduction = iteratorId first,
      secondReduction = iteratorId second
    }
  where
    function = mustSucceed $ buildComputeFunction "mixed_reduction" $ do
      inputTensor <- input "input" (tensorTypeF32 [StaticDim 4, StaticDim 8, StaticDim 6])
      output <- compute "output" $ do
        firstIterator <- reduction "first_reduction" (StaticDim 4)
        rowIterator <- parallel "row" (StaticDim 8)
        secondIterator <- reduction "second_reduction" (StaticDim 6)
        value <- readTensor inputTensor [IteratorIndex firstIterator, IteratorIndex rowIterator, IteratorIndex secondIterator]
        accumulator <- reductionInit (F32Literal 0.0)
        add accumulator value
      markOutput output
    operation = onlyOperation function
    (first, row, second) = threeIterators operation

onlyOperation :: VerifiedComputeFunction -> ComputeOp
onlyOperation function = case functionOperations (verifiedFunction function) of
  [operation] -> operation
  _ -> error "fixture must contain exactly one operation"

oneIterator :: ComputeOp -> Iterator
oneIterator operation = case computeIterators operation of
  [iterator] -> iterator
  _ -> error "fixture operation must contain exactly one iterator"

twoIterators :: ComputeOp -> (Iterator, Iterator)
twoIterators operation = case computeIterators operation of
  [first, second] -> (first, second)
  _ -> error "fixture operation must contain exactly two iterators"

threeIterators :: ComputeOp -> (Iterator, Iterator, Iterator)
threeIterators operation = case computeIterators operation of
  [first, second, third] -> (first, second, third)
  _ -> error "fixture operation must contain exactly three iterators"

onlySymbol :: VerifiedComputeFunction -> Symbol
onlySymbol function = case functionSymbols (verifiedFunction function) of
  [extent] -> extent
  _ -> error "fixture must contain exactly one symbol"

mustSucceed :: Show error => Either error value -> value
mustSucceed = either (error . show) id

splitIndex :: LoopId -> LoopId -> Word64 -> LoopIndexExpr
splitIndex outer inner factor = AddIndex (MulIndex (LoopIndex outer) (LoopConstant factor)) (LoopIndex inner)

data LoopAxisView = LoopAxisView LoopId IteratorId String LoopExtent IteratorKind (Maybe CudaBinding)
  deriving (Eq, Show)

loopAxisView :: LoopAxis -> LoopAxisView
loopAxisView axis =
  LoopAxisView
    (loopAxisId axis)
    (loopSourceIterator axis)
    (loopName axis)
    (loopExtent axis)
    (loopKind axis)
    (loopBinding axis)

data LogicalIndexView = LogicalIndexView IteratorId LoopIndexExpr [(LoopIndexExpr, LoopExtent)]
  deriving (Eq, Show)

logicalIndexView :: LogicalIndex -> LogicalIndexView
logicalIndexView logicalIndex =
  LogicalIndexView
    (logicalIterator logicalIndex)
    (logicalExpression logicalIndex)
    (map tailPredicateView (logicalTailPredicates logicalIndex))
  where
    tailPredicateView predicate = (tailPredicateIndex predicate, tailPredicateExtent predicate)

spec :: Spec
spec = do
  describe "CPU scheduling" $ do
    it "normalizes parallel loops before reductions" $ do
      let fixture = mixedReduction
          schedule = mustSucceed $ newCpuSchedule (reductionFunction fixture) (reductionOperation fixture)
          verified = mustSucceed $ verifyCpuSchedule schedule
          plan = verifiedCpuPlan verified
          rowLoop = mustFind $ loopFor (reductionRow fixture) plan
          firstReductionLoop = mustFind $ loopFor (firstReduction fixture) plan
          secondReductionLoop = mustFind $ loopFor (secondReduction fixture) plan
          expectedLoops =
            [ LoopAxisView rowLoop (reductionRow fixture) "row" (StaticExtent 8) Parallel Nothing,
              LoopAxisView firstReductionLoop (firstReduction fixture) "first_reduction" (StaticExtent 4) Reduction Nothing,
              LoopAxisView secondReductionLoop (secondReduction fixture) "second_reduction" (StaticExtent 6) Reduction Nothing
            ]
      map loopAxisView (planLoops plan) `shouldBe` expectedLoops
      logicalIndexView <$> lookupLogicalIndex (reductionRow fixture) plan
        `shouldBe` Just (LogicalIndexView (reductionRow fixture) (LoopIndex rowLoop) [])
      computeOpId (verifiedCpuOperation verified) `shouldBe` reductionOperation fixture

    it "rewrites a split logical index without a divisible tail" $ do
      let fixture = parallel2d 16 7
          initial = mustSucceed $ newCpuSchedule (parallelFunction fixture) (parallelOperation fixture)
          firstLoop = mustFind $ cpuLoopFor (parallelFirst fixture) initial
          secondLoop = mustFind $ cpuLoopFor (parallelSecond fixture) initial
          (outer, inner, transformed) = mustSucceed $ splitCpuSchedule firstLoop 4 initial
          verified = mustSucceed $ verifyCpuSchedule transformed
          plan = verifiedCpuPlan verified
          expectedLoops =
            [ LoopAxisView outer (parallelFirst fixture) "first_outer" (StaticExtent 4) Parallel Nothing,
              LoopAxisView inner (parallelFirst fixture) "first_inner" (StaticExtent 4) Parallel Nothing,
              LoopAxisView secondLoop (parallelSecond fixture) "second" (StaticExtent 7) Parallel Nothing
            ]
          expectedIndex = LogicalIndexView (parallelFirst fixture) (splitIndex outer inner 4) []
      map loopAxisView (planLoops plan) `shouldBe` expectedLoops
      logicalIndexView <$> lookupLogicalIndex (parallelFirst fixture) plan `shouldBe` Just expectedIndex
      loopFor (parallelFirst fixture) plan `shouldBe` Nothing

    it "adds a tail predicate for a partial tile" $ do
      let fixture = parallel2d 10 7
          initial = mustSucceed $ newCpuSchedule (parallelFunction fixture) (parallelOperation fixture)
          firstLoop = mustFind $ cpuLoopFor (parallelFirst fixture) initial
          (outer, inner, transformed) = mustSucceed $ splitCpuSchedule firstLoop 4 initial
          verified = mustSucceed $ verifyCpuSchedule transformed
          plan = verifiedCpuPlan verified
          expression = splitIndex outer inner 4
          expectedIndex =
            LogicalIndexView
              (parallelFirst fixture)
              expression
              [(expression, StaticExtent 10)]
      loopExtent <$> lookupLoopAxis outer plan `shouldBe` Just (StaticExtent 3)
      loopExtent <$> lookupLoopAxis inner plan `shouldBe` Just (StaticExtent 4)
      logicalIndexView <$> lookupLogicalIndex (parallelFirst fixture) plan `shouldBe` Just expectedIndex

    it "changes the physical loop order" $ do
      let fixture = parallel2d 8 16
          initial = mustSucceed $ newCpuSchedule (parallelFunction fixture) (parallelOperation fixture)
          firstLoop = mustFind $ cpuLoopFor (parallelFirst fixture) initial
          secondLoop = mustFind $ cpuLoopFor (parallelSecond fixture) initial
          transformed = mustSucceed $ reorderCpuSchedule [secondLoop, firstLoop] initial
          verified = mustSucceed $ verifyCpuSchedule transformed
          plan = verifiedCpuPlan verified
      map loopAxisId (planLoops plan) `shouldBe` [secondLoop, firstLoop]
      logicalIndexView <$> lookupLogicalIndex (parallelFirst fixture) plan
        `shouldBe` Just (LogicalIndexView (parallelFirst fixture) (LoopIndex firstLoop) [])

  describe "CUDA scheduling" $ do
    it "constructs target limits through read-only accessors" $ do
      let block = newCudaDim3 64 32 16
          grid = newCudaDim3 1024 512 256
          target = newCudaTarget 128 block grid
          observed =
            ( cudaMaxThreadsPerBlock target,
              (cudaDimX (cudaMaxBlockDimensions target), cudaDimY (cudaMaxBlockDimensions target), cudaDimZ (cudaMaxBlockDimensions target)),
              ( cudaDimGet (cudaMaxGridDimensions target) DimensionX,
                cudaDimGet (cudaMaxGridDimensions target) DimensionY,
                cudaDimGet (cudaMaxGridDimensions target) DimensionZ
              )
            )
      observed `shouldBe` (128, (64, 32, 16), (1024, 512, 256))

    it "builds a block and thread loop plan" $ do
      let fixture = parallel2d 128 70
          target = defaultCudaTarget
          initial = mustSucceed $ newCudaSchedule (parallelFunction fixture) (parallelOperation fixture) target
          firstLoop = mustFind $ cudaLoopFor (parallelFirst fixture) initial
          secondLoop = mustFind $ cudaLoopFor (parallelSecond fixture) initial
          (firstOuter, firstInner, firstSplit) = mustSucceed $ splitCudaSchedule firstLoop 16 initial
          (secondOuter, secondInner, secondSplit) = mustSucceed $ splitCudaSchedule secondLoop 32 firstSplit
          reordered = mustSucceed $ reorderCudaSchedule [firstOuter, secondOuter, firstInner, secondInner] secondSplit
          boundBlockY = mustSucceed $ bindCudaSchedule firstOuter BlockY reordered
          boundBlockX = mustSucceed $ bindCudaSchedule secondOuter BlockX boundBlockY
          boundThreadY = mustSucceed $ bindCudaSchedule firstInner ThreadY boundBlockX
          boundThreadX = mustSucceed $ bindCudaSchedule secondInner ThreadX boundThreadY
          verified = mustSucceed $ verifyCudaSchedule boundThreadX
          plan = verifiedCudaPlan verified
          expectedLoops =
            [ LoopAxisView firstOuter (parallelFirst fixture) "first_outer" (StaticExtent 8) Parallel (Just BlockY),
              LoopAxisView secondOuter (parallelSecond fixture) "second_outer" (StaticExtent 3) Parallel (Just BlockX),
              LoopAxisView firstInner (parallelFirst fixture) "first_inner" (StaticExtent 16) Parallel (Just ThreadY),
              LoopAxisView secondInner (parallelSecond fixture) "second_inner" (StaticExtent 32) Parallel (Just ThreadX)
            ]
          secondExpression = splitIndex secondOuter secondInner 32
          expectedSecondIndex =
            LogicalIndexView
              (parallelSecond fixture)
              secondExpression
              [(secondExpression, StaticExtent 70)]
      verifiedCudaTarget verified `shouldBe` target
      map loopAxisView (planLoops plan) `shouldBe` expectedLoops
      logicalIndexView <$> lookupLogicalIndex (parallelSecond fixture) plan `shouldBe` Just expectedSecondIndex

    it "allows a dynamic grid with static threads" $ do
      let fixture = dynamicParallel1d
          initial = mustSucceed $ newCudaSchedule (dynamicFunction fixture) (dynamicOperation fixture) defaultCudaTarget
          element = mustFind $ cudaLoopFor (dynamicIterator fixture) initial
          (outer, inner, splitSchedule) = mustSucceed $ splitCudaSchedule element 32 initial
          blockBound = mustSucceed $ bindCudaSchedule outer BlockX splitSchedule
          threadBound = mustSucceed $ bindCudaSchedule inner ThreadX blockBound
          verified = mustSucceed $ verifyCudaSchedule threadBound
          plan = verifiedCudaPlan verified
          expression = splitIndex outer inner 32
          expectedIndex =
            LogicalIndexView
              (dynamicIterator fixture)
              expression
              [(expression, SymbolExtent (dynamicExtent fixture))]
      loopExtent <$> lookupLoopAxis outer plan
        `shouldBe` Just (CeilDivExtent (SymbolExtent (dynamicExtent fixture)) 32)
      loopExtent <$> lookupLoopAxis inner plan `shouldBe` Just (StaticExtent 32)
      logicalIndexView <$> lookupLogicalIndex (dynamicIterator fixture) plan `shouldBe` Just expectedIndex

    it "rejects zero target limits" $ do
      let fixture = parallel1d 1
          block = newCudaDim3 1024 1024 64
          grid = newCudaDim3 2147483647 65535 65535
          cases =
            [ (newCudaTarget 0 block grid, "max threads per block"),
              (newCudaTarget 1024 (newCudaDim3 0 1024 64) grid, "max block dimension x"),
              (newCudaTarget 1024 (newCudaDim3 1024 0 64) grid, "max block dimension y"),
              (newCudaTarget 1024 (newCudaDim3 1024 1024 0) grid, "max block dimension z"),
              (newCudaTarget 1024 block (newCudaDim3 0 65535 65535), "max grid dimension x"),
              (newCudaTarget 1024 block (newCudaDim3 2147483647 0 65535), "max grid dimension y"),
              (newCudaTarget 1024 block (newCudaDim3 2147483647 65535 0), "max grid dimension z")
            ]
      map
        (\(target, _) -> newCudaSchedule (parallelFunction fixture) (parallelOperation fixture) target)
        cases
        `shouldBe` map (Left . InvalidCudaTargetLimit . snd) cases

    it "rejects duplicate bindings" $ do
      let fixture = parallel2d 8 16
          initial = mustSucceed $ newCudaSchedule (parallelFunction fixture) (parallelOperation fixture) defaultCudaTarget
          firstLoop = mustFind $ cudaLoopFor (parallelFirst fixture) initial
          secondLoop = mustFind $ cudaLoopFor (parallelSecond fixture) initial
          firstBound = mustSucceed $ bindCudaSchedule firstLoop ThreadX initial
      bindCudaSchedule firstLoop ThreadY firstBound
        `shouldBe` Left (LoopAlreadyBound "first" ThreadX)
      bindCudaSchedule secondLoop ThreadX firstBound
        `shouldBe` Left (BindingAlreadyUsed ThreadX "first")

    it "rejects a dynamic thread extent" $ do
      let fixture = dynamicParallel1d
          initial = mustSucceed $ newCudaSchedule (dynamicFunction fixture) (dynamicOperation fixture) defaultCudaTarget
          element = mustFind $ cudaLoopFor (dynamicIterator fixture) initial
          bound = mustSucceed $ bindCudaSchedule element ThreadX initial
      verifyCudaSchedule bound `shouldBe` Left (DynamicCudaThreadExtent "element")

    it "enforces block and grid dimension limits" $ do
      let fixture = parallel1d 33
          threadTarget = newCudaTarget 1024 (newCudaDim3 32 1024 64) (newCudaDim3 1024 1024 1024)
          threadInitial = mustSucceed $ newCudaSchedule (parallelFunction fixture) (parallelOperation fixture) threadTarget
          threadElement = mustFind $ cudaLoopFor (parallelFirst fixture) threadInitial
          threadBound = mustSucceed $ bindCudaSchedule threadElement ThreadX threadInitial
          blockTarget = newCudaTarget 1024 (newCudaDim3 1024 1024 64) (newCudaDim3 32 1024 1024)
          blockInitial = mustSucceed $ newCudaSchedule (parallelFunction fixture) (parallelOperation fixture) blockTarget
          blockElement = mustFind $ cudaLoopFor (parallelFirst fixture) blockInitial
          blockBound = mustSucceed $ bindCudaSchedule blockElement BlockX blockInitial
      verifyCudaSchedule threadBound `shouldBe` Left (CudaDimensionExceeded ThreadX 33 32)
      verifyCudaSchedule blockBound `shouldBe` Left (CudaDimensionExceeded BlockX 33 32)

    it "enforces total threads per block" $ do
      let fixture = parallel2d 32 16
          target = newCudaTarget 256 (newCudaDim3 1024 1024 64) (newCudaDim3 1024 1024 1024)
          initial = mustSucceed $ newCudaSchedule (parallelFunction fixture) (parallelOperation fixture) target
          firstLoop = mustFind $ cudaLoopFor (parallelFirst fixture) initial
          secondLoop = mustFind $ cudaLoopFor (parallelSecond fixture) initial
          firstBound = mustSucceed $ bindCudaSchedule firstLoop ThreadX initial
          secondBound = mustSucceed $ bindCudaSchedule secondLoop ThreadY firstBound
      verifyCudaSchedule secondBound `shouldBe` Left (CudaThreadsPerBlockExceeded 512 256)

  describe "invalid transformations" $ do
    it "rejects zero, stale, and foreign loops in precedence order" $ do
      let fixture = parallel2d 16 8
          firstSchedule = mustSucceed $ newCpuSchedule (parallelFunction fixture) (parallelOperation fixture)
          secondSchedule = mustSucceed $ newCpuSchedule (parallelFunction fixture) (parallelOperation fixture)
          original = mustFind $ cpuLoopFor (parallelFirst fixture) firstSchedule
          foreignLoop = mustFind $ cpuLoopFor (parallelFirst fixture) secondSchedule
          (_, _, splitSchedule) = mustSucceed $ splitCpuSchedule original 4 firstSchedule
      splitCpuSchedule original 0 firstSchedule `shouldBe` Left ZeroSplitFactor
      splitCpuSchedule original 2 splitSchedule `shouldBe` Left (UnknownLoop original)
      splitCpuSchedule foreignLoop 2 splitSchedule `shouldBe` Left (ForeignLoop foreignLoop)

    it "does not split, reorder, or bind strict reduction loops" $ do
      let fixture = mixedReduction
          splitInitial = mustSucceed $ newCpuSchedule (reductionFunction fixture) (reductionOperation fixture)
          splitReduction = mustFind $ cpuLoopFor (firstReduction fixture) splitInitial
          reorderInitial = mustSucceed $ newCpuSchedule (reductionFunction fixture) (reductionOperation fixture)
          rowLoop = mustFind $ cpuLoopFor (reductionRow fixture) reorderInitial
          firstReductionLoop = mustFind $ cpuLoopFor (firstReduction fixture) reorderInitial
          secondReductionLoop = mustFind $ cpuLoopFor (secondReduction fixture) reorderInitial
          cudaInitial = mustSucceed $ newCudaSchedule (reductionFunction fixture) (reductionOperation fixture) defaultCudaTarget
          cudaReduction = mustFind $ cudaLoopFor (firstReduction fixture) cudaInitial
      splitCpuSchedule splitReduction 2 splitInitial
        `shouldBe` Left (ReductionSplitUnsupported "first_reduction")
      reorderCpuSchedule [rowLoop, secondReductionLoop, firstReductionLoop] reorderInitial
        `shouldBe` Left ReductionReorderUnsupported
      reorderCpuSchedule [firstReductionLoop, rowLoop, secondReductionLoop] reorderInitial
        `shouldBe` Left ParallelLoopInsideReduction
      bindCudaSchedule cudaReduction ThreadX cudaInitial
        `shouldBe` Left (ReductionBindUnsupported "first_reduction")

    it "checks Word64 overflow in nested split divisors" $ do
      let fixture = dynamicParallel1d
          initial = mustSucceed $ newCpuSchedule (dynamicFunction fixture) (dynamicOperation fixture)
          element = mustFind $ cpuLoopFor (dynamicIterator fixture) initial
          (outer, _, splitSchedule) = mustSucceed $ splitCpuSchedule element maxBound initial
      splitCpuSchedule outer 2 splitSchedule `shouldBe` Left (ArithmeticOverflow "split loop divisor")

mustFind :: Maybe value -> value
mustFind = fromMaybeError "fixture lookup failed"

fromMaybeError :: String -> Maybe value -> value
fromMaybeError message optional = case optional of
  Just value -> value
  Nothing -> error message
