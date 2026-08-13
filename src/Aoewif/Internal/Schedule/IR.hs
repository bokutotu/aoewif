module Aoewif.Internal.Schedule.IR (
    LoopId (..),
    IndexExpr (..),
    Predicate (..),
    AxisBinding (..),
    ExecutionKind (..),
    CudaBinding (..),
    LoopDim (..),
    ScheduleNode (..),
    ScheduleIR (..),
    initialScheduleIR,
    splitScheduleIR,
    reorderScheduleIR,
    parallelScheduleIR,
    unrollScheduleIR,
    bindCudaScheduleIR,
) where

import qualified Aoewif.Internal.Compute.IR as Compute
import           Data.List                  (find)
import           Data.Maybe                 (fromJust)
import           Data.Word                  (Word64)

newtype LoopId = LoopId Int
    deriving stock (Eq, Ord, Show)

data IndexExpr
    = LoopIndex LoopId
    | ConstantIndex Word64
    | AddIndex IndexExpr IndexExpr
    | MulIndex IndexExpr IndexExpr
    deriving stock (Eq, Ord, Show)

data Predicate
    = IndexLessThan IndexExpr Compute.DimExpr
    deriving stock (Eq, Show)

data AxisBinding = AxisBinding
    { bindingAxis  :: Compute.AxisId
    , bindingIndex :: IndexExpr
    }
    deriving stock (Eq, Show)

data ExecutionKind
    = Serial
    | Parallel
    deriving stock (Eq, Show)

data CudaBinding
    = BlockX
    | BlockY
    | BlockZ
    | ThreadX
    | ThreadY
    | ThreadZ
    deriving stock (Eq, Ord, Show)

data LoopDim = LoopDim
    { loopId           :: LoopId
    , loopName         :: Compute.Name
    , loopLowerBound   :: Compute.DimExpr
    , loopExtent       :: Compute.DimExpr
    , loopExecution    :: ExecutionKind
    , loopUnrollFactor :: Maybe Word64
    , loopCudaBinding  :: Maybe CudaBinding
    }
    deriving stock (Eq, Show)

data ScheduleNode
    = Band [LoopDim] ScheduleNode
    | Sequence [ScheduleNode]
    | Guard Predicate ScheduleNode
    | Leaf Compute.BlockId [AxisBinding]
    deriving stock (Eq, Show)

newtype ScheduleIR = ScheduleIR
    { scheduleRoot :: ScheduleNode
    }
    deriving stock (Eq, Show)

initialScheduleIR :: Compute.ComputeIR -> (ScheduleIR, [(Compute.BlockId, Compute.AxisId, LoopId)], Int)
initialScheduleIR computeIR =
    ( ScheduleIR
        { scheduleRoot = Sequence nodes
        }
    , handles
    , nextLoop
    )
  where
    (nodes, handles, nextLoop) = buildBlocks 0 (Compute.computeBlocks computeIR)

buildBlocks :: Int -> [Compute.ComputeBlock] -> ([ScheduleNode], [(Compute.BlockId, Compute.AxisId, LoopId)], Int)
buildBlocks next [] = ([], [], next)
buildBlocks next (computeBlock : rest) =
    let axes = Compute.blockAxes computeBlock
        loopIdentifiers = map LoopId [next .. next + length axes - 1]
        dimensions = zipWith newLoopDim loopIdentifiers axes
        bindings = zipWith (AxisBinding . Compute.axisId) axes (map LoopIndex loopIdentifiers)
        node = Band dimensions (Leaf (Compute.blockId computeBlock) bindings)
        blockHandles = zipWith (\axisDecl identifier -> (Compute.blockId computeBlock, Compute.axisId axisDecl, identifier)) axes loopIdentifiers
        (remainingNodes, remainingHandles, finalNext) = buildBlocks (next + length axes) rest
     in (node : remainingNodes, blockHandles ++ remainingHandles, finalNext)

newLoopDim :: LoopId -> Compute.AxisDecl -> LoopDim
newLoopDim identifier axisDecl =
    LoopDim
        { loopId = identifier
        , loopName = Compute.axisName axisDecl
        , loopLowerBound = Compute.axisLower axisDecl
        , loopExtent = Compute.axisExtent axisDecl
        , loopExecution = Serial
        , loopUnrollFactor = Nothing
        , loopCudaBinding = Nothing
        }

splitScheduleIR :: Compute.BlockId -> LoopId -> Word64 -> LoopId -> LoopId -> ScheduleIR -> ScheduleIR
splitScheduleIR blockIdentifier target factor outerIdentifier innerIdentifier scheduleIR =
    scheduleIR
        { scheduleRoot = splitBlock blockIdentifier target factor outerIdentifier innerIdentifier (scheduleRoot scheduleIR)
        }

splitBlock :: Compute.BlockId -> LoopId -> Word64 -> LoopId -> LoopId -> ScheduleNode -> ScheduleNode
splitBlock blockIdentifier target factor outerIdentifier innerIdentifier node = case node of
    Band dimensions child
        | any ((== target) . loopId) dimensions && containsBlock blockIdentifier child ->
            let original = loopDimFor target node
                replacement = AddIndex (MulIndex (LoopIndex outerIdentifier) (ConstantIndex factor)) (LoopIndex innerIdentifier)
                outer =
                    original
                        { loopId = outerIdentifier
                        , loopName = appendName (loopName original) "_outer"
                        , loopExtent = Compute.CeilDivDim (loopExtent original) factor
                        }
                inner =
                    original
                        { loopId = innerIdentifier
                        , loopName = appendName (loopName original) "_inner"
                        , loopLowerBound = Compute.StaticDim 0
                        , loopExtent = Compute.StaticDim factor
                        , loopExecution = Serial
                        , loopUnrollFactor = Nothing
                        , loopCudaBinding = Nothing
                        }
                rewritten = replaceLoopNode target replacement outer inner node
             in if needsTailGuard factor (loopExtent original)
                    then addGuard (IndexLessThan replacement (loopExtent original)) rewritten
                    else rewritten
        | otherwise -> Band dimensions (splitBlock blockIdentifier target factor outerIdentifier innerIdentifier child)
    Sequence children -> Sequence (map (splitBlock blockIdentifier target factor outerIdentifier innerIdentifier) children)
    Guard predicate child -> Guard predicate (splitBlock blockIdentifier target factor outerIdentifier innerIdentifier child)
    Leaf _ _ -> node

reorderScheduleIR :: Compute.BlockId -> [LoopId] -> ScheduleIR -> ScheduleIR
reorderScheduleIR blockIdentifier order scheduleIR =
    scheduleIR
        { scheduleRoot = reorderBlock blockIdentifier order (scheduleRoot scheduleIR)
        }

parallelScheduleIR :: LoopId -> ScheduleIR -> ScheduleIR
parallelScheduleIR identifier scheduleIR =
    scheduleIR
        { scheduleRoot = mapLoop identifier (\dimension -> dimension{loopExecution = Parallel}) (scheduleRoot scheduleIR)
        }

unrollScheduleIR :: LoopId -> Word64 -> ScheduleIR -> ScheduleIR
unrollScheduleIR identifier factor scheduleIR =
    scheduleIR
        { scheduleRoot = mapLoop identifier (\dimension -> dimension{loopUnrollFactor = Just factor}) (scheduleRoot scheduleIR)
        }

bindCudaScheduleIR :: LoopId -> CudaBinding -> ScheduleIR -> ScheduleIR
bindCudaScheduleIR identifier cudaBinding scheduleIR =
    scheduleIR
        { scheduleRoot = mapLoop identifier (\dimension -> dimension{loopCudaBinding = Just cudaBinding}) (scheduleRoot scheduleIR)
        }

reorderBlock :: Compute.BlockId -> [LoopId] -> ScheduleNode -> ScheduleNode
reorderBlock blockIdentifier order node = case node of
    Band dimensions child
        | containsBlock blockIdentifier child ->
            Band (map (dimensionFor dimensions) order) (reorderBlock blockIdentifier order child)
        | otherwise -> Band dimensions (reorderBlock blockIdentifier order child)
    Sequence children -> Sequence (map (reorderBlock blockIdentifier order) children)
    Guard predicate child -> Guard predicate (reorderBlock blockIdentifier order child)
    Leaf _ _ -> node

dimensionFor :: [LoopDim] -> LoopId -> LoopDim
dimensionFor dimensions identifier = fromJust (find ((== identifier) . loopId) dimensions)

replaceLoopNode :: LoopId -> IndexExpr -> LoopDim -> LoopDim -> ScheduleNode -> ScheduleNode
replaceLoopNode target replacement outer inner node = case node of
    Band dimensions child ->
        Band (concatMap replaceDimension dimensions) (replaceLoopNode target replacement outer inner child)
      where
        replaceDimension dimension
            | loopId dimension == target = [outer, inner]
            | otherwise = [dimension]
    Sequence children -> Sequence (map (replaceLoopNode target replacement outer inner) children)
    Guard predicate child -> Guard (replacePredicate target replacement predicate) (replaceLoopNode target replacement outer inner child)
    Leaf blockIdentifier bindings -> Leaf blockIdentifier (map replaceBinding bindings)
      where
        replaceBinding binding = binding{bindingIndex = replaceIndex target replacement (bindingIndex binding)}

mapLoop :: LoopId -> (LoopDim -> LoopDim) -> ScheduleNode -> ScheduleNode
mapLoop target transform node = case node of
    Band dimensions child -> Band (map update dimensions) (mapLoop target transform child)
      where
        update dimension
            | loopId dimension == target = transform dimension
            | otherwise = dimension
    Sequence children -> Sequence (map (mapLoop target transform) children)
    Guard predicate child -> Guard predicate (mapLoop target transform child)
    Leaf _ _ -> node

replacePredicate :: LoopId -> IndexExpr -> Predicate -> Predicate
replacePredicate target replacement (IndexLessThan indexExpression extent) =
    IndexLessThan (replaceIndex target replacement indexExpression) extent

replaceIndex :: LoopId -> IndexExpr -> IndexExpr -> IndexExpr
replaceIndex target replacement expression = case expression of
    LoopIndex identifier
        | identifier == target -> replacement
        | otherwise -> expression
    ConstantIndex _ -> expression
    AddIndex lhs rhs -> AddIndex (replaceIndex target replacement lhs) (replaceIndex target replacement rhs)
    MulIndex lhs rhs -> MulIndex (replaceIndex target replacement lhs) (replaceIndex target replacement rhs)

addGuard :: Predicate -> ScheduleNode -> ScheduleNode
addGuard predicate node = case node of
    Band dimensions child -> Band dimensions (addGuard predicate child)
    Sequence children     -> Sequence (map (addGuard predicate) children)
    Guard existing child  -> Guard existing (addGuard predicate child)
    Leaf _ _              -> Guard predicate node

loopDimFor :: LoopId -> ScheduleNode -> LoopDim
loopDimFor identifier = fromJust . find ((== identifier) . loopId) . allLoopDims

allLoopDims :: ScheduleNode -> [LoopDim]
allLoopDims node = case node of
    Band dimensions child -> dimensions ++ allLoopDims child
    Sequence children     -> concatMap allLoopDims children
    Guard _ child         -> allLoopDims child
    Leaf _ _              -> []

containsBlock :: Compute.BlockId -> ScheduleNode -> Bool
containsBlock blockIdentifier node = case node of
    Band _ child      -> containsBlock blockIdentifier child
    Sequence children -> any (containsBlock blockIdentifier) children
    Guard _ child     -> containsBlock blockIdentifier child
    Leaf identifier _ -> identifier == blockIdentifier

needsTailGuard :: Word64 -> Compute.DimExpr -> Bool
needsTailGuard divisor extent = case extent of
    Compute.StaticDim value -> value `mod` divisor /= 0
    Compute.SymbolDim _     -> True
    Compute.CeilDivDim _ _  -> True

appendName :: Compute.Name -> String -> Compute.Name
appendName (Compute.Name name) suffix = Compute.Name (name ++ suffix)
