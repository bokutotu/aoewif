module Aoewif.Target.Cuda.Sm90.DSL (
    bulkCommitGroup,
    bulkWaitGroup,
    clusterBarrierArrive,
    clusterBarrierWait,
    declareWgmmaFragment,
    electSync,
    fenceMBarrierInit,
    fenceProxyAsync,
    getCtaRank,
    mBarrierArrive,
    mBarrierArriveExpectTx,
    mBarrierArriveExpectTxRemote,
    mBarrierArriveRemote,
    mBarrierExpectTx,
    mBarrierExpectTxRemote,
    mBarrierInit,
    mBarrierTryWaitParity,
    mapSharedCluster,
    readClusterSpecialRegister,
    setMaxNReg,
    tmaTensorLoad,
    tmaTensorStore,
    wgmmaCommitGroup,
    wgmmaFence,
    wgmmaMmaAsync,
    wgmmaWaitGroup,
    zeroWgmmaFragment,
) where

import           Aoewif.Target.Cuda.DSL              (Block, Type, declare,
                                                      emit, int, (.=))
import           Aoewif.Target.Cuda.Sm90.Instruction
import           Aoewif.Target.Cuda.Sm90.Render      ()
import           Aoewif.Target.Cuda.Syntax           (Expr, Stmt (Op))
import           Aoewif.Target.Cuda.TensorCoreOp     (TensorCoreOp (TensorCoreOp))

declareWgmmaFragment :: Type -> String -> Int -> Block WgmmaFragment
declareWgmmaFragment registerType prefix registerCount = WgmmaFragment <$> mapM declareRegister [0 .. registerCount - 1]
  where
    declareRegister index = declare registerType (prefix ++ show index)

zeroWgmmaFragment :: WgmmaFragment -> Block ()
zeroWgmmaFragment = mapM_ (.= int 0) . wgmmaFragmentRegisters

wgmmaMmaAsync :: WgmmaMma -> Block ()
wgmmaMmaAsync = emitSm90 . WgmmaMmaAsync

wgmmaFence :: WgmmaMma -> Block ()
wgmmaFence = emitSm90 . WgmmaFence

wgmmaCommitGroup :: Block ()
wgmmaCommitGroup = emitSm90 WgmmaCommitGroup

wgmmaWaitGroup :: Int -> Block ()
wgmmaWaitGroup = emitSm90 . WgmmaWaitGroup

tmaTensorLoad :: TmaLoadDestination -> TmaTensorMap -> TmaCoordinates -> MBarrier -> Maybe TmaCachePolicy -> Block ()
tmaTensorLoad destination tensorMap coordinates barrier cachePolicy =
    emitSm90
        ( TmaTensor
            (TmaTensorLoad destination tensorMap coordinates barrier cachePolicy)
        )

tmaTensorStore :: TmaTensorMap -> TmaCoordinates -> Expr -> Maybe TmaCachePolicy -> Block ()
tmaTensorStore tensorMap coordinates source cachePolicy =
    emitSm90 (TmaTensor (TmaTensorStore tensorMap coordinates source cachePolicy))

bulkCommitGroup :: Block ()
bulkCommitGroup = emitSm90 BulkCommitGroup

bulkWaitGroup :: BulkWaitMode -> Int -> Block ()
bulkWaitGroup mode = emitSm90 . BulkWaitGroup mode

mBarrierInit :: MBarrier -> Expr -> Block ()
mBarrierInit barrier = emitSm90 . MBarrierInstruction . MBarrierInit barrier

mBarrierArrive :: Maybe Expr -> MBarrier -> Maybe Expr -> Block ()
mBarrierArrive destination barrier = emitSm90 . MBarrierInstruction . MBarrierArrive destination barrier

mBarrierArriveRemote :: ClusterAddressWidth -> MBarrier -> Maybe Expr -> Block ()
mBarrierArriveRemote width barrier = emitSm90 . MBarrierInstruction . MBarrierArriveRemote width barrier

mBarrierArriveExpectTx :: Maybe Expr -> MBarrier -> Expr -> Block ()
mBarrierArriveExpectTx destination barrier transactionCount =
    emitSm90
        ( MBarrierInstruction
            (MBarrierArriveExpectTx destination barrier transactionCount)
        )

mBarrierArriveExpectTxRemote :: ClusterAddressWidth -> MBarrier -> Expr -> Block ()
mBarrierArriveExpectTxRemote width barrier =
    emitSm90 . MBarrierInstruction . MBarrierArriveExpectTxRemote width barrier

mBarrierExpectTx :: MBarrier -> Expr -> Block ()
mBarrierExpectTx barrier =
    emitSm90 . MBarrierInstruction . MBarrierExpectTx barrier

mBarrierExpectTxRemote :: ClusterAddressWidth -> MBarrier -> Expr -> Block ()
mBarrierExpectTxRemote width barrier =
    emitSm90 . MBarrierInstruction . MBarrierExpectTxRemote width barrier

mBarrierTryWaitParity :: Expr -> MBarrier -> Expr -> Maybe Expr -> Block ()
mBarrierTryWaitParity destination barrier parity =
    emitSm90 . MBarrierInstruction . MBarrierTryWaitParity destination barrier parity

fenceProxyAsync :: SharedScope -> Block ()
fenceProxyAsync = emitSm90 . FenceProxyAsync

fenceMBarrierInit :: Block ()
fenceMBarrierInit = emitSm90 FenceMBarrierInit

electSync :: ElectDestination -> Expr -> Block ()
electSync destination = emitSm90 . ElectSync destination

setMaxNReg :: RegisterAdjustment -> Int -> Block ()
setMaxNReg adjustment = emitSm90 . SetMaxNReg adjustment

clusterBarrierArrive :: ClusterBarrierArrival -> Block ()
clusterBarrierArrive = emitSm90 . ClusterBarrierArrive

clusterBarrierWait :: Block ()
clusterBarrierWait = emitSm90 ClusterBarrierWait

readClusterSpecialRegister :: Expr -> ClusterSpecialRegister -> Block ()
readClusterSpecialRegister destination = emitSm90 . ReadClusterSpecialRegister destination

mapSharedCluster :: ClusterAddressWidth -> Expr -> Expr -> Expr -> Block ()
mapSharedCluster width destination source = emitSm90 . MapSharedCluster width destination source

getCtaRank :: ClusterAddressWidth -> Expr -> Expr -> Block ()
getCtaRank width destination = emitSm90 . GetCtaRank width destination

emitSm90 :: Sm90Op -> Block ()
emitSm90 = emit . Op . TensorCoreOp
