{-# LANGUAGE DataKinds       #-}
{-# LANGUAGE RoleAnnotations #-}

module Aoewif.Cuda (
    Name (..),
    Builder,
    BuildError (..),
    Buffer,
    Access (..),
    Extent,
    Index,
    Predicate,
    F32,
    kernel,
    staticExtent,
    dynamicExtent,
    ceilDiv,
    input,
    output,
    launch1D,
    serial,
    extentIndex,
    lessThan,
    when,
    shared,
    syncThreads,
    load,
    store,
    indexLiteral,
    addIndex,
    multiplyIndex,
    f32,
    add,
    sub,
    mul,
    divide,
) where

import           Aoewif.Cuda.IR (Access (..), Name (..))
import qualified Aoewif.Cuda.IR as IR
import           Data.Char      (isAsciiLower, isAsciiUpper, isDigit)
import           Data.Word      (Word32, Word64)

data BuildError
    = InvalidIdentifier Name
    | DuplicateIdentifier Name
    | InvalidRank Name
    | RankMismatch Name Int Int
    | NonPositiveExtent String
    | ZeroCeilDivisor
    | NonPositiveThreadCount
    | ThreadCountExceedsLimit Word32
    | MissingLaunch
    | MultipleLaunches
    | DeclarationInsideLaunch Name
    | DeclarationAfterLaunch Name
    | OperationOutsideLaunch String
    | SharedInsideSerial Name
    | SharedInsideConditional Name
    | SyncThreadsInsideConditional
    | SharedSizeOverflow Name
    deriving stock (Eq, Show)

data Buffer scope (access :: Access) = Buffer IR.BufferId Name [IR.Extent]

type role Buffer nominal nominal

newtype Extent scope = Extent IR.Extent

type role Extent nominal

newtype Index scope = Index IR.IndexExpr

type role Index nominal

newtype Predicate scope = Predicate IR.Predicate

type role Predicate nominal

newtype F32 scope = F32 IR.F32Expr

type role F32 nominal

data BuildLocation
    = TopLevel
    | LaunchBody Int Int

data BuildState = BuildState
    { stateSymbols         :: [IR.Symbol]
    , stateGlobalBuffers   :: [IR.BufferDecl]
    , stateLaunch          :: Maybe (IR.Launch, [IR.Statement])
    , stateStatements      :: [IR.Statement]
    , stateUsedIdentifiers :: [Name]
    , stateLocation        :: BuildLocation
    , stateNextSymbol      :: !Int
    , stateNextBuffer      :: !Int
    , stateNextValue       :: !Int
    , stateNextLoop        :: !Int
    }

newtype Builder scope value = Builder
    { runBuilder :: BuildState -> Either BuildError (value, BuildState)
    }

type role Builder nominal representational

instance Functor (Builder scope) where
    fmap transform (Builder action) = Builder $ \state -> do
        (value, nextState) <- action state
        pure (transform value, nextState)

instance Applicative (Builder scope) where
    pure value = Builder $ \state -> Right (value, state)
    Builder functionAction <*> Builder valueAction = Builder $ \state -> do
        (function, functionState) <- functionAction state
        (value, valueState) <- valueAction functionState
        pure (function value, valueState)

instance Monad (Builder scope) where
    Builder action >>= next = Builder $ \state -> do
        (value, nextState) <- action state
        runBuilder (next value) nextState

kernel :: Name -> (forall scope. Builder scope ()) -> Either BuildError IR.Kernel
kernel name build = do
    checkIdentifier name
    (_, finalState) <- runBuilder build (initialState name)
    case stateLaunch finalState of
        Nothing -> Left MissingLaunch
        Just (launch, body) ->
            Right
                IR.Kernel
                    { IR.kernelName = name
                    , IR.kernelSymbols = stateSymbols finalState
                    , IR.kernelBuffers = stateGlobalBuffers finalState
                    , IR.kernelLaunch = launch
                    , IR.kernelBody = body
                    }

staticExtent :: Word64 -> Extent scope
staticExtent = Extent . IR.StaticExtent

dynamicExtent :: Name -> Builder scope (Extent scope)
dynamicExtent name = Builder $ \state -> do
    declarationState <- prepareDeclaration name state
    let identifier = IR.SymbolId (stateNextSymbol declarationState)
        symbol = IR.Symbol identifier name
    pure
        ( Extent (IR.DynamicExtent identifier)
        , declarationState
            { stateSymbols = stateSymbols declarationState ++ [symbol]
            , stateNextSymbol = stateNextSymbol declarationState + 1
            }
        )

ceilDiv :: Extent scope -> Word64 -> Builder scope (Extent scope)
ceilDiv (Extent dividend) divisor = Builder $ \state ->
    if divisor == 0
        then Left ZeroCeilDivisor
        else Right (Extent (IR.CeilDivExtent dividend divisor), state)

input :: Name -> [Extent scope] -> Builder scope (Buffer scope 'ReadOnly)
input = declareBuffer ReadOnly

output :: Name -> [Extent scope] -> Builder scope (Buffer scope 'ReadWrite)
output = declareBuffer ReadWrite

launch1D :: Extent scope -> Word32 -> (Index scope -> Index scope -> Builder scope ()) -> Builder scope ()
launch1D (Extent gridExtent) threads build = Builder $ \state -> do
    ensureTopLevelLaunch state
    checkExtent "launch grid" gridExtent
    if threads == 0
        then Left NonPositiveThreadCount
        else
            if threads > 1024
                then Left (ThreadCountExceedsLimit threads)
                else do
                    let bodyInitialState =
                            state
                                { stateStatements = []
                                , stateLocation = LaunchBody 0 0
                                }
                        blockIndex = Index IR.BlockIndexX
                        threadIndex = Index IR.ThreadIndexX
                    (_, bodyFinalState) <- runBuilder (build blockIndex threadIndex) bodyInitialState
                    pure
                        ( ()
                        , bodyFinalState
                            { stateLaunch = Just (IR.Launch gridExtent threads, stateStatements bodyFinalState)
                            , stateStatements = stateStatements state
                            , stateLocation = TopLevel
                            }
                        )

serial :: Extent scope -> (Index scope -> Builder scope ()) -> Builder scope ()
serial (Extent extent) build = Builder $ \state -> do
    (serialDepth, conditionalDepth) <- launchDepths "serial" state
    checkExtent "serial loop" extent
    let identifier = IR.LoopId (stateNextLoop state)
        bodyInitialState =
            state
                { stateStatements = []
                , stateLocation = LaunchBody (serialDepth + 1) conditionalDepth
                , stateNextLoop = stateNextLoop state + 1
                }
    (_, bodyFinalState) <- runBuilder (build (Index (IR.LoopIndex identifier))) bodyInitialState
    pure
        ( ()
        , bodyFinalState
            { stateStatements =
                stateStatements state
                    ++ [IR.SerialFor identifier extent (stateStatements bodyFinalState)]
            , stateLocation = stateLocation state
            }
        )

extentIndex :: Extent scope -> Index scope
extentIndex (Extent extent) = Index (IR.ExtentIndex extent)

lessThan :: Index scope -> Index scope -> Predicate scope
lessThan (Index lhs) (Index rhs) = Predicate (IR.IndexLessThan lhs rhs)

when :: Predicate scope -> Builder scope () -> Builder scope ()
when (Predicate predicate) build = Builder $ \state -> do
    (serialDepth, conditionalDepth) <- launchDepths "when" state
    let bodyInitialState =
            state
                { stateStatements = []
                , stateLocation = LaunchBody serialDepth (conditionalDepth + 1)
                }
    (_, bodyFinalState) <- runBuilder build bodyInitialState
    pure
        ( ()
        , bodyFinalState
            { stateStatements =
                stateStatements state
                    ++ [IR.IfThen predicate (stateStatements bodyFinalState)]
            , stateLocation = stateLocation state
            }
        )

shared :: Name -> [Word64] -> (Buffer scope 'ReadWrite -> Builder scope ()) -> Builder scope ()
shared name staticShape build = Builder $ \state -> do
    (serialDepth, conditionalDepth) <- launchDepths "shared" state
    case () of
        _
            | serialDepth /= 0 -> Left (SharedInsideSerial name)
            | conditionalDepth /= 0 -> Left (SharedInsideConditional name)
            | otherwise -> pure ()
    checkIdentifier name
    checkUnusedIdentifier name state
    checkSharedShape name staticShape
    let identifier = IR.BufferId (stateNextBuffer state)
        shape = map IR.StaticExtent staticShape
        declaration = IR.SharedDecl identifier name staticShape
        buffer = Buffer identifier name shape
        bodyInitialState =
            state
                { stateStatements = []
                , stateUsedIdentifiers = stateUsedIdentifiers state ++ [name]
                , stateNextBuffer = stateNextBuffer state + 1
                }
    (_, bodyFinalState) <- runBuilder (build buffer) bodyInitialState
    pure
        ( ()
        , bodyFinalState
            { stateStatements =
                stateStatements state
                    ++ [IR.AllocateShared declaration (stateStatements bodyFinalState)]
            , stateLocation = stateLocation state
            }
        )

syncThreads :: Builder scope ()
syncThreads = Builder $ \state -> do
    (_, conditionalDepth) <- launchDepths "syncThreads" state
    if conditionalDepth /= 0
        then Left SyncThreadsInsideConditional
        else
            pure
                ( ()
                , state{stateStatements = stateStatements state ++ [IR.SyncThreads]}
                )

load :: Buffer scope access -> [Index scope] -> Builder scope (F32 scope)
load (Buffer identifier name shape) indices = Builder $ \state -> do
    _ <- launchDepths "load" state
    address <- bufferAddress name shape indices
    let valueIdentifier = IR.ValueId (stateNextValue state)
        statement = IR.LetF32 valueIdentifier (IR.LoadF32 identifier address)
    pure
        ( F32 (IR.F32Value valueIdentifier)
        , state
            { stateStatements = stateStatements state ++ [statement]
            , stateNextValue = stateNextValue state + 1
            }
        )

store :: Buffer scope 'ReadWrite -> [Index scope] -> F32 scope -> Builder scope ()
store (Buffer identifier name shape) indices (F32 value) = Builder $ \state -> do
    _ <- launchDepths "store" state
    address <- bufferAddress name shape indices
    pure
        ( ()
        , state
            { stateStatements =
                stateStatements state ++ [IR.StoreF32 identifier address value]
            }
        )

indexLiteral :: Word64 -> Index scope
indexLiteral = Index . IR.ConstantIndex

addIndex :: Index scope -> Index scope -> Index scope
addIndex (Index lhs) (Index rhs) = Index (IR.AddIndex lhs rhs)

multiplyIndex :: Index scope -> Index scope -> Index scope
multiplyIndex (Index lhs) (Index rhs) = Index (IR.MulIndex lhs rhs)

f32 :: Float -> F32 scope
f32 = F32 . IR.F32Literal

add :: F32 scope -> F32 scope -> F32 scope
add (F32 lhs) (F32 rhs) = F32 (IR.AddF32 lhs rhs)

sub :: F32 scope -> F32 scope -> F32 scope
sub (F32 lhs) (F32 rhs) = F32 (IR.SubF32 lhs rhs)

mul :: F32 scope -> F32 scope -> F32 scope
mul (F32 lhs) (F32 rhs) = F32 (IR.MulF32 lhs rhs)

divide :: F32 scope -> F32 scope -> F32 scope
divide (F32 lhs) (F32 rhs) = F32 (IR.DivF32 lhs rhs)

initialState :: Name -> BuildState
initialState name =
    BuildState
        { stateSymbols = []
        , stateGlobalBuffers = []
        , stateLaunch = Nothing
        , stateStatements = []
        , stateUsedIdentifiers = [name]
        , stateLocation = TopLevel
        , stateNextSymbol = 0
        , stateNextBuffer = 0
        , stateNextValue = 0
        , stateNextLoop = 0
        }

declareBuffer :: Access -> Name -> [Extent scope] -> Builder scope (Buffer scope access)
declareBuffer access name extents = Builder $ \state -> do
    declarationState <- prepareDeclaration name state
    let shape = map (\(Extent extent) -> extent) extents
    checkShape name shape
    let identifier = IR.BufferId (stateNextBuffer declarationState)
        declaration = IR.BufferDecl identifier name access shape
    pure
        ( Buffer identifier name shape
        , declarationState
            { stateGlobalBuffers = stateGlobalBuffers declarationState ++ [declaration]
            , stateNextBuffer = stateNextBuffer declarationState + 1
            }
        )

prepareDeclaration :: Name -> BuildState -> Either BuildError BuildState
prepareDeclaration name state = do
    case stateLocation state of
        LaunchBody _ _ -> Left (DeclarationInsideLaunch name)
        TopLevel -> case stateLaunch state of
            Just _  -> Left (DeclarationAfterLaunch name)
            Nothing -> pure ()
    checkIdentifier name
    checkUnusedIdentifier name state
    pure state{stateUsedIdentifiers = stateUsedIdentifiers state ++ [name]}

ensureTopLevelLaunch :: BuildState -> Either BuildError ()
ensureTopLevelLaunch state = case stateLocation state of
    LaunchBody _ _ -> Left MultipleLaunches
    TopLevel -> case stateLaunch state of
        Nothing -> Right ()
        Just _  -> Left MultipleLaunches

launchDepths :: String -> BuildState -> Either BuildError (Int, Int)
launchDepths operation state = case stateLocation state of
    TopLevel -> Left (OperationOutsideLaunch operation)
    LaunchBody serialDepth conditionalDepth -> Right (serialDepth, conditionalDepth)

bufferAddress :: Name -> [IR.Extent] -> [Index scope] -> Either BuildError IR.IndexExpr
bufferAddress name shape indices
    | expectedRank /= actualRank =
        Left (RankMismatch name expectedRank actualRank)
    | otherwise = Right (flattenAddress shape (map indexExpression indices))
  where
    expectedRank = length shape
    actualRank = length indices
    indexExpression (Index expression) = expression

flattenAddress :: [IR.Extent] -> [IR.IndexExpr] -> IR.IndexExpr
flattenAddress _ [] = error "bufferAddress accepted an empty index list"
flattenAddress shape (firstIndex : remainingIndices) =
    foldl flatten firstIndex (zip (drop 1 shape) remainingIndices)
  where
    flatten address (extent, nextIndex) =
        IR.AddIndex
            (IR.MulIndex address (IR.ExtentIndex extent))
            nextIndex

checkShape :: Name -> [IR.Extent] -> Either BuildError ()
checkShape name shape
    | null shape = Left (InvalidRank name)
    | otherwise = mapM_ (checkExtent ("buffer " ++ nameText name)) shape

checkSharedShape :: Name -> [Word64] -> Either BuildError ()
checkSharedShape name shape
    | null shape = Left (InvalidRank name)
    | 0 `elem` shape = Left (NonPositiveExtent ("shared buffer " ++ nameText name))
    | otherwise = checkedProduct 4 shape
  where
    checkedProduct _ [] = Right ()
    checkedProduct productSoFar (extent : remaining)
        | productSoFar > maxBound `div` extent = Left (SharedSizeOverflow name)
        | otherwise = checkedProduct (productSoFar * extent) remaining

checkExtent :: String -> IR.Extent -> Either BuildError ()
checkExtent context extent = case extent of
    IR.StaticExtent 0 -> Left (NonPositiveExtent context)
    IR.StaticExtent _ -> Right ()
    IR.DynamicExtent _ -> Right ()
    IR.CeilDivExtent dividend 0 -> checkExtent context dividend >> Left ZeroCeilDivisor
    IR.CeilDivExtent dividend _ -> checkExtent context dividend

checkUnusedIdentifier :: Name -> BuildState -> Either BuildError ()
checkUnusedIdentifier name state
    | name `elem` stateUsedIdentifiers state = Left (DuplicateIdentifier name)
    | otherwise = Right ()

checkIdentifier :: Name -> Either BuildError ()
checkIdentifier name
    | not (validIdentifier name) = Left (InvalidIdentifier name)
    | otherwise = Right ()

validIdentifier :: Name -> Bool
validIdentifier (Name name) = case name of
    []           -> False
    first : rest -> identifierStart first && all identifierContinue rest
  where
    identifierStart character =
        isAsciiLower character || isAsciiUpper character || character == '_'
    identifierContinue character = identifierStart character || isDigit character

nameText :: Name -> String
nameText (Name name) = name
