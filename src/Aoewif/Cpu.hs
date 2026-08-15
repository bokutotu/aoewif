{-# LANGUAGE DataKinds       #-}
{-# LANGUAGE RoleAnnotations #-}

module Aoewif.Cpu (
    Name (..),
    Program,
    Extent,
    Access (..),
    F32,
    Buffer,
    Index,
    Cpu,
    BuildError (..),
    program,
    staticExtent,
    dynamicExtent,
    input,
    output,
    serial,
    parallel,
    local,
    load,
    store,
    f32,
    add,
    sub,
    mul,
    divide,
) where

import           Aoewif.Cpu.IR (Access (..), Extent, Name (..), Program)
import qualified Aoewif.Cpu.IR as IR
import           Data.Char     (isAsciiLower, isAsciiUpper, isDigit)
import           Data.Word     (Word64)

newtype Buffer scope (access :: Access) = Buffer IR.Buffer

type role Buffer nominal nominal

newtype Index scope = Index IR.IndexId

type role Index nominal

newtype F32 scope = F32 IR.Expr

type role F32 nominal

data BuildError
    = InvalidIdentifier Name
    | DuplicateIdentifier Name
    | UnknownDynamicExtent Name
    | RankMismatch Name Int Int
    | ZeroLocalExtent Name
    | LocalSizeOverflow Name
    deriving stock (Eq, Show)

data BuildState = BuildState
    { stateExtents     :: [Name]
    , stateBuffers     :: [IR.Buffer]
    , stateStatements  :: [IR.Statement]
    , stateIdentifiers :: [Name]
    , stateNextBuffer  :: !Int
    , stateNextIndex   :: !Int
    , stateNextValue   :: !Int
    }

newtype Cpu scope value = Cpu
    { runCpu :: BuildState -> Either BuildError (value, BuildState)
    }

type role Cpu nominal representational

instance Functor (Cpu scope) where
    fmap transform (Cpu action) = Cpu $ \state -> do
        (value, nextState) <- action state
        pure (transform value, nextState)

instance Applicative (Cpu scope) where
    pure value = Cpu (Right . (value,))
    Cpu functionAction <*> Cpu valueAction = Cpu $ \state -> do
        (function, functionState) <- functionAction state
        (value, valueState) <- valueAction functionState
        pure (function value, valueState)

instance Monad (Cpu scope) where
    Cpu action >>= next = Cpu $ \state -> do
        (value, nextState) <- action state
        runCpu (next value) nextState

program :: Name -> (forall scope. Cpu scope ()) -> Either BuildError Program
program name action
    | not (validIdentifier name) = Left (InvalidIdentifier name)
    | otherwise = do
        (_, finalState) <- runCpu action initialState
        pure
            IR.Program
                { IR.programName = name
                , IR.programExtents = stateExtents finalState
                , IR.programBuffers = stateBuffers finalState
                , IR.programBody = stateStatements finalState
                }
  where
    initialState = BuildState [] [] [] [] 0 0 0

staticExtent :: Word64 -> Extent
staticExtent = IR.StaticExtent

dynamicExtent :: Name -> Cpu scope Extent
dynamicExtent name = do
    registerIdentifier name
    Cpu $ \state ->
        Right
            ( IR.DynamicExtent name
            , state{stateExtents = stateExtents state ++ [name]}
            )

input :: Name -> [Extent] -> Cpu scope (Buffer scope 'ReadOnly)
input name shape = Buffer <$> declareBuffer IR.ReadOnly name shape

output :: Name -> [Extent] -> Cpu scope (Buffer scope 'ReadWrite)
output name shape = Buffer <$> declareBuffer IR.ReadWrite name shape

declareBuffer :: IR.Access -> Name -> [Extent] -> Cpu scope IR.Buffer
declareBuffer access name shape = do
    ensureKnownExtents shape
    registerIdentifier name
    Cpu $ \state ->
        let identifier = IR.BufferId (stateNextBuffer state)
            buffer = IR.Buffer identifier name access shape
         in Right
                ( buffer
                , state
                    { stateBuffers = stateBuffers state ++ [buffer]
                    , stateNextBuffer = stateNextBuffer state + 1
                    }
                )

serial :: Name -> Extent -> (Index scope -> Cpu scope ()) -> Cpu scope ()
serial = buildLoop IR.Serial

parallel :: Name -> Extent -> (Index scope -> Cpu scope ()) -> Cpu scope ()
parallel = buildLoop IR.Parallel

buildLoop :: IR.LoopKind -> Name -> Extent -> (Index scope -> Cpu scope ()) -> Cpu scope ()
buildLoop kind name extent action = do
    ensureKnownExtent extent
    registerIdentifier name
    Cpu $ \state ->
        let identifier = IR.IndexId (stateNextIndex state)
            outerStatements = stateStatements state
            bodyState =
                state
                    { stateStatements = []
                    , stateNextIndex = stateNextIndex state + 1
                    }
         in do
                (_, finalBodyState) <- runCpu (action (Index identifier)) bodyState
                let loop =
                        IR.Loop
                            { IR.loopKind = kind
                            , IR.loopIndex = identifier
                            , IR.loopName = name
                            , IR.loopExtent = extent
                            , IR.loopBody = stateStatements finalBodyState
                            }
                pure
                    ( ()
                    , finalBodyState
                        { stateStatements = outerStatements ++ [IR.For loop]
                        }
                    )

local :: Name -> [Word64] -> (Buffer scope 'ReadWrite -> Cpu scope ()) -> Cpu scope ()
local name dimensions action
    | 0 `elem` dimensions = buildFailure (ZeroLocalExtent name)
    | not (localSizeFits dimensions) = buildFailure (LocalSizeOverflow name)
    | otherwise = do
        registerIdentifier name
        Cpu $ \state ->
            let identifier = IR.BufferId (stateNextBuffer state)
                buffer =
                    IR.Buffer
                        identifier
                        name
                        IR.ReadWrite
                        (map IR.StaticExtent dimensions)
                outerStatements = stateStatements state
                bodyState =
                    state
                        { stateStatements = []
                        , stateNextBuffer = stateNextBuffer state + 1
                        }
             in do
                    (_, finalBodyState) <- runCpu (action (Buffer buffer)) bodyState
                    pure
                        ( ()
                        , finalBodyState
                            { stateStatements =
                                outerStatements
                                    ++ [IR.Allocate buffer (stateStatements finalBodyState)]
                            }
                        )

load :: Buffer scope access -> [Index scope] -> Cpu scope (F32 scope)
load (Buffer buffer) indices = do
    ensureRank buffer indices
    Cpu $ \state ->
        let identifier = IR.ValueId (stateNextValue state)
            statement = IR.Let identifier (IR.bufferId buffer) (map unwrapIndex indices)
         in Right
                ( F32 (IR.ValueExpr identifier)
                , state
                    { stateStatements = stateStatements state ++ [statement]
                    , stateNextValue = stateNextValue state + 1
                    }
                )

store :: Buffer scope 'ReadWrite -> [Index scope] -> F32 scope -> Cpu scope ()
store (Buffer buffer) indices (F32 value) = do
    ensureRank buffer indices
    appendStatement (IR.Store (IR.bufferId buffer) (map unwrapIndex indices) value)

f32 :: Float -> F32 scope
f32 = F32 . IR.F32Literal

add :: F32 scope -> F32 scope -> F32 scope
add (F32 lhs) (F32 rhs) = F32 (IR.AddExpr lhs rhs)

sub :: F32 scope -> F32 scope -> F32 scope
sub (F32 lhs) (F32 rhs) = F32 (IR.SubExpr lhs rhs)

mul :: F32 scope -> F32 scope -> F32 scope
mul (F32 lhs) (F32 rhs) = F32 (IR.MulExpr lhs rhs)

divide :: F32 scope -> F32 scope -> F32 scope
divide (F32 lhs) (F32 rhs) = F32 (IR.DivExpr lhs rhs)

ensureRank :: IR.Buffer -> [Index scope] -> Cpu scope ()
ensureRank buffer indices
    | expected == actual = pure ()
    | otherwise = buildFailure (RankMismatch (IR.bufferName buffer) expected actual)
  where
    expected = length (IR.bufferShape buffer)
    actual = length indices

ensureKnownExtents :: [Extent] -> Cpu scope ()
ensureKnownExtents = mapM_ ensureKnownExtent

ensureKnownExtent :: Extent -> Cpu scope ()
ensureKnownExtent extent = Cpu $ \state -> case extent of
    IR.StaticExtent _ -> Right ((), state)
    IR.DynamicExtent name
        | name `elem` stateExtents state -> Right ((), state)
        | otherwise -> Left (UnknownDynamicExtent name)

registerIdentifier :: Name -> Cpu scope ()
registerIdentifier name = Cpu $ \state ->
    if not (validIdentifier name)
        then Left (InvalidIdentifier name)
        else
            if name `elem` stateIdentifiers state
                then Left (DuplicateIdentifier name)
                else
                    Right
                        ( ()
                        , state{stateIdentifiers = name : stateIdentifiers state}
                        )

appendStatement :: IR.Statement -> Cpu scope ()
appendStatement statement = Cpu $ \state ->
    Right ((), state{stateStatements = stateStatements state ++ [statement]})

buildFailure :: BuildError -> Cpu scope value
buildFailure buildError = Cpu (const (Left buildError))

localSizeFits :: [Word64] -> Bool
localSizeFits = go 4
  where
    go _ [] = True
    go _ (0 : _) = False
    go bytes (dimension : remaining) =
        bytes <= maxBound `div` dimension
            && go (bytes * dimension) remaining

unwrapIndex :: Index scope -> IR.IndexId
unwrapIndex (Index identifier) = identifier

validIdentifier :: Name -> Bool
validIdentifier (Name value) = case value of
    [] -> False
    first : rest ->
        identifierStart first
            && all identifierContinue rest
  where
    identifierStart character = asciiLetter character || character == '_'
    identifierContinue character = identifierStart character || asciiDigit character
    asciiLetter character = isAsciiLower character || isAsciiUpper character
    asciiDigit = isDigit
