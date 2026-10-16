{-# LANGUAGE QuasiQuotes #-}

module Aoewif.Target.Cuda.CpAsync (
    CpAsync (..),
    CpAsyncCopy (..),
    CpAsyncSource (..),
    L2PrefetchSize (..),
)
where

import                          Aoewif.Target.Cuda.AsmOperand (AsmInput)
import {-# SOURCE #-}           Aoewif.Target.Cuda.Expr       (Expr)
import                          Aoewif.Target.Cuda.Render     (Render (..))
import                          Aoewif.Target.Template        (template)

data CpAsyncCopy = Cg16 | Ca4 | Ca8 | Ca16
    deriving stock (Eq, Show)

data L2PrefetchSize = L2Prefetch64 | L2Prefetch128 | L2Prefetch256
    deriving stock (Eq, Show)

data CpAsyncSource
    = EntireSource
    | SourceSize AsmInput
    | IgnoreSource Expr
    deriving stock (Eq, Show)

data CpAsync = CpAsync
    { copy        :: CpAsyncCopy
    , destination :: AsmInput
    , source      :: AsmInput
    , sourceMode  :: CpAsyncSource
    , l2Prefetch  :: Maybe L2PrefetchSize
    }
    deriving stock (Eq, Show)

instance Render CpAsync where
    render (CpAsync kind dst src mode prefetch) =
        [template|
        asm volatile(
            ${predicateSetup}
            "cp.async${cacheOperator}.shared.global${prefetchQualifier} [%0], [%1], ${copySize}${sourceOperand};\n"
            ${predicateScopeEnd}
            :
            : ${destinationInput}, ${sourceInput}${extraInput}
            : "memory"
        )
        |]
      where
        cacheOperator = renderCacheOperator kind
        copySize = renderCopySize kind
        prefetchQualifier = renderL2Prefetch prefetch
        destinationInput = render dst
        sourceInput = render src
        sourceOperand = renderSourceOperand mode
        extraInput = renderSourceInput mode
        predicateSetup = renderPredicateSetup mode
        predicateScopeEnd = renderPredicateScopeEnd mode

renderCacheOperator :: CpAsyncCopy -> String
renderCacheOperator Cg16 = ".cg"
renderCacheOperator _    = ".ca"

renderCopySize :: CpAsyncCopy -> String
renderCopySize Ca4 = "4"
renderCopySize Ca8 = "8"
renderCopySize _   = "16"

renderL2Prefetch :: Maybe L2PrefetchSize -> String
renderL2Prefetch Nothing              = ""
renderL2Prefetch (Just L2Prefetch64)  = ".L2::64B"
renderL2Prefetch (Just L2Prefetch128) = ".L2::128B"
renderL2Prefetch (Just L2Prefetch256) = ".L2::256B"

renderSourceOperand :: CpAsyncSource -> String
renderSourceOperand EntireSource     = ""
renderSourceOperand (SourceSize _)   = ", %2"
renderSourceOperand (IgnoreSource _) = ", p"

renderSourceInput :: CpAsyncSource -> String
renderSourceInput EntireSource = ""
renderSourceInput (SourceSize value) = ", " ++ render value
renderSourceInput (IgnoreSource condition) =
    [template|, "r"(static_cast<uint32_t>(${conditionExpression}))|]
  where
    conditionExpression = render condition

renderPredicateSetup :: CpAsyncSource -> String
renderPredicateSetup (IgnoreSource _) =
    [template|
    "{\n"
    ".reg .pred p;\n"
    "setp.ne.u32 p, %2, 0;\n"
    |]
renderPredicateSetup _ = ""

renderPredicateScopeEnd :: CpAsyncSource -> String
renderPredicateScopeEnd (IgnoreSource _) = [template|"}\n"|]
renderPredicateScopeEnd _                = ""
