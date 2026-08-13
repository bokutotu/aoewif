module Aoewif.Internal.Codegen.Base (
    Backend (..),
    GeneratedSource,
    generatedText,
    generatedName,
    generateSource,
) where

import qualified Aoewif.Internal.Compute.IR as Compute
import qualified Aoewif.Internal.Kernel.IR  as Kernel
import           Data.Ratio                 (denominator, numerator, (%))
import           Data.Word                  (Word32, Word64)
import           GHC.Float                  (castFloatToWord32,
                                             castWord32ToFloat)

data Backend = Backend
    { backendPreambleLines     :: [String]
    , backendFunctionPrefix    :: String
    , backendParallelDirective :: Maybe String
    , backendUnrollDirective   :: Word64 -> String
    , backendAddExpression     :: String -> String -> String
    , backendSubExpression     :: String -> String -> String
    , backendMulExpression     :: String -> String -> String
    , backendDivExpression     :: String -> String -> String
    , backendFmaExpression     :: String -> String -> String -> String
    }

data GeneratedSource = GeneratedSource
    { generatedText :: String
    , generatedName :: String
    }

data SourceNode
    = SourceLine String
    | BlankLine
    | SourceScope String [SourceNode] String

generateSource :: Backend -> Kernel.Kernel -> GeneratedSource
generateSource backend kernel =
    GeneratedSource
        { generatedText = renderSource sourceNodes
        , generatedName = Kernel.kernelName kernel
        }
  where
    sourceNodes =
        map preambleNode (backendPreambleLines backend)
            ++ [ SourceScope
                    (functionDeclaration backend kernel ++ " {")
                    (concatMap (statementNodes backend) (Kernel.kernelBody kernel))
                    "}"
               ]
    preambleNode ""   = BlankLine
    preambleNode line = SourceLine line

functionDeclaration :: Backend -> Kernel.Kernel -> String
functionDeclaration backend kernel =
    backendFunctionPrefix backend
        ++ Kernel.kernelName kernel
        ++ "("
        ++ commaSeparated parameters
        ++ ")"
  where
    parameters =
        map inputParameter (Kernel.kernelInputs kernel)
            ++ map outputParameter (Kernel.kernelOutputs kernel)
            ++ map symbolParameter (Kernel.kernelSymbols kernel)
    inputParameter input = "const float* input" ++ show (Kernel.inputIndex input)
    outputParameter output = "float* output" ++ show (Kernel.outputIndex output)
    symbolParameter symbol = "size_t symbol" ++ show (symbolIndex (Compute.symbolId symbol))

statementNodes :: Backend -> Kernel.Statement -> [SourceNode]
statementNodes backend statement = case statement of
    Kernel.DefineValue value ->
        [ SourceLine
            ( scalarTypeName (Kernel.valueType value)
                ++ " "
                ++ scalarName (Kernel.valueId value)
                ++ " = "
                ++ scalarOperationExpression backend (Kernel.valueOperation value)
                ++ ";"
            )
        ]
    Kernel.ForLoop identifier lower extent execution unroll builtin body -> case builtin of
        Just _ -> concatMap (statementNodes backend) body
        Nothing ->
            loopDirectives backend execution unroll
                ++ [ SourceScope
                        ( "for (size_t "
                            ++ loopName identifier
                            ++ " = "
                            ++ indexExpression lower
                            ++ "; "
                            ++ loopName identifier
                            ++ " < "
                            ++ indexExpression (loopUpperBound lower extent)
                            ++ "; ++"
                            ++ loopName identifier
                            ++ ") {"
                        )
                        (concatMap (statementNodes backend) body)
                        "}"
                   ]
    Kernel.Conditional predicates body ->
        [ SourceScope
            ("if (" ++ joinWith " && " (map indexPredicate predicates) ++ ") {")
            (concatMap (statementNodes backend) body)
            "}"
        ]
    Kernel.StoreBuffer buffer address value _ ->
        [SourceLine (bufferName buffer ++ "[" ++ indexExpression address ++ "] = " ++ scalarName value ++ ";")]

scalarOperationExpression :: Backend -> Kernel.ScalarOperation -> String
scalarOperationExpression backend operation = case operation of
    Kernel.LiteralOperation literal -> scalarLiteral literal
    Kernel.IndexOperation expression -> indexExpression expression
    Kernel.LoadOperation buffer address ->
        bufferName buffer ++ "[" ++ indexExpression address ++ "]"
    Kernel.AddOperation lhs rhs -> binary (backendAddExpression backend) lhs rhs
    Kernel.SubOperation lhs rhs -> binary (backendSubExpression backend) lhs rhs
    Kernel.MulOperation lhs rhs -> binary (backendMulExpression backend) lhs rhs
    Kernel.DivOperation lhs rhs -> binary (backendDivExpression backend) lhs rhs
    Kernel.FmaOperation lhs rhs accumulator ->
        backendFmaExpression backend (scalarName lhs) (scalarName rhs) (scalarName accumulator)
    Kernel.MinOperation lhs rhs -> function2 "fminf" lhs rhs
    Kernel.MaxOperation lhs rhs -> function2 "fmaxf" lhs rhs
    Kernel.ExpOperation input -> function1 "expf" input
    Kernel.LogOperation input -> function1 "logf" input
    Kernel.CompareOperation predicate lhs rhs ->
        "(" ++ scalarName lhs ++ " " ++ compareOperator predicate ++ " " ++ scalarName rhs ++ ")"
    Kernel.SelectOperation condition trueValue falseValue ->
        "("
            ++ scalarName condition
            ++ " ? "
            ++ scalarName trueValue
            ++ " : "
            ++ scalarName falseValue
            ++ ")"
  where
    binary render lhs rhs = render (scalarName lhs) (scalarName rhs)
    function1 function input = function ++ "(" ++ scalarName input ++ ")"
    function2 function lhs rhs = function ++ "(" ++ scalarName lhs ++ ", " ++ scalarName rhs ++ ")"

indexExpression :: Kernel.IndexExpression -> String
indexExpression expression = case expression of
    Kernel.LoopValue identifier -> loopName identifier
    Kernel.BuiltinValue builtin -> builtinExpression builtin
    Kernel.ConstantValue value -> show value ++ "u"
    Kernel.DimensionValue value -> show value
    Kernel.SymbolValue symbol -> "symbol" ++ show (symbolIndex symbol)
    Kernel.AddValue lhs rhs -> binary "+" lhs rhs
    Kernel.MulValue lhs rhs -> binary "*" lhs rhs
    Kernel.CeilDivValue dividend divisor ->
        "(("
            ++ indexExpression dividend
            ++ ") + "
            ++ show (divisor - 1)
            ++ "u) / "
            ++ show divisor
            ++ "u"
    Kernel.FlattenedValue address dimension index ->
        "(("
            ++ indexExpression address
            ++ ") * "
            ++ indexExpression dimension
            ++ " + "
            ++ indexExpression index
            ++ ")"
  where
    binary operator lhs rhs =
        "(" ++ indexExpression lhs ++ " " ++ operator ++ " " ++ indexExpression rhs ++ ")"

indexPredicate :: Kernel.IndexPredicate -> String
indexPredicate predicate = case predicate of
    Kernel.IndexLessThan lhs rhs -> indexExpression lhs ++ " < " ++ indexExpression rhs
    Kernel.IndexEqual lhs rhs -> indexExpression lhs ++ " == " ++ indexExpression rhs

builtinExpression :: Kernel.BuiltinIndex -> String
builtinExpression builtin = case builtin of
    Kernel.BlockX  -> "((size_t)blockIdx.x)"
    Kernel.BlockY  -> "((size_t)blockIdx.y)"
    Kernel.BlockZ  -> "((size_t)blockIdx.z)"
    Kernel.ThreadX -> "((size_t)threadIdx.x)"
    Kernel.ThreadY -> "((size_t)threadIdx.y)"
    Kernel.ThreadZ -> "((size_t)threadIdx.z)"

loopName :: Kernel.LoopId -> String
loopName (Kernel.LoopId identifier) = "loop" ++ show identifier

scalarName :: Kernel.ValueId -> String
scalarName identifier = case Kernel.valueRole identifier of
    Kernel.TemporaryValue -> "scalar" ++ show (Kernel.valueIndex identifier)
    Kernel.AccumulatorValue ordinal
        | ordinal == 0 -> "accumulator"
        | otherwise -> "accumulator" ++ show ordinal

scalarTypeName :: Compute.DType -> String
scalarTypeName scalarType = case scalarType of
    Compute.F32Type   -> "float"
    Compute.BoolType  -> "bool"
    Compute.IndexType -> "size_t"

scalarLiteral :: Compute.ScalarLiteral -> String
scalarLiteral literal = case literal of
    Compute.F32Literal value
        | isNaN value -> "NAN"
        | isInfinite value && value > 0 -> "INFINITY"
        | isInfinite value -> "-INFINITY"
        | otherwise -> rustDebugFloat value ++ "f"
    Compute.BoolLiteral value -> if value then "true" else "false"
    Compute.IndexLiteral value -> show value ++ "u"

rustDebugFloat :: Float -> String
rustDebugFloat value
    | value == 0 = if isNegativeZero value then "-0.0" else "0.0"
    | decimalExponent < -4 || decimalExponent >= 16 = scientific
    | decimalPoint <= 0 = sign ++ "0." ++ replicate (negate decimalPoint) '0' ++ digits
    | decimalPoint < length digits =
        sign ++ take decimalPoint digits ++ "." ++ drop decimalPoint digits
    | otherwise =
        sign ++ digits ++ replicate (decimalPoint - length digits) '0' ++ ".0"
  where
    sign = if value < 0 then "-" else ""
    (coefficient, scaleExponent) = shortestFloatDecimal (abs value)
    digits = show coefficient
    decimalPoint = length digits + scaleExponent
    decimalExponent = decimalPoint - 1
    scientific = case digits of
        [] -> error "shortestFloatDecimal returned no digits"
        leadingDigit : remainingDigits ->
            sign
                ++ [leadingDigit]
                ++ (if null remainingDigits then "" else "." ++ remainingDigits)
                ++ "e"
                ++ show decimalExponent

shortestFloatDecimal :: Float -> (Integer, Int)
shortestFloatDecimal value = normalizeDecimal (findCoefficient 1)
  where
    bits = castFloatToWord32 value
    exact = floatRational value
    previous
        | bits == 1 = 0
        | otherwise = floatRational (castWord32ToFloat (bits - 1))
    following
        | bits == maxFiniteFloatBits = (2 ^ (128 :: Int)) % 1
        | otherwise = floatRational (castWord32ToFloat (bits + 1))
    lowerBoundary = (previous + exact) / 2
    upperBoundary = (exact + following) / 2
    inclusiveBoundary = even bits
    magnitudeExponent = floatDecimalMagnitude exact

    findCoefficient significantDigits
        | significantDigits > 9 = error "a finite f32 must have a nine-digit decimal representation"
        | lowerCoefficient <= upperCoefficient = (coefficient, scaleExponent)
        | otherwise = findCoefficient (significantDigits + 1)
      where
        scaleExponent = magnitudeExponent - significantDigits + 1
        scale = decimalPower scaleExponent
        lowerCoefficient =
            max
                (10 ^ (significantDigits - 1))
                (integerAbove inclusiveBoundary (lowerBoundary / scale))
        upperCoefficient =
            min
                (10 ^ significantDigits)
                (integerBelow inclusiveBoundary (upperBoundary / scale))
        nearestCoefficient = roundShortest (exact / scale)
        coefficient = max lowerCoefficient (min upperCoefficient nearestCoefficient)

normalizeDecimal :: (Integer, Int) -> (Integer, Int)
normalizeDecimal (coefficient, scaleExponent)
    | coefficient `rem` 10 == 0 = normalizeDecimal (coefficient `quot` 10, scaleExponent + 1)
    | otherwise = (coefficient, scaleExponent)

maxFiniteFloatBits :: Word32
maxFiniteFloatBits = 0x7f7fffff

floatRational :: Float -> Rational
floatRational value =
    let (floatSignificand, floatExponent) = decodeFloat value
     in (floatSignificand % 1) * binaryPower floatExponent

binaryPower :: Int -> Rational
binaryPower power
    | power >= 0 = (2 ^ power) % 1
    | otherwise = 1 % (2 ^ negate power)

decimalPower :: Int -> Rational
decimalPower power
    | power >= 0 = (10 ^ power) % 1
    | otherwise = 1 % (10 ^ negate power)

floatDecimalMagnitude :: Rational -> Int
floatDecimalMagnitude value = findMagnitude (-46)
  where
    findMagnitude magnitude
        | value < decimalPower (magnitude + 1) = magnitude
        | otherwise = findMagnitude (magnitude + 1)

integerAbove :: Bool -> Rational -> Integer
integerAbove inclusive value
    | inclusive = ceiling value
    | denominator value == 1 = numerator value + 1
    | otherwise = ceiling value

integerBelow :: Bool -> Rational -> Integer
integerBelow inclusive value
    | inclusive = floor value
    | denominator value == 1 = numerator value - 1
    | otherwise = floor value

roundShortest :: Rational -> Integer
roundShortest value
    | remainder * 2 < denominator value = quotient
    | otherwise = quotient + 1
  where
    (quotient, remainder) = quotRem (numerator value) (denominator value)

compareOperator :: Compute.ComparePredicate -> String
compareOperator predicate = case predicate of
    Compute.Equal        -> "=="
    Compute.NotEqual     -> "!="
    Compute.Less         -> "<"
    Compute.LessEqual    -> "<="
    Compute.Greater      -> ">"
    Compute.GreaterEqual -> ">="

symbolIndex :: Compute.SymbolId -> Int
symbolIndex (Compute.SymbolId identifier) = identifier

bufferName :: Kernel.Buffer -> String
bufferName buffer = case buffer of
    Kernel.InputBuffer identifier -> "input" ++ show identifier
    Kernel.OutputBuffer identifier ->
        "output" ++ show identifier

loopDirectives :: Backend -> Kernel.LoopExecution -> Maybe Word64 -> [SourceNode]
loopDirectives backend execution unroll = parallelDirective ++ unrollDirective
  where
    parallelDirective = case (execution, backendParallelDirective backend) of
        (Kernel.ParallelExecution, Just directive) -> [SourceLine directive]
        _                                          -> []
    unrollDirective = case unroll of
        Just factor -> [SourceLine (backendUnrollDirective backend factor)]
        Nothing     -> []

loopUpperBound :: Kernel.IndexExpression -> Kernel.IndexExpression -> Kernel.IndexExpression
loopUpperBound lower extent = case lower of
    Kernel.DimensionValue 0 -> extent
    _                       -> Kernel.AddValue lower extent

renderSource :: [SourceNode] -> String
renderSource = concatMap (renderNode 0)
  where
    renderNode indentation node = case node of
        SourceLine line -> indent indentation ++ line ++ "\n"
        BlankLine -> "\n"
        SourceScope openLine body closeLine ->
            indent indentation
                ++ openLine
                ++ "\n"
                ++ concatMap (renderNode (indentation + 1)) body
                ++ indent indentation
                ++ closeLine
                ++ "\n"
    indent indentation = replicate (indentation * 4) ' '

commaSeparated :: [String] -> String
commaSeparated = joinWith ", "

joinWith :: String -> [String] -> String
joinWith _ [] = ""
joinWith _ [value] = value
joinWith separator (value : values) = value ++ separator ++ joinWith separator values
