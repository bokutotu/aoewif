module Aoewif.Internal.Codegen.Base (
    Backend (..),
    GeneratedSource,
    generatedText,
    generatedName,
    generateSource,
) where

import qualified Aoewif.Internal.IR               as IR
import qualified Aoewif.Internal.Kernel.Operation as Kernel
import           Data.Ratio                       (denominator, numerator, (%))
import           Data.Word                        (Word32)
import           GHC.Float                        (castFloatToWord32,
                                                   castWord32ToFloat)

data Backend = Backend
    { backendPreambleLines  :: [String]
    , backendFunctionPrefix :: String
    , backendAddExpression  :: String -> String -> String
    , backendSubExpression  :: String -> String -> String
    , backendMulExpression  :: String -> String -> String
    , backendDivExpression  :: String -> String -> String
    , backendFmaExpression  :: String -> String -> String -> String
    }

data GeneratedSource = GeneratedSource
    { generatedText :: String
    , generatedName :: String
    }

data SourceNode
    = SourceLine String
    | BlankLine
    | SourceScope String [SourceNode] String

generateSource :: Backend -> IR.IR Kernel.KernelBlock -> GeneratedSource
generateSource backend kernelIR =
    GeneratedSource
        { generatedText = renderSource sourceNodes
        , generatedName = nameText (IR.irName kernelIR)
        }
  where
    sourceNodes =
        map preambleNode (backendPreambleLines backend)
            ++ [ SourceScope
                    (functionDeclaration backend kernelIR ++ " {")
                    (loopIRNodes backend (IR.irBody kernelIR))
                    "}"
               ]
    preambleNode ""   = BlankLine
    preambleNode line = SourceLine line

functionDeclaration :: Backend -> IR.IR operation -> String
functionDeclaration backend kernelIR =
    backendFunctionPrefix backend
        ++ nameText (IR.irName kernelIR)
        ++ "("
        ++ commaSeparated parameters
        ++ ")"
  where
    parameters =
        map tensorParameter inputTensors
            ++ map tensorParameter outputTensors
            ++ map symbolParameter (IR.irSymbols kernelIR)
    inputTensors = filter isInput (IR.irTensors kernelIR)
    outputTensors = filter (not . isInput) (IR.irTensors kernelIR)
    isInput tensor = case IR.tensorKind tensor of
        IR.InputTensor _  -> True
        IR.OutputTensor _ -> False
    tensorParameter tensor = case IR.tensorKind tensor of
        IR.InputTensor index -> "const " ++ scalarTypeName (IR.tensorType tensor) ++ "* input" ++ show index
        IR.OutputTensor index -> scalarTypeName (IR.tensorType tensor) ++ "* output" ++ show index
    symbolParameter symbol = "size_t symbol" ++ show (symbolIndex (IR.symbolId symbol))

loopIRNodes :: Backend -> IR.LoopIR Kernel.KernelBlock -> [SourceNode]
loopIRNodes backend = concatMap (statementNodes backend) . IR.loopIRStatements

statementNodes :: Backend -> IR.Statement Kernel.KernelBlock -> [SourceNode]
statementNodes backend statement = case statement of
    IR.For loop body ->
        [ SourceScope
            ( "for (size_t "
                ++ loopName (IR.loopId loop)
                ++ " = "
                ++ dimensionExpression (IR.loopLowerBound loop)
                ++ "; "
                ++ loopName (IR.loopId loop)
                ++ " < "
                ++ loopUpperBound (IR.loopLowerBound loop) (IR.loopExtent loop)
                ++ "; ++"
                ++ loopName (IR.loopId loop)
                ++ ") {"
            )
            (loopIRNodes backend body)
            "}"
        ]
    IR.Guard predicate body ->
        [ SourceScope
            ("if (" ++ indexPredicate predicate ++ ") {")
            (loopIRNodes backend body)
            "}"
        ]
    IR.Execute block ->
        concatMap (kernelStatementNodes backend) (Kernel.kernelStatements (IR.blockOperation block))

kernelStatementNodes :: Backend -> Kernel.KernelStatement -> [SourceNode]
kernelStatementNodes backend statement = case statement of
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
    Kernel.StoreBuffer buffer address value ->
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

indexExpression :: IR.IndexExpr -> String
indexExpression expression = case expression of
    IR.LoopIndex identifier -> loopName identifier
    IR.DimensionIndex dimension -> dimensionExpression dimension
    IR.ConstantIndex value -> show value ++ "u"
    IR.AddIndex lhs rhs -> binary "+" lhs rhs
    IR.MulIndex lhs rhs -> binary "*" lhs rhs
    IR.CeilDivIndex dividend divisor ->
        "(("
            ++ indexExpression dividend
            ++ ") + "
            ++ show (divisor - 1)
            ++ "u) / "
            ++ show divisor
            ++ "u"
  where
    binary operator lhs rhs =
        "(" ++ indexExpression lhs ++ " " ++ operator ++ " " ++ indexExpression rhs ++ ")"

dimensionExpression :: IR.DimExpr -> String
dimensionExpression dimension = case dimension of
    IR.StaticDim value -> show value
    IR.SymbolDim symbol -> "symbol" ++ show (symbolIndex symbol)
    IR.CeilDivDim dividend divisor ->
        "(("
            ++ dimensionExpression dividend
            ++ ") + "
            ++ show (divisor - 1)
            ++ "u) / "
            ++ show divisor
            ++ "u"

indexPredicate :: IR.Predicate -> String
indexPredicate predicate = case predicate of
    IR.IndexLessThan lhs rhs -> indexExpression lhs ++ " < " ++ indexExpression rhs
    IR.IndexEqual lhs rhs -> indexExpression lhs ++ " == " ++ indexExpression rhs

loopName :: IR.LoopId -> String
loopName (IR.LoopId identifier) = "loop" ++ show identifier

scalarName :: Kernel.ValueId -> String
scalarName (Kernel.ValueId identifier) = "scalar" ++ show identifier

scalarTypeName :: IR.DType -> String
scalarTypeName scalarType = case scalarType of
    IR.F32Type   -> "float"
    IR.BoolType  -> "bool"
    IR.IndexType -> "size_t"

scalarLiteral :: IR.ScalarLiteral -> String
scalarLiteral literal = case literal of
    IR.F32Literal value
        | isNaN value -> "NAN"
        | isInfinite value && value > 0 -> "INFINITY"
        | isInfinite value -> "-INFINITY"
        | otherwise -> rustDebugFloat value ++ "f"
    IR.BoolLiteral value -> if value then "true" else "false"
    IR.IndexLiteral value -> show value ++ "u"

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

compareOperator :: IR.ComparePredicate -> String
compareOperator predicate = case predicate of
    IR.Equal        -> "=="
    IR.NotEqual     -> "!="
    IR.Less         -> "<"
    IR.LessEqual    -> "<="
    IR.Greater      -> ">"
    IR.GreaterEqual -> ">="

symbolIndex :: IR.SymbolId -> Int
symbolIndex (IR.SymbolId identifier) = identifier

bufferName :: Kernel.Buffer -> String
bufferName buffer = case buffer of
    Kernel.InputBuffer identifier -> "input" ++ show identifier
    Kernel.OutputBuffer identifier ->
        "output" ++ show identifier

loopUpperBound :: IR.DimExpr -> IR.DimExpr -> String
loopUpperBound lower extent = case lower of
    IR.StaticDim 0 -> dimensionExpression extent
    _ ->
        "("
            ++ dimensionExpression lower
            ++ " + "
            ++ dimensionExpression extent
            ++ ")"

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

nameText :: IR.Name -> String
nameText (IR.Name name) = name
