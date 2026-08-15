module Aoewif.Cpu.Codegen (
    CSource,
    cSourceText,
    cFunctionName,
    generateC,
) where

import qualified Aoewif.Cpu.IR as IR
import           Data.List     (intercalate)
import           Data.Maybe    (fromJust)

data CSource = CSource String String
    deriving stock (Eq, Show)

cSourceText :: CSource -> String
cSourceText (CSource sourceText _) = sourceText

cFunctionName :: CSource -> String
cFunctionName (CSource _ functionName) = functionName

generateC :: IR.Program -> CSource
generateC programIR = CSource (renderProgram programIR) (sourceFunctionName programIR)

renderProgram :: IR.Program -> String
renderProgram programIR =
    unlines
        ( [ "#include <math.h>"
          , "#include <stddef.h>"
          , ""
          , functionDeclaration programIR ++ " {"
          ]
            ++ renderStatements
                (map (\buffer -> (IR.bufferId buffer, buffer)) (IR.programBuffers programIR))
                []
                1
                (IR.programBody programIR)
            ++ ["}"]
        )

functionDeclaration :: IR.Program -> String
functionDeclaration programIR =
    "void "
        ++ sourceFunctionName programIR
        ++ "("
        ++ parameters
        ++ ")"
  where
    declarations =
        map bufferParameter (IR.programBuffers programIR)
            ++ map extentParameter (IR.programExtents programIR)
    parameters
        | null declarations = "void"
        | otherwise = intercalate ", " declarations
    bufferParameter buffer = case IR.bufferAccess buffer of
        IR.ReadOnly  -> "const float* " ++ sourceBufferName buffer
        IR.ReadWrite -> "float* " ++ sourceBufferName buffer
    extentParameter name = "size_t " ++ extentName name

renderStatements :: [(IR.BufferId, IR.Buffer)] -> [(IR.IndexId, IR.Name)] -> Int -> [IR.Statement] -> [String]
renderStatements buffers indices indentation =
    concatMap (renderStatement buffers indices indentation)

renderStatement :: [(IR.BufferId, IR.Buffer)] -> [(IR.IndexId, IR.Name)] -> Int -> IR.Statement -> [String]
renderStatement buffers indices indentation statement = case statement of
    IR.Let valueIdentifier bufferIdentifier accessIndices ->
        [ indent indentation
            ++ "float "
            ++ valueName valueIdentifier
            ++ " = "
            ++ bufferAccess buffers indices bufferIdentifier accessIndices
            ++ ";"
        ]
    IR.Store bufferIdentifier accessIndices value ->
        [ indent indentation
            ++ bufferAccess buffers indices bufferIdentifier accessIndices
            ++ " = "
            ++ renderExpr value
            ++ ";"
        ]
    IR.For loop ->
        pragma
            ++ [ indent indentation
                    ++ "for (size_t "
                    ++ sourceLoopName loop
                    ++ " = 0; "
                    ++ sourceLoopName loop
                    ++ " < "
                    ++ renderExtent (IR.loopExtent loop)
                    ++ "; ++"
                    ++ sourceLoopName loop
                    ++ ") {"
               ]
            ++ renderStatements
                buffers
                ((IR.loopIndex loop, IR.loopName loop) : indices)
                (indentation + 1)
                (IR.loopBody loop)
            ++ [indent indentation ++ "}"]
      where
        pragma = case IR.loopKind loop of
            IR.Serial   -> []
            IR.Parallel -> [indent indentation ++ "#pragma omp parallel for"]
    IR.Allocate buffer body ->
        [ indent indentation ++ "{"
        , indent (indentation + 1)
            ++ "float "
            ++ sourceBufferName buffer
            ++ "["
            ++ allocationSize (IR.bufferShape buffer)
            ++ "];"
        ]
            ++ renderStatements
                ((IR.bufferId buffer, buffer) : buffers)
                indices
                (indentation + 1)
                body
            ++ [indent indentation ++ "}"]

renderExpr :: IR.Expr -> String
renderExpr expression = case expression of
    IR.F32Literal value     -> renderFloat value
    IR.ValueExpr identifier -> valueName identifier
    IR.AddExpr lhs rhs      -> binary "+" lhs rhs
    IR.SubExpr lhs rhs      -> binary "-" lhs rhs
    IR.MulExpr lhs rhs      -> binary "*" lhs rhs
    IR.DivExpr lhs rhs      -> binary "/" lhs rhs
  where
    binary operator lhs rhs =
        "(" ++ renderExpr lhs ++ " " ++ operator ++ " " ++ renderExpr rhs ++ ")"

bufferAccess :: [(IR.BufferId, IR.Buffer)] -> [(IR.IndexId, IR.Name)] -> IR.BufferId -> [IR.IndexId] -> String
bufferAccess buffers indices identifier accessIndices =
    sourceBufferName buffer
        ++ "["
        ++ rowMajorAddress indices (IR.bufferShape buffer) accessIndices
        ++ "]"
  where
    buffer = fromJust (lookup identifier buffers)

rowMajorAddress :: [(IR.IndexId, IR.Name)] -> [IR.Extent] -> [IR.IndexId] -> String
rowMajorAddress indices shape accessIndices = case map (indexName indices) accessIndices of
    [] -> "0"
    first : remaining ->
        foldl flatten first (zip (drop 1 shape) remaining)
  where
    flatten address (dimension, indexText) =
        "(("
            ++ address
            ++ ") * ("
            ++ renderExtent dimension
            ++ ") + ("
            ++ indexText
            ++ "))"

allocationSize :: [IR.Extent] -> String
allocationSize [] = "1"
allocationSize dimensions = intercalate " * " (map (parenthesize . renderExtent) dimensions)
  where
    parenthesize value = "(" ++ value ++ ")"

renderExtent :: IR.Extent -> String
renderExtent extent = case extent of
    IR.StaticExtent value -> show value ++ "ull"
    IR.DynamicExtent name -> extentName name

renderFloat :: Float -> String
renderFloat value
    | isNaN value = "NAN"
    | isInfinite value && value < 0 = "(-INFINITY)"
    | isInfinite value = "INFINITY"
    | otherwise = show value ++ "f"

valueName :: IR.ValueId -> String
valueName (IR.ValueId identifier) = "value" ++ show identifier

sourceBufferName :: IR.Buffer -> String
sourceBufferName = nameText . IR.bufferName

sourceLoopName :: IR.Loop -> String
sourceLoopName = nameText . IR.loopName

indexName :: [(IR.IndexId, IR.Name)] -> IR.IndexId -> String
indexName indices identifier = nameText (fromJust (lookup identifier indices))

extentName :: IR.Name -> String
extentName = nameText

sourceFunctionName :: IR.Program -> String
sourceFunctionName = nameText . IR.programName

nameText :: IR.Name -> String
nameText (IR.Name value) = value

indent :: Int -> String
indent indentation = replicate (indentation * 4) ' '
