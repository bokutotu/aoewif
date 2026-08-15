## Parse, Don't Validate

We should critically evaluate whether `validate` or `verify` functions are actually needed. Most of the time, they aren't.
Validate untrusted user input while parsing it. Once parsed, assume the input is valid, process it directly, and return an error if processing encounters an inconsistency.

Do not validate values returned by this project's functions or modules. If a function returns an inconsistent value, add a failing test and fix that function.

## Writing Tests

Never create separate test data generator functions. You should always inline test inputs and expected outputs within the test function itself.
Decoupling the test data from the test logic compromises code readability.

In most test cases, abstractions for testing are unnecessary.

Must not handle error manually in tests.

Bad
```haskell
it "returns the expected result" $ do
    case targetFunctionReturningEither of
        Right result -> result `shouldBe` expected
        Left err -> error (show err)
```

Good: expected success
```haskell
it "returns the expected result" $ do
    Right result <- targetFunctionReturningEither
    result `shouldBe` expected
```

Good: expected failure
```haskell
it "returns the expected error" $ do
    targetFunctionReturningEither `shouldBe` Left expectedError
```

## Documentation

Don't write documentation unless the user requests it.

## Shared modules

### When

If two types have the same domain meaning and invariants, and their conversion only renames equivalent constructors or fields, prefer replacing them with one shared type.

Module A
```haskell
newtype Dim = Dim [Symbol]

data Symbol = Dynamic | Static Int
```

Module B
```haskell
newtype Shape = Shape [Item]

data Item = Dyn | Sta Int

shapeFromDim :: Dim -> Shape
shapeFromDim (Dim symbols) = Shape (fmap convertSymbol symbols)
  where
    convertSymbol Dynamic = Dyn
    convertSymbol (Static size) = Sta size
```

In this case, replace both types with a shared `Dim` type. Do not share types merely because their structures match; keep them separate when they represent different domain concepts or invariants.

### Where

Place the shared module in the lowest common parent namespace of all consumers.

Before:
```
ParentParent
  Parent
    A
    B
```

After:
```
ParentParent
  Parent
    Dim
    A
    B
```

## Linting

Run the project's linter and apply its fixes instead of manually patching lint warnings.
