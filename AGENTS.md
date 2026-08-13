## Parse, Don't Validate

We should critically evaluate whether `validate` or `verify` functions are actually needed. Most of the time, they aren't.
Instead of creating a separate validation function, we should assume the input is valid, process it, and return an error immediately if we hit an inconsistency.

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
