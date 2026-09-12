## Design Doc

Check DESIGN.md

## CUDA / PTX Specification References

Prioritize the following official NVIDIA documentation when checking CUDA, PTX, or GPU architecture specifications.
Prefer fetching the relevant documents directly with `curl -fL` before searching the internet.

- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Inline PTX Assembly](https://docs.nvidia.com/cuda/inline-ptx-assembly/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)
- [NVIDIA Ampere Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [NVIDIA Blackwell Architecture Technical Brief](https://dam-cdn.nvd.orangelogic.com/AssetLink/gl2l4l4812s5fw0p614s6i8bv6mi3vx5.pdf)

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
