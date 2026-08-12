## Parse, Don't Validate

We should critically evaluate whether `validate` or `verify` functions are actually needed. Most of the time, they aren't.
Instead of creating a separate validation function, we should assume the input is valid, process it, and return an error immediately if we hit an inconsistency.

## Writing Tests

Never create separate test data generator functions. You should always inline test inputs and expected outputs within the test function itself.
Decoupling the test data from the test logic compromises code readability.

In most test cases, abstractions for testing are unnecessary.
