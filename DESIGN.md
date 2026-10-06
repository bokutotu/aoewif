## Target Layer

The Target layer represents the target language's syntax and types in Haskell and is responsible only for code generation. Type distinctions must correspond to those in the target language; do not introduce additional type constraints based on intended use or execution state.

This layer performs no explicit validation. Its users are responsible for the validity of generated code. Stronger type safety and semantic constraints belong in higher layers.

### PTX

PTX is a target language alongside CUDA C++. PTX instructions are represented as syntax and are all treated as expressions. Instructions without a result are used as expression statements.

CUDA C++ built-ins such as `__syncthreads()` and `threadIdx` are part of the target language and may be represented by dedicated constructors.
