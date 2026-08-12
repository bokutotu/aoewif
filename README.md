# aoewif

Tensor IR construction, CPU/CUDA scheduling, and C/CUDA source generation in Haskell.

The library depends only on `base`. Its test suite uses Hspec.

```console
direnv allow
cabal build all
cabal test all
```

The Nix development shell uses GHC 9.14.1 and includes Cabal, Haskell Language
Server (`haskell-language-server` and `hls`), Fourmolu, Stylish Haskell, and
`cabal-gild`.
