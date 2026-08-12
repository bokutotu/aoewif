# aoewif

Tensor IR construction, CPU/CUDA scheduling, and C/CUDA source generation in Haskell.

The library depends only on `base`. Its test suite uses Hspec.

```console
direnv allow
hpack
cabal build all
cabal test all
```

Package metadata is maintained in `package.yaml`. Hpack generates the ignored
`aoewif.cabal` file locally.

The Nix development shell uses GHC 9.14.1 and includes Cabal, Hpack, Haskell
Language Server (`haskell-language-server` and `hls`), Fourmolu, and Stylish
Haskell.
