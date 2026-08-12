{
  description = "aoewif Haskell development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      mkPackage =
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          haskellPackages = pkgs.haskell.packages.ghc9141;
        in
        haskellPackages.callCabal2nixWithOptions "aoewif" ./. "--hpack" {
          hpack = pkgs.haskellPackages.hpack;
        };
    in
    {
      packages = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.haskell.lib.dontCheck (mkPackage system);
        }
      );

      checks = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          aoewif-test = pkgs.haskell.lib.doCheck (mkPackage system);
        }
      );

      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          hlib = pkgs.haskell.lib;
          disableFlags =
            flags: package: builtins.foldl' (result: flag: hlib.disableCabalFlag result flag) package flags;
          haskellPackages = pkgs.haskell.packages.ghc9141.override {
            overrides = self: super: {
              algebraic-graphs = super.algebraic-graphs_0_8;
              constraints-extras = hlib.doJailbreak super.constraints-extras;
              dependent-map = hlib.doJailbreak super.dependent-map;
              enummapset = hlib.dontCheck super.enummapset;
              generic-lens = hlib.dontCheck super.generic-lens;
              ghcide = hlib.doJailbreak super.ghcide;
              ghc-trace-events = hlib.doJailbreak super.ghc-trace-events;
              hiedb = hlib.dontCheck (hlib.doJailbreak super.hiedb);
              hie-compat = hlib.doJailbreak super.hie-compat;
              lucid = hlib.doJailbreak super.lucid;
              lsp = hlib.doJailbreak super.lsp;
              lsp-test = hlib.doJailbreak super.lsp-test;
              lsp-types = hlib.doJailbreak super.lsp-types;
              rebase = hlib.doJailbreak super.rebase;
              regex-tdfa = hlib.dontCheck super.regex-tdfa;
              string-interpolate = hlib.doJailbreak super.string-interpolate;
              tasty-hspec = hlib.doJailbreak super.tasty-hspec;
              haskell-language-server =
                hlib.overrideCabal
                  (disableFlags
                    [
                      "cabal"
                      "cabalfmt"
                      "cabalgild"
                      "floskell"
                      "fourmolu"
                      "ghc-lib"
                      "hlint"
                      "ormolu"
                      "retrie"
                      "splice"
                      "stan"
                      "stylishHaskell"
                    ]
                    (
                      super.haskell-language-server.overrideScope (
                        _: _: {
                          Cabal = null;
                          Cabal-syntax = null;
                          apply-refact = null;
                          cabal-add = null;
                          eventlog2html = null;
                          fourmolu = null;
                          hlint = null;
                          ormolu = null;
                          refact = null;
                          shake-bench = null;
                          stan = null;
                          stylish-haskell = null;
                        }
                      )
                    )
                  )
                  (_: {
                    buildDepends = [ ];
                  });
            };
          };
          ghc = haskellPackages.ghcWithPackages (haskellPkgs: [ haskellPkgs.hspec ]);
          hls = pkgs.writeShellScriptBin "hls" ''
            exec ${haskellPackages.haskell-language-server}/bin/haskell-language-server-wrapper "$@"
          '';
        in
        {
          default = pkgs.mkShellNoCC {
            packages = [
              ghc
              pkgs.cabal-install
              pkgs.haskellPackages.fast-tags
              pkgs.fourmolu
              pkgs.haskellPackages.hpack
              haskellPackages.haskell-language-server
              pkgs.stylish-haskell
              hls
            ];
          };
        }
      );

      formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.nixfmt);
    };
}
