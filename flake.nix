{
  description = "aoewif Haskell development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    git-hooks = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { git-hooks, nixpkgs, ... }:
    let
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      mkHaskellPackages =
        pkgs:
        let
          hlib = pkgs.haskell.lib;
          disableFlags =
            flags: package: builtins.foldl' (result: flag: hlib.disableCabalFlag result flag) package flags;
        in
        pkgs.haskell.packages.ghc9141.override {
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
            ordered-containers = hlib.doJailbreak super.ordered-containers;
            rebase = hlib.doJailbreak super.rebase;
            regex-tdfa = hlib.dontCheck super.regex-tdfa;
            string-interpolate = hlib.doJailbreak super.string-interpolate;
            tasty-hspec = hlib.doJailbreak super.tasty-hspec;
            toml-reader = hlib.dontCheck super.toml-reader;
            weeder = hlib.justStaticExecutables (
              hlib.dontCheck (hlib.doJailbreak super.weeder)
            );
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
      mkPackage =
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          haskellPackages = pkgs.haskell.packages.ghc9141;
        in
        haskellPackages.callCabal2nixWithOptions "aoewif" ./. "--hpack" {
          hpack = pkgs.haskellPackages.hpack;
        };
      mkPreCommitCheck =
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          haskellPackages = mkHaskellPackages pkgs;
          ghc = haskellPackages.ghcWithPackages (haskellPkgs: [ haskellPkgs.hspec ]);
          cabalConfig = pkgs.writeText "weeder-cabal.config" ''
            repository hackage.haskell.org
              url: https://hackage.haskell.org/
              secure: False
          '';
          haskellFormat = pkgs.writeShellApplication {
            name = "fourmolu-then-stylish-haskell";
            runtimeInputs = [
              pkgs.fourmolu
              pkgs.stylish-haskell
            ];
            text = ''
              fourmolu --mode inplace "$@"
              stylish-haskell --inplace "$@"
            '';
          };
          testCheck = pkgs.writeShellApplication {
            name = "test-check";
            runtimeInputs = [
              ghc
              pkgs.cabal-install
              pkgs.haskellPackages.hpack
            ];
            text = ''
              export CABAL_CONFIG=${cabalConfig}
              hpack
              cabal test --offline
            '';
          };
          weederCheck = pkgs.writeShellApplication {
            name = "weeder-check";
            runtimeInputs = [
              ghc
              pkgs.cabal-install
              pkgs.haskellPackages.hpack
              haskellPackages.weeder
            ];
            text = ''
              export CABAL_CONFIG=${cabalConfig}
              hpack
              weeder_build_dir="dist-newstyle/weeder-$(ghc --numeric-version)"
              cabal build all \
                --offline \
                --enable-tests \
                --builddir="$weeder_build_dir" \
                --ghc-options=-fwrite-ide-info
              weeder \
                --config weeder.toml \
                --hie-directory "$weeder_build_dir" \
                --require-hs-files
            '';
          };
        in
        git-hooks.lib.${system}.run {
          src = ./.;
          hooks = {
            fourmolu-then-stylish-haskell = {
              enable = true;
              name = "fourmolu then stylish-haskell";
              package = haskellFormat;
              entry = "${haskellFormat}/bin/fourmolu-then-stylish-haskell";
              files = "\\.l?hs(-boot)?$";
              before = [
                "hlint"
                "test"
                "weeder"
              ];
            };
            hlint = {
              enable = true;
              name = "hlint";
              description = "Lint Haskell sources.";
              package = pkgs.hlint;
              entry = "${pkgs.hlint}/bin/hlint";
              files = "\\.l?hs(-boot)?$";
              before = [
                "test"
                "weeder"
              ];
            };
            test = {
              enable = true;
              name = "test";
              description = "Run the test suite.";
              package = testCheck;
              entry = "${testCheck}/bin/test-check";
              files = "(\\.l?hs(-boot)?$|(^|/)package\\.yaml$)";
              pass_filenames = false;
              before = [ "weeder" ];
            };
            weeder = {
              enable = true;
              name = "weeder";
              description = "Whole-program dead-code analysis.";
              package = weederCheck;
              entry = "${weederCheck}/bin/weeder-check";
              files = "(\\.l?hs(-boot)?$|(^|/)package\\.yaml$|^weeder\\.toml$)";
              pass_filenames = false;
            };
          };
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
          pre-commit-check = mkPreCommitCheck system;
        }
      );

      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          haskellPackages = mkHaskellPackages pkgs;
          ghc = haskellPackages.ghcWithPackages (haskellPkgs: [ haskellPkgs.hspec ]);
          hls = pkgs.writeShellScriptBin "hls" ''
            exec ${haskellPackages.haskell-language-server}/bin/haskell-language-server-wrapper "$@"
          '';
          preCommitCheck = mkPreCommitCheck system;
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
              haskellPackages.weeder
              pkgs.stylish-haskell
              hls
            ]
            ++ preCommitCheck.enabledPackages;
            shellHook = preCommitCheck.shellHook;
          };
        }
      );

      formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.nixfmt);
    };
}
