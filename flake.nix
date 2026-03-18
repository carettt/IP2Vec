{
  description = "IP2Vec Embedding Neural Network, LGPL-3.0-or-later";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/6a08e6bb4e46ff7fcbb53d409b253f6bad8a28ce?narHash=sha256-Q/uhWNvd7V7k1H1ZPMy/vkx3F8C13ZcdrKjO7Jv7v0c%3D";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { nixpkgs, fenix, ... }: let
    version = (builtins.fromTOML (builtins.readFile ./Cargo.toml)).package.version;

    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        cudaSupport = true;
      };
    };

    toolchain = fenix.packages.${system}.stable.minimalToolchain;

    libtorch = pkgs.callPackage ./libtorch.nix {};
    nsight_systems = pkgs.callPackage ./nsight_systems.nix {};

    runtimeDeps = [
      pkgs.openssl
    ];
  in {
    devShells.${system}.default = pkgs.mkShell (let
      toolchain = fenix.packages.${system}.stable.defaultToolchain;
    in rec {
      nativeBuildInputs = [
        pkgs.pkg-config

        toolchain
        pkgs.rust-analyzer

        nsight_systems
      ];

      buildInputs = runtimeDeps ++ [ pkgs.stdenv.cc.cc.lib libtorch ];

      shellHook = ''
        ${pkgs.cowsay}/bin/cowsay "entered dev env!" | ${pkgs.lolcat}/bin/lolcat -F 0.5
      '';

      LD_LIBRARY_PATH =
        (pkgs.lib.makeLibraryPath (nativeBuildInputs ++ buildInputs ++ ["/run/opengl-driver"]));

      LIBTORCH = "${libtorch}/";
    });

    packages.${system} = rec {
      default = (pkgs.makeRustPlatform {
        cargo = toolchain;
        rustc = toolchain;
      }).buildRustPackage {
        pname = "ip2vec";
        inherit version;

        src = ./.;
        cargoLock.lockFile = ./Cargo.lock;

        nativeBuildInputs = [
          pkgs.pkg-config
          pkgs.patchelf
          pkgs.pax-utils
        ];

        buildInputs = runtimeDeps;

        doCheck = false;

        LIBTORCH = "${libtorch}/";
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeDeps;

        postFixup = ''
          mkdir -p $out/lib

          # Copy all dereferenced library deps
          lddtree -l $out/bin/ip2vec \
            | grep "^/nix" \
            | grep -v "^$out" \
            | grep -v "libtorch\|libcuda\|libnv" \
            | xargs -I{} cp -L {} $out/lib/
          
          chmod +w $out/bin/*
          chmod +w $out/lib/*

          # Patch all library deps
          for lib in $out/lib/*; do
            patchelf --set-rpath '$ORIGIN:/run/opengl-driver/lib' "$lib"
          done

          # Patch binaries
          patchelf --set-rpath '$ORIGIN/../lib:/run/opengl-driver/lib' $out/bin/*
        '';
      };

      bundle = pkgs.runCommand "bundle" {} ''
        mkdir -p $out
        tar -czf $out/${default.pname}-v${default.version}-bundle.tar.gz \
          -C ${default} bin lib
      '';
    };
  };
}

