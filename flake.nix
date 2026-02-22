{
  description = "IP2Vec Embedding Neural Network";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/6a08e6bb4e46ff7fcbb53d409b253f6bad8a28ce?narHash=sha256-Q/uhWNvd7V7k1H1ZPMy/vkx3F8C13ZcdrKjO7Jv7v0c%3D";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { nixpkgs, fenix, ... }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        cudaSupport = true;
      };
    };

    toolchain = fenix.packages.${system}.stable.defaultToolchain;

    libtorch = pkgs.callPackage ./libtorch.nix {};
    nsight_systems = pkgs.callPackage ./nsight_systems.nix {};
  in {
    devShells.${system}.default = pkgs.mkShell rec {
      nativeBuildInputs = [
        toolchain

        pkgs.rust-analyzer

        pkgs.openssl
        pkgs.pkg-config

        nsight_systems
      ];

      buildInputs = [
        libtorch
        pkgs.stdenv.cc.cc.lib
        pkgs.linuxPackages.nvidia_x11
        pkgs.openblas.dev
      ];

      shellHook = ''
        ${pkgs.cowsay}/bin/cowsay "entered dev env!" | ${pkgs.lolcat}/bin/lolcat -F 0.5
      '';

      LD_LIBRARY_PATH =
        pkgs.lib.makeLibraryPath (nativeBuildInputs ++ buildInputs);

      LIBTORCH = "${libtorch}/";
    };
  };
}

