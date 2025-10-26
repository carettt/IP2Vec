{
  description = "rust project";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, fenix, ... }: let
    system = "x86_64-linux";

    pkgs = nixpkgs.legacyPackages.${system};
    toolchain = fenix.packages.${system}.stable.defaultToolchain;
  in {
    devShells.${system}.default = pkgs.mkShell rec {
      nativeBuildInputs = [
        toolchain

        pkgs.rust-analyzer
      ];

      shellHook = ''
        ${pkgs.cowsay}/bin/cowsay "entered dev env!" | ${pkgs.lolcat}/bin/lolcat -F 0.5
      '';

      LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath nativeBuildInputs;
    };
  };
}

