{ stdenv, pkgs }:

stdenv.mkDerivation {
  pname = "nsight_systems";
  version = pkgs.cudaPackages.nsight_systems.version;

  src = pkgs.cudaPackages.nsight_systems;

  dontUnpack = true;

  installPhase = ''
    mkdir -p $out
    cp -r ${pkgs.cudaPackages.nsight_systems}/* $out/
  '';

  postFixup = ''
    patchShebangs $out/host-linux-x64/nsys-ui
  '';
}
