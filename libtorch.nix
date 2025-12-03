{ stdenv, fetchurl, lib, unzip, autoPatchelfHook, glib, rdma-core, pkgs }:

stdenv.mkDerivation rec {
  pname = "libtorch";
  version = "2.9.0";

  src = fetchurl {
    url = "https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.9.0%2Bcu128.zip";
    sha256 = "sha256-dVVfd4hPFJ1TdJTaaVOjYmzqaJlrB/txHkqD1PSLT9s=";
  };

  nativeBuildInputs = [ unzip autoPatchelfHook ];
  buildInputs = [
    stdenv.cc.cc.lib
    glib
    rdma-core
    pkgs.linuxPackages.nvidia_x11
  ];

  sourceRoot = "libtorch";

  dontBuild = true;

  installPhase = ''
    runHook preInstall

    mkdir -p $out
    cp -r * $out/

    runHook postInstall
  '';

  dontStrip = true;

  meta = with lib; {
    description = "C++ API of the PyTorch machine learning framework";
    homepage = "https://pytorch.org/";
    license = licenses.bsd3;
    platforms = [
      "x86_64-linux"
    ];
  };
}
