{ pkgs ? import <nixpkgs> {} }:

pkgs.stdenv.mkDerivation rec {
  name = "my-project";
  src = ./.;

  buildInputs = [
    pkgs.pkgconfig
    pkgs.libcairo
    pkgs.meson
    pkgs.cmake
    pkgs.gcc
    pkgs.glibc
  ];

  shellHook = ''
    export PKG_CONFIG_PATH=${pkgs.pkgconfig}/lib/pkgconfig
  '';
}
