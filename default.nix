{ pkgs ? import <nixpkgs> {} }:

pkgs.stdenv.mkDerivation rec {
  name = "my-python-project";
  src = ./.;

  buildInputs = [
    pkgs.pkg-config
    pkgs.libcairo
    pkgs.meson
    pkgs.gcc
    pkgs.python3
    pkgs.python3Packages.virtualenv
  ];

  shellHook = ''
    export PKG_CONFIG_PATH=${pkgs.pkgconfig}/lib/pkgconfig
  '';
}
