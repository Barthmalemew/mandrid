{ pkgs ? import <nixpkgs> {} }:

let
  # We define a custom python environment with exactly the packages we need.
  # This builds them from source or fetches cached binaries from NixOS cache.
  my-python = pkgs.python3.withPackages (ps: with ps; [
    lancedb
    sentence-transformers
    numpy
    typer
    rich
  ]);
in
pkgs.mkShell {
  buildInputs = [
    # Python fallback / comparison
    my-python

    # Rust toolchain for the native port
    pkgs.rustc
    pkgs.cargo

    # Native deps commonly needed by Rust crates
    pkgs.pkg-config
    pkgs.openssl
    pkgs.cacert

  ];

  shellHook = ''
    # Ensure TLS works on NixOS for tools that rely on env probing.
    export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
    export SSL_CERT_DIR="${pkgs.cacert}/etc/ssl/certs"
    export NIX_SSL_CERT_FILE="$SSL_CERT_FILE"
    export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"


    # Define `mem` as a shell function so it works
    # both interactively and via `nix-shell --run`.
    mem() { cargo run --quiet -- "$@"; }
    mem-py() { python "$PWD/cli.py" "$@"; }
    export -f mem mem-py

    echo "Mandrid ready. Try: mem --help"
  '';
}
