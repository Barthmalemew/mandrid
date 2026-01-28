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
    my-python
  ];

  shellHook = ''
    # Define `mem` as a shell function so it works
    # both interactively and via `nix-shell --run`.
    mem() { python "$PWD/cli.py" "$@"; }
    export -f mem

    echo "Memory Agent ready. Type 'mem --help' to start."
  '';
}
