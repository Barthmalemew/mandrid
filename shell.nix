{ pkgs ? import <nixpkgs> {} }:

let
  # We define a custom python environment with exactly the packages we need.
  # This builds them from source or fetches cached binaries from NixOS cache.
  my-python = pkgs.python3.withPackages (ps: with ps; [
    lancedb
    sentence-transformers
    numpy
    typer
  ]);
in
pkgs.mkShell {
  buildInputs = [
    my-python
  ];

shellHook = ''
    # ... existing code ...
    
    # Create an alias so 'mem' runs your python script
    alias mem="python $PWD/cli.py"
    
    echo "Memory Agent ready. Type 'mem --help' to start."
  '';
}
