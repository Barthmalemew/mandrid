{
  description = "Mandrid (mem) - Local persistent memory for AI agents";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    crane = {
      url = "github:ipetkov/crane";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, crane, fenix, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Use fenix to get a minimal rust toolchain if needed, 
        # or just use pkgs.rustPlatform (crane uses its own logic usually).
        # We'll use the default stable toolchain.
        craneLib = crane.lib.${system};

        # Common native dependencies
        buildInputs = with pkgs; [
          openssl
          onnxruntime
        ];

        nativeBuildInputs = with pkgs; [
          pkg-config
          makeWrapper
        ];

        # Def to build the crate
        mandrid = craneLib.buildPackage {
          src = craneLib.cleanCargoSource (craneLib.path ./.);
          strictDeps = true;

          inherit buildInputs nativeBuildInputs;

          # Fix for compiling openssl-sys
          OPENSSL_NO_VENDOR = 1;
          OPENSSL_DIR = "${pkgs.openssl.dev}";
          OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";

          # Ensure we can find onnxruntime at runtime
          postInstall = ''
            wrapProgram $out/bin/mem \
              --set ORT_DYLIB_PATH "${pkgs.onnxruntime}/lib/libonnxruntime.so" \
              --prefix LD_LIBRARY_PATH : "${pkgs.onnxruntime}/lib" \
              --set SSL_CERT_FILE "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
          '';
        };

      in {
        packages.default = mandrid;
        packages.mandrid = mandrid;

        apps.default = flake-utils.lib.mkApp {
          drv = mandrid;
          name = "mem";
        };

        devShells.default = pkgs.mkShell {
          inputsFrom = [ mandrid ];
          packages = with pkgs; [
            cargo
            rustc
            rustfmt
            clippy
            sqlite
          ];

          shellHook = ''
            export ORT_DYLIB_PATH="${pkgs.onnxruntime}/lib/libonnxruntime.so"
            export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
            echo "Mandrid Dev Shell"
          '';
        };
      }
    ) // {
      # Overlay for system configurations to use:
      # nixpkgs.overlays = [ mandrid.overlays.default ];
      overlays.default = final: prev: {
        mandrid = self.packages.${prev.system}.default;
      };
    };
}
