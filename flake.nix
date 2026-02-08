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

        # Crane requires us to instantiate the library with our pkgs
        craneLib = crane.mkLib pkgs;

        # Common native dependencies
        commonArgs = {
          src = craneLib.cleanCargoSource (craneLib.path ./.);
          strictDeps = true;

          buildInputs = with pkgs; [
            openssl
            onnxruntime
          ];

          nativeBuildInputs = with pkgs; [
            pkg-config
            makeWrapper
            protobuf
          ];

          # Fix for compiling lance/prost (protobuf)
          PROTOC = "${pkgs.protobuf}/bin/protoc";

          # Fix for compiling openssl-sys
          OPENSSL_NO_VENDOR = 1;
          OPENSSL_DIR = "${pkgs.openssl.dev}";
          OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
        };

        # Build *only* the cargo dependencies, to cache them
        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        # Build the actual crate
        mandrid = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;

          # Ensure we can find onnxruntime at runtime
          postInstall = ''
            wrapProgram $out/bin/mem \
              --set ORT_DYLIB_PATH "${pkgs.onnxruntime}/lib/libonnxruntime.so" \
              --prefix LD_LIBRARY_PATH : "${pkgs.onnxruntime}/lib" \
              --set SSL_CERT_FILE "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
          '';
        });

        # Fast-path: Download pre-built binary from GitHub
        mandrid-bin = pkgs.stdenv.mkDerivation rec {
          pname = "mandrid-bin";
          version = "0.1.5";

          src = pkgs.fetchurl {
            url = "https://github.com/Barthmalemew/mandrid/releases/download/v${version}/mandrid-linux-amd64";
            # This is a placeholder; you'll get a hash error once, then copy the real hash
            sha256 = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
          };

          nativeBuildInputs = [ pkgs.makeWrapper ];
          phases = [ "installPhase" ];

          installPhase = ''
            mkdir -p $out/bin
            cp $src $out/bin/mem
            chmod +x $out/bin/mem
            wrapProgram $out/bin/mem \
              --set ORT_DYLIB_PATH "${pkgs.onnxruntime}/lib/libonnxruntime.so" \
              --prefix LD_LIBRARY_PATH : "${pkgs.onnxruntime}/lib" \
              --set SSL_CERT_FILE "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
          '';
        };

      in {
        packages.default = mandrid;
        packages.mandrid = mandrid;
        packages.bin = mandrid-bin;

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
