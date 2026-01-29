#!/bin/bash
set -e

# Mandrid Installer
# Builds the release binary and installs it to ~/.local/bin

echo "⚡ Building Mandrid (mem)..."

# Ensure we are in the repo root
cd "$(dirname "$0")"

# Build release binary (using nix-shell wrapper if available, else direct cargo)
if command -v nix-shell >/dev/null; then
    nix-shell --run "cargo build --release"
else
    cargo build --release
fi

BINARY_PATH="target/release/mem"
INSTALL_DIR="$HOME/.local/bin"
INSTALL_PATH="$INSTALL_DIR/mem"

if [ ! -f "$BINARY_PATH" ]; then
    echo "❌ Build failed. Binary not found at $BINARY_PATH"
    exit 1
fi

echo "📦 Installing to $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR"
cp "$BINARY_PATH" "$INSTALL_PATH"

echo "✅ Installed 'mem' to $INSTALL_PATH"
echo "   Verifying..."
"$INSTALL_PATH" --version

echo ""
echo "🎉 Done! You can now run 'mem init' in any project."
echo "   (Ensure $INSTALL_DIR is in your \$PATH)"
