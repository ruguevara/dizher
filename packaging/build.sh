#!/bin/sh
# Builds dist/Dizher (dist/Dizher.app on macOS) with the active env's python (pip install pyinstaller first).
# Ops are resolved by "module:name" strings and discovered via mokit.ops entry points, and imageio reads its own
# metadata, hence collect-submodules and copy-metadata; data covers the blue-noise PNGs and imgui_bundle's fonts.
set -e
cd "$(dirname "$0")/.."
python -m PyInstaller --noconfirm --windowed --name Dizher \
  --osx-bundle-identifier ru.ruguevara.dizher \
  --collect-submodules dizher --collect-submodules mokit \
  --collect-data dizher --collect-data mokit --collect-data imgui_bundle \
  --collect-submodules imageio --recursive-copy-metadata dizher \
  --workpath build/pyinstaller --specpath build/pyinstaller \
  packaging/launcher.py
