set -euo pipefail

if [ -z "$1" ]; then
    echo "Usage: $0 <Path to Blender Binary>"
    exit 1
fi

BLENDER_BIN="$1"
BASE_DIR="$PWD"
SRC_DIR="${BASE_DIR}/src"
BUILD_DIR="${BASE_DIR}/build"
TEMP_DIR="${BASE_DIR}/src_build_tmp"
TEMP_SRC="${TEMP_DIR}/src"
WHEELS_DIR="${TEMP_SRC}/wheels"
MANIFEST="${TEMP_SRC}/blender_manifest.toml"

rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"
mkdir -p "$BUILD_DIR"

# Copy the entire src directory into the temp workspace
cp -r "$SRC_DIR" "$TEMP_DIR"

# Switch into the temp workspace
cd "$TEMP_DIR/src"

mkdir -p "$WHEELS_DIR"
cd "$WHEELS_DIR"
echo "Downloading wheels..."
# PyTorch wheels
curl -O https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp311-cp311-linux_x86_64.whl
curl -O https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp311-cp311-win_amd64.whl
curl -O https://download.pytorch.org/whl/cpu/torch-2.6.0-cp311-none-macosx_11_0_arm64.whl
# SciPy wheels
pip download scipy --only-binary=:all: --platform manylinux_2_17_x86_64 --python-version 3.11 --implementation cp
pip download scipy --only-binary=:all: --platform win_amd64 --python-version 3.11 --implementation cp
pip download scipy --only-binary=:all: --platform macosx_14_0_arm64 --python-version 3.11 --implementation cp

echo "Appending wheels list to manifest..."
# Build TOML list directly
echo $'\n' >> $MANIFEST
wheelstr="wheels = ["
for wheel in "$WHEELS_DIR"/*; do
    if [ -f "$wheel" ] && [[ "$wheel" == *.whl ]]; then
        wheelstr+="\"./wheels/$(basename "$wheel")\", "
    fi 
done
wheelstr="${wheelstr%, }]"
wheelstr+=$'\n'

echo $wheelstr >> $MANIFEST

echo "Wheels block appended."

echo "Building extension..."

cd "$TEMP_SRC"
# Specify the output directory (Blender supports --output)
$BLENDER_BIN --command extension build --split-platforms --output-dir "$BUILD_DIR"

echo "Cleaning up..."
rm -rf "$TEMP_DIR"

echo "Build complete. Files in $BUILD_DIR"
