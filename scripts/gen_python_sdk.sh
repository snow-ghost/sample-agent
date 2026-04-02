#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/bitgn-local-sdk/src"
PROTO_DIR="$ROOT_DIR/proto"

rm -rf "$OUT_DIR/bitgn"
mkdir -p "$OUT_DIR/bitgn/vm"
cat > "$OUT_DIR/bitgn/__init__.py" <<'PY'
"""Locally generated BitGN protocol modules."""
PY
cat > "$OUT_DIR/bitgn/vm/__init__.py" <<'PY'
"""Locally generated BitGN VM protocol modules."""
PY

PROTO_INCLUDE="$(
  uvx --from grpcio-tools python - <<'PY'
import pathlib
import grpc_tools

print(pathlib.Path(grpc_tools.__file__).resolve().parent / "_proto")
PY
)"

uvx \
  --from grpcio-tools \
  --with protoc-gen-connect-python \
  --with protobuf \
  --with connectrpc \
  python -m grpc_tools.protoc \
  -I "$PROTO_DIR" \
  -I "$PROTO_INCLUDE" \
  --python_out="$OUT_DIR" \
  --connect-python_out="$OUT_DIR" \
  "$PROTO_DIR/bitgn/harness.proto" \
  "$PROTO_DIR/bitgn/vm/mini.proto" \
  "$PROTO_DIR/bitgn/vm/pcm.proto"
