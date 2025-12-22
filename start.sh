#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Iniciando Legal KA..."

DATA_ROOT="${RENDER_DISK_PATH:-/var/data}"
BUNDLE_URL="${INDEX_BUNDLE_URL:-}"

# Donde queremos que termine el índice
INDEX_ROOT="${INDEX_DIR:-$DATA_ROOT/index}"

# ✅ Todo adentro de INDEX_ROOT (coherente con el bundle)
CHROMA_DIR="${CHROMA_DIR:-$INDEX_ROOT/chroma}"
BM25_PATH="${BM25_PATH:-$INDEX_ROOT/bm25.pkl}"
META_PATH="${META_PATH:-$INDEX_ROOT/meta.pkl}"

echo "📁 DATA_ROOT   = $DATA_ROOT"
echo "📁 INDEX_ROOT  = $INDEX_ROOT"
echo "📁 CHROMA_DIR  = $CHROMA_DIR"
echo "📄 BM25_PATH   = $BM25_PATH"
echo "📄 META_PATH   = $META_PATH"
echo "🌐 BUNDLE_URL  = ${BUNDLE_URL:+(set)}"

mkdir -p "$DATA_ROOT"
mkdir -p "$INDEX_ROOT"

# Export para que tu config.py use estos paths
export INDEX_DIR="$INDEX_ROOT"
export CHROMA_DIR="$CHROMA_DIR"
export BM25_PATH="$BM25_PATH"
export META_PATH="$META_PATH"

need_restore="false"
if [ ! -d "$CHROMA_DIR" ] || [ ! -f "$BM25_PATH" ] || [ ! -f "$META_PATH" ]; then
  need_restore="true"
fi

if [ "$need_restore" = "true" ]; then
  echo "📦 Índice incompleto o inexistente. Restaurando desde bundle..."

  if [ -z "$BUNDLE_URL" ]; then
    echo "❌ ERROR: falta INDEX_BUNDLE_URL en variables de entorno"
    exit 1
  fi

  rm -rf "$INDEX_ROOT"
  mkdir -p "$INDEX_ROOT"

  curl -L "$BUNDLE_URL" -o /tmp/index_bundle.tar.gz
  tar -xzf /tmp/index_bundle.tar.gz -C "$DATA_ROOT"

  # Si el bundle trae data/index, lo movemos a INDEX_ROOT
  if [ -d "$DATA_ROOT/data/index" ] && [ ! -d "$INDEX_ROOT/chroma" ]; then
    echo "🔁 Bundle trae estructura data/index. Moviendo a INDEX_ROOT..."
    rm -rf "$INDEX_ROOT"
    mkdir -p "$(dirname "$INDEX_ROOT")"
    mv "$DATA_ROOT/data/index" "$INDEX_ROOT"
  fi

  # Validación final (coherente)
  if [ ! -d "$CHROMA_DIR" ] || [ ! -f "$BM25_PATH" ] || [ ! -f "$META_PATH" ]; then
    echo "❌ ERROR: Después de extraer, el índice sigue incompleto."
    echo "   Esperaba: $CHROMA_DIR y bm25/meta en $INDEX_ROOT"
    echo "   Estructura real:"
    ls -la "$DATA_ROOT" || true
    ls -la "$INDEX_ROOT" || true
    exit 1
  fi

  echo "✅ Índice restaurado correctamente"
else
  echo "✅ Índice ya presente y completo. No se descarga nada."
fi

streamlit run src/app.py --server.port "${PORT:-8501}" --server.address 0.0.0.0
