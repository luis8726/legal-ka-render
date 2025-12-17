#!/usr/bin/env bash
set -e

echo "🚀 Iniciando Legal KA..."

DATA_ROOT="${RENDER_DISK_PATH:-/var/data}"
INDEX_ROOT="$DATA_ROOT/data/index"
BUNDLE_URL="${INDEX_BUNDLE_URL:-}"

BM25_FILE="$INDEX_ROOT/bm25.pkl"
META_FILE="$INDEX_ROOT/meta.pkl"

echo "📁 DATA_ROOT = $DATA_ROOT"
echo "📁 INDEX_ROOT = $INDEX_ROOT"
echo "📄 BM25_FILE  = $BM25_FILE"
echo "📄 META_FILE  = $META_FILE"

mkdir -p "$DATA_ROOT"

need_restore="false"
if [ ! -d "$INDEX_ROOT" ]; then
  need_restore="true"
elif [ ! -f "$BM25_FILE" ] || [ ! -f "$META_FILE" ]; then
  need_restore="true"
fi

if [ "$need_restore" = "true" ]; then
  echo "📦 Índice incompleto o inexistente. Restaurando desde bundle..."

  if [ -z "$BUNDLE_URL" ]; then
    echo "❌ ERROR: falta INDEX_BUNDLE_URL en variables de entorno"
    exit 1
  fi

  # Limpieza para evitar estados viejos/incompletos
  rm -rf "$DATA_ROOT/data/index"
  mkdir -p "$DATA_ROOT/data"

  curl -L "$BUNDLE_URL" -o /tmp/index_bundle.tar.gz
  tar -xzf /tmp/index_bundle.tar.gz -C "$DATA_ROOT"

  # Validación post-extract
  if [ ! -f "$BM25_FILE" ] || [ ! -f "$META_FILE" ]; then
    echo "❌ ERROR: Después de extraer, siguen faltando bm25.pkl o meta.pkl."
    echo "   Esto indica que el bundle no tiene la estructura 'data/index/...'."
    exit 1
  fi

  echo "✅ Índice restaurado correctamente"
else
  echo "✅ Índice ya presente y completo. No se descarga nada."
fi

streamlit run src/app.py \
  --server.port $PORT \
  --server.address 0.0.0.0
