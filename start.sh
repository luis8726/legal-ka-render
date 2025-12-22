#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Iniciando Legal KA..."

# -----------------------------
# 1) Disk persistente (Render)
# -----------------------------
DATA_ROOT="${RENDER_DISK_PATH:-/var/data}"
BUNDLE_URL="${INDEX_BUNDLE_URL:-}"

# ✅ Nuevo: Permitimos definir INDEX_DIR desde Render (recomendado)
# Si no está seteado, usamos /var/data/index
INDEX_ROOT="${INDEX_DIR:-$DATA_ROOT/index}"

# También permitimos granularidad
CHROMA_DIR_ENV="${CHROMA_DIR:-$INDEX_ROOT/chroma}"
BM25_FILE_ENV="${BM25_PATH:-$INDEX_ROOT/bm25.pkl}"
META_FILE_ENV="${META_PATH:-$INDEX_ROOT/meta.pkl}"

echo "📁 DATA_ROOT        = $DATA_ROOT"
echo "📁 INDEX_ROOT       = $INDEX_ROOT"
echo "📁 CHROMA_DIR       = $CHROMA_DIR_ENV"
echo "📄 BM25_PATH        = $BM25_FILE_ENV"
echo "📄 META_PATH        = $META_FILE_ENV"
echo "🌐 INDEX_BUNDLE_URL = ${BUNDLE_URL:+(set)}"

mkdir -p "$DATA_ROOT"
mkdir -p "$INDEX_ROOT"

# Exportamos para que config.py los tome sí o sí
export INDEX_DIR="$INDEX_ROOT"
export CHROMA_DIR="$CHROMA_DIR_ENV"
export BM25_PATH="$BM25_FILE_ENV"
export META_PATH="$META_FILE_ENV"

# -----------------------------
# 2) ¿Hay que restaurar?
# -----------------------------
need_restore="false"

# Validación mínima: existen BM25 y META, y carpeta chroma
if [ ! -d "$CHROMA_DIR_ENV" ]; then
  need_restore="true"
elif [ ! -f "$BM25_FILE_ENV" ] || [ ! -f "$META_FILE_ENV" ]; then
  need_restore="true"
fi

if [ "$need_restore" = "true" ]; then
  echo "📦 Índice incompleto o inexistente. Restaurando desde bundle..."

  if [ -z "$BUNDLE_URL" ]; then
    echo "❌ ERROR: falta INDEX_BUNDLE_URL en variables de entorno"
    exit 1
  fi

  # Limpieza
  rm -rf "$INDEX_ROOT"
  mkdir -p "$INDEX_ROOT"

  curl -L "$BUNDLE_URL" -o /tmp/index_bundle.tar.gz
  tar -xzf /tmp/index_bundle.tar.gz -C "$DATA_ROOT"

  # -----------------------------
  # 3) Compatibilidad con bundles
  #    a) /var/data/data/index/...
  #    b) /var/data/index/...
  # -----------------------------
  if [ -d "$DATA_ROOT/data/index" ] && [ ! -d "$INDEX_ROOT/chroma" ]; then
    echo "🔁 Bundle trae estructura data/index. Moviendo a INDEX_ROOT..."
    rm -rf "$INDEX_ROOT"
    mkdir -p "$(dirname "$INDEX_ROOT")"
    mv "$DATA_ROOT/data/index" "$INDEX_ROOT"
  fi

  # Validación post-restore
  if [ ! -d "$CHROMA_DIR_ENV" ] || [ ! -f "$BM25_FILE_ENV" ] || [ ! -f "$META_FILE_ENV" ]; then
    echo "❌ ERROR: Después de extraer, el índice sigue incompleto."
    echo "   Esperaba: $CHROMA_DIR_ENV y archivos bm25/meta en $INDEX_ROOT"
    echo "   Revisá la estructura dentro del bundle."
    exit 1
  fi

  echo "✅ Índice restaurado correctamente"
else
  echo "✅ Índice ya presente y completo. No se descarga nada."
fi

# -----------------------------
# 4) Arrancar Streamlit
# -----------------------------
streamlit run src/app.py \
  --server.port "${PORT:-8501}" \
  --server.address 0.0.0.0
