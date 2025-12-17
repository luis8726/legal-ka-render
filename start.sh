#!/usr/bin/env bash
set -e

echo "🚀 Iniciando Legal KA..."

# Path base del disk persistente
DATA_ROOT="${RENDER_DISK_PATH:-/var/data}"

# Donde vive tu índice (según tu estructura actual)
INDEX_ROOT="$DATA_ROOT/data/index"

# URL del bundle (GitHub Release)
BUNDLE_URL="${INDEX_BUNDLE_URL:-}"

echo "📁 DATA_ROOT = $DATA_ROOT"
echo "📁 INDEX_ROOT = $INDEX_ROOT"

# Crear carpeta base si no existe
mkdir -p "$DATA_ROOT"

# Si el índice no existe o está vacío, lo restauramos
if [ ! -d "$INDEX_ROOT" ] || [ -z "$(ls -A "$INDEX_ROOT" 2>/dev/null)" ]; then
  echo "📦 Índice no encontrado. Restaurando desde bundle..."
  
  if [ -z "$BUNDLE_URL" ]; then
    echo "❌ ERROR: falta INDEX_BUNDLE_URL en variables de entorno"
    exit 1
  fi

  curl -L "$BUNDLE_URL" -o /tmp/index_bundle.tar.gz
  tar -xzf /tmp/index_bundle.tar.gz -C "$DATA_ROOT"

  echo "✅ Índice restaurado correctamente"
else
  echo "✅ Índice ya presente. No se descarga nada."
fi

# Arrancar Streamlit
streamlit run src/app.py \
  --server.port $PORT \
  --server.address 0.0.0.0

