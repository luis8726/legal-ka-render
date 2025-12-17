#!/usr/bin/env bash
set -e

echo "🚀 Iniciando Legal KA..."

streamlit run src/app.py \
  --server.port $PORT \
  --server.address 0.0.0.0
