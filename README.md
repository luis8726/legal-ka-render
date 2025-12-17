# Legal KA (Societario) - Local RAG

## 1) Requisitos
- Python 3.10+
- API Key de OpenAI

## 2) Setup
```bash
cd legal-ka
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# Editar .env y poner OPENAI_API_KEY=...
