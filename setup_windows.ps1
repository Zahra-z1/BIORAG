$ErrorActionPreference = 'Stop'
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env -ErrorAction SilentlyContinue
python -m scripts.build_index --lexical-only
Write-Host 'Setup complete.'
Write-Host 'Frontend: streamlit run app.py'
Write-Host 'Optional REST API: uvicorn api:app --reload --port 8000'
Write-Host 'For semantic retrieval, run later: python -m scripts.build_index'
