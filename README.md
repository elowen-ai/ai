`python fix_tokenizer.py`

Serve the model: 
`python -m vllm.entrypoints.openai.api_server \
  --model "TheDrummer/Big-Tiger-Gemma-27B-v1" \
  --tokenizer tokenizer.fixed \
  --dtype auto \
  --max-model-len 8192 \
  --port 8000
`

`pip install "uvicorn[standard]" python-socketio[asgi] "openai>=1.40.0"`

Serve the app: 
`uvicorn ai:app --host 0.0.0.0 --port 8001 --reload`

`python test.py`