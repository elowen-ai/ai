Install the tokenizer: [Big-Tiger-Gemma-27B-v1/blob/main/tokenizer.json](https://huggingface.co/TheDrummer/Big-Tiger-Gemma-27B-v1/blob/main/tokenizer.json)

Run the tokenizer fixing script:
`python fix_tokenizer.py`

Serve the model:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model "TheDrummer/Big-Tiger-Gemma-27B-v1" \
  --tokenizer tokenizer.fixed \
  --dtype auto \
  --max-model-len 8192 \
  --port 8000
```

Serve the app: `uvicorn ai:app --host 0.0.0.0 --port 8001 --reload`