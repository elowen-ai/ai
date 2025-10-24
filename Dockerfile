FROM python:3.10-slim

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

COPY . /workspace

EXPOSE 4002

CMD ["/bin/sh","-lc","python /workspace/app/fix_tokenizer.py && uvicorn server:app --host 0.0.0.0 --port 4002 --reload"]
