from transformers import PreTrainedTokenizerFast

SRC = "tokenizer/tokenizer.json" # tokenizer.json path
DST = "tokenizer.fixed" # new output folder

CHAT_TEMPLATE = r"""{{- bos_token -}}
{%- if messages[0]["role"] == "system" -%}
  {{- raise_exception("System role not supported") -}}
{%- endif -%}
{%- for message in messages -%}
  {%- if message["role"] == "user" != (loop.index0 % 2 == 0) -%}
    {{- raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") -}}
  {%- endif -%}
  {%- if message["role"] == "assistant" -%}
    {%- set role = "model" -%}
  {%- else -%}
    {%- set role = message["role"] -%}
  {%- endif -%}
  {{- "<start_of_turn>" + role + "\n" + message["content"] | trim + "<end_of_turn>\n" -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
  {{- "<start_of_turn>model\n" -}}
{%- endif -%}
"""

tok = PreTrainedTokenizerFast(tokenizer_file=SRC)

# additional specials
need = ["<start_of_turn>", "<end_of_turn>"]
existing = set(tok.get_vocab().keys())
to_add = [t for t in need if t not in existing]
if to_add:
    tok.add_special_tokens({"additional_special_tokens": to_add})

# core specials
if tok.bos_token is None: tok.add_special_tokens({"bos_token": "<bos>"})
if tok.eos_token is None: tok.add_special_tokens({"eos_token": "<eos>"})
if tok.unk_token is None: tok.add_special_tokens({"unk_token": "<unk>"})
if tok.pad_token is None:
    if "<pad>" in tok.get_vocab(): tok.add_special_tokens({"pad_token": "<pad>"})
    else: tok.pad_token = tok.eos_token

# Attach template and save
tok.chat_template = CHAT_TEMPLATE
tok.save_pretrained(DST)
print("Tokenizer path", DST)