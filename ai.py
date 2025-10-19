from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from openai import AsyncOpenAI, BadRequestError
from prompts import PromptManager

import socketio

# ----CONFIGURATION------
@dataclass(frozen=True)
class ModelConfig:
    base_url: str = "http://localhost:8000/v1"
    model_name: str = "TheDrummer/Big-Tiger-Gemma-27B-v1"

@dataclass
class GenDefaults:
    temperature: float = 0.9
    top_p: float = 0.9
    max_tokens: int = 768
    stop: List[str] = None
    frequency_penalty: float = 0.2
    presence_penalty: float = 0.2
    extra_body: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {"repetition_penalty": 1.08}
    )

    def __post_init__(self):
        if self.stop is None:
            self.stop = ["<end_of_turn>"]

    def as_kwargs(self) -> Dict[str, Any]:
        kw = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stop": self.stop,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        if self.extra_body:
            kw["extra_body"] = self.extra_body
        return kw

# ----------- LLM WRAPPER -------
class LLMClient:
    def __init__(self, cfg: ModelConfig):
        self.client = AsyncOpenAI(base_url=cfg.base_url, api_key="EMPTY")
        self.model = cfg.model_name

    async def stream_completion(self, prompt: str, **kwargs):
        return await self.client.completions.create(
            model=self.model, prompt=prompt, stream=True, **kwargs
        )

    async def completion_once(self, prompt: str, **kwargs) -> str:
        resp = await self.client.completions.create(
            model=self.model, prompt=prompt, **kwargs
        )
        return (resp.choices[0].text or "").strip()
    
# ---------- PROMPT BUILDER ----------
class RPFormatter:
    def __init__(self, prompt_manager: PromptManager):
        self.pm = prompt_manager

    def _preamble(self, character: Dict) -> str:
        name = character.get("name", "")
        description = character.get("description", "")
        personality = character.get("personality", "")
        speaking_style = character.get("speakingStyle") or character.get("speaking_style") or ""
        samples = character.get("samples", []) or []
        card = self.pm.generate_character_prompt(
            name=name,
            description=description,
            personality=personality,
            speakingStyle=speaking_style,
            samples=samples,
        )
        return self.pm.build_preamble(card)

    @staticmethod
    def _render_turns(hist: List[Dict[str, str]]) -> str:
        # hist already sanitized to alternate, end with assistant
        blocks = []
        for m in hist:
            role = (m.get("role") or "").lower()
            content = (m.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                blocks.append(f"<start_of_turn>{role}\n{content}\n<end_of_turn>")
        return "\n".join(blocks)

    def build_manual_prompt(self, character: Dict, history: List[Dict[str, str]], user_input: str) -> str:
        pre = self._preamble(character)
        hist_txt = self._render_turns(history)
        ui = (user_input or "").strip()
        return (
            "<start_of_turn>user\n"
            f"{pre}\n\n"
            "## Conversation So Far\n"
            f"{hist_txt}\n"
            "## Conversation Continues\n"
            f"{ui}\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

# --------- HISTORY SANITIZER ---------
def sanitize_history(raw_hist: Optional[List[Dict[str, str]]], max_pairs: int = 10) -> List[Dict[str, str]]:
    clean = []
    for m in (raw_hist or []):
        role = (m.get("role") or "").lower()
        content = (m.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            clean.append({"role": role, "content": content})

    # force start with user
    while clean and clean[0]["role"] != "user":
        clean.pop(0)

    out, expect = [], "user"
    for m in clean:
        if m["role"] == expect:
            out.append(m)
            expect = "assistant" if expect == "user" else "user"

    # must end with assistant
    if out and out[-1]["role"] != "assistant":
        out = out[:-1]

    # keep only last N pairs
    if len(out) > 2 * max_pairs:
        out = out[-2 * max_pairs :]

    return out
    
# --------------SOCKET.IO SERVER----------
class RPServer:
    def __init__(self, llm: LLMClient, formatter: RPFormatter, defaults: GenDefaults):
        self.sio = socketio.AsyncServer(
            async_mode="asgi",
            cors_allowed_origins="*",
            ping_interval=25,
            ping_timeout=60,
            logger=True,            
            engineio_logger=True,
        )
        self.app = socketio.ASGIApp(self.sio)
        self.llm = llm
        self.fmt = formatter
        self.defaults = defaults
        self._register_events()

    def _register_events(self):
        @self.sio.event
        async def connect(sid, environ):
            print(f"[SIO] CONNECTED: {sid}", flush=True)
            await self.sio.emit("gpu_connected", {"ok": True}, to=sid)

        @self.sio.on("rp_start")
        async def rp_start(sid, data):
            rid = data.get("request_id", "")
            try:
                kwargs = self.defaults.as_kwargs()
                character = data["character"]
                user_input = data["user_input"]
                history = sanitize_history(data.get("history"))

                prompt = self.fmt.build_manual_prompt(character, history, user_input)
                stream = await self.llm.stream_completion(prompt, **kwargs)

                full = []
                async for chunk in stream:
                    tok = chunk.choices[0].text if (chunk.choices and chunk.choices[0].text) else None
                    if tok:
                        full.append(tok)
                        await self.sio.emit("rp_token", {"request_id": rid, "token": tok}, to=sid)
                await self.sio.emit("rp_done", {"request_id": rid, "text": "".join(full).strip()}, to=sid)

            except Exception as e:
                await self.sio.emit("rp_error", {"request_id": rid, "message": str(e)}, to=sid)

        @self.sio.on("rp_once")
        async def rp_once(sid, data):
            rid = data.get("request_id", "")
            try:
                kwargs = self.defaults.as_kwargs()
                character = data["character"]
                user_input = data["user_input"]
                history = sanitize_history(data.get("history"))

                prompt = self.fmt.build_manual_prompt(character, history, user_input)
                text = await self.llm.completion_once(prompt, **kwargs)
                await self.sio.emit("rp_once_result", {"request_id": rid, "text": text}, to=sid)

            except Exception as e:
                await self.sio.emit("rp_error", {"request_id": rid, "message": str(e)}, to=sid)

# ----------ASSEMBLE APP-----------
def create_app() -> socketio.ASGIApp:
    cfg = ModelConfig()
    defaults = GenDefaults()
    llm = LLMClient(cfg)
    formatter = RPFormatter(PromptManager())
    server = RPServer(llm, formatter, defaults)
    return server.app

app = create_app()