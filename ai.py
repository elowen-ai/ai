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

    async def stream_chat(self, messages: List[Dict[str, str]], **kwargs):
        return await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **kwargs
        )
    
    async def chat_once(self, messages: List[Dict[str, str]], **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return (response.choices[0].message.content or "").strip()

    # --- Fallback: completions (manual prompt) ---
    async def stream_completion(self, prompt: str, **kwargs):
        return await self.client.completions.create(
            model=self.model,
            prompt=prompt,
            stream=True,
            **kwargs
        )

    async def completion_once(self, prompt: str, **kwargs) -> str:
        resp = await self.client.completions.create(
            model=self.model,
            prompt=prompt,
            **kwargs
        )
        return (resp.choices[0].text or "").strip()
    
# ---------- PROMPT BUILDER ----------
class RPFormatter:
    def __init__(self, prompt_manager: PromptManager):
        self.pm = prompt_manager

    @staticmethod
    def _san(text: str) -> str:
        return (text or "").replace("<end_of_turn>", "< end_of_turn >").strip()

    def _preamble(self, character: Dict) -> str:
        name = character["name"]
        description = character["description"]
        personality = character["personality"]
        speaking_style = character.get("speakingStyle") or character.get("speaking_style") or ""
        samples = character.get("samples", [])
        char_prompt = self.pm.generate_character_prompt(
            name=name,
            description=description,
            personality=personality,
            speakingStyle=speaking_style,
            samples=samples,
        )
        return self.pm.build_preamble(char_prompt)

    def build_first_user_message(self, character: Dict, user_input: str) -> List[Dict[str, str]]:
        preamble = self._preamble(character)
        user_text = self._san(user_input) + " Keep the response under 30–40 words."
        return [{"role": "user", "content": f"{preamble}\n\n## Conversation Starts\n{user_text}"}]

    def _assert_alternation_for_history(self, hist: List[Dict[str, str]]):
        # history must be alternating and end with assistant (so we can append new user)
        for i, m in enumerate(hist):
            r = (m.get("role") or "").lower()
            if i % 2 == 0 and r != "user":
                raise ValueError("History must start with 'user' and alternate.")
            if i % 2 == 1 and r != "assistant":
                raise ValueError("History must alternate user/assistant.")
        if hist and (hist[-1].get("role") or "").lower() != "assistant":
            raise ValueError("History must end with 'assistant' before appending the new user turn.")

    def build_messages_with_preamble_every_time(
        self,
        character: Dict,
        user_input: str,
        history: Optional[List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        
        hist = history or []
        hist = [{"role": (m["role"]).lower(), "content": self._san(m.get("content", ""))} for m in hist]

        if not hist:
            return self.build_first_user_message(character, user_input)

        self._assert_alternation_for_history(hist)

        if not hist[0]["content"].lstrip().startswith("## System Instructions"):
            hist[0]["content"] = f"{self._preamble(character)}\n\n{hist[0]['content'].strip()}"

        hist.append({"role": "user", "content": self._san(user_input)})

        if hist[-1]["role"] != "user":
            raise ValueError("Final message must be 'user'.")
        return hist

    # manual prompt for fallback on the very first turn only
    def build_manual_prompt(self, character: Dict, user_input: str) -> str:
        pre = self._preamble(character)
        return (
            "<start_of_turn>user\n"
            f"{pre}\n\n## Conversation Starts\n{self._san(user_input)}\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

# --------------SOCKET.IO SERVER----------
class RPServer:
    def __init__(self, llm: LLMClient, formatter: RPFormatter, defaults: GenDefaults):
        self.sio = socketio.AsyncServer(
            async_mode="asgi",
            cors_allowed_origins="*",
            ping_interval=25,
            ping_timeout=60,
        )
        self.app = socketio.ASGIApp(self.sio)
        self.llm = llm
        self.fmt = formatter
        self.defaults = defaults
        self._register_events()

    def _register_events(self):
        @self.sio.event
        async def connect(sid, environ):
            await self.sio.emit("gpu_connected", {"ok": True}, to=sid)

        @self.sio.on("rp_start")
        async def rp_start(sid, data):
            rid = data.get("request_id", "")
            try:
                kwargs = self.defaults.as_kwargs()

                messages = self.fmt.build_messages_with_preamble_every_time(
                    character=data["character"],
                    user_input=data["user_input"],
                    history=data.get("history"),
                )

                try:
                    stream = await self.llm.stream_chat(messages, **kwargs)
                    full = []
                    async for chunk in stream:
                        delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
                        if not delta:
                            continue
                        full.append(delta)
                        await self.sio.emit("rp_token", {"request_id": rid, "token": delta}, to=sid)
                    await self.sio.emit("rp_done", {"request_id": rid, "text": "".join(full).strip()}, to=sid)
                    return
                except BadRequestError as bre:
                    # Only fall back for alternation error on the *first* turn (no history)
                    msg = (getattr(bre, "message", None) or str(bre)).lower()
                    if "conversation roles must alternate" in msg and not data.get("history"):
                        prompt = self.fmt.build_manual_prompt(data["character"], data["user_input"])
                        stream = await self.llm.stream_completion(prompt, **kwargs)
                        full = []
                        async for chunk in stream:
                            tok = chunk.choices[0].text if (chunk.choices and chunk.choices[0].text) else None
                            if not tok:
                                continue
                            full.append(tok)
                            await self.sio.emit("rp_token", {"request_id": rid, "token": tok}, to=sid)
                        await self.sio.emit("rp_done", {"request_id": rid, "text": "".join(full).strip()}, to=sid)
                        return
                    raise

            except Exception as e:
                await self.sio.emit("rp_error", {"request_id": rid, "message": str(e)}, to=sid)

        @self.sio.on("rp_once")
        async def rp_once(sid, data):
            rid = data.get("request_id", "")
            try:
                kwargs = self.defaults.as_kwargs()

                messages = self.fmt.build_messages_with_preamble_every_time(
                    character=data["character"],
                    user_input=data["user_input"],
                    history=data.get("history"),
                )

                try:
                    text = await self.llm.chat_once(messages, **kwargs)
                    await self.sio.emit("rp_once_result", {"request_id": rid, "text": text}, to=sid)
                    return
                except BadRequestError as bre:
                    msg = (getattr(bre, "message", None) or str(bre)).lower()
                    if "conversation roles must alternate" in msg and not data.get("history"):
                        prompt = self.fmt.build_manual_prompt(data["character"], data["user_input"])
                        text = await self.llm.completion_once(prompt, **kwargs)
                        await self.sio.emit("rp_once_result", {"request_id": rid, "text": text}, to=sid)
                        return
                    raise
            except Exception as e:
                await self.sio.emit("rp_error", {"request_id": rid, "message": str(e)}, to=sid)

# ----------ASSEMBLE APP-----------
def create_app() -> socketio.ASGIApp:
    cfg = ModelConfig()
    defaults = GenDefaults(
        temperature=0.9,
        top_p=0.9,
        max_tokens=1024,
        stop=["<end_of_turn>"],
        frequency_penalty=0.2,
        presence_penalty=0.2,
        extra_body={"repetition_penalty": 1.08},
    )
    llm = LLMClient(cfg)
    formatter = RPFormatter(PromptManager())
    server = RPServer(llm, formatter, defaults)
    return server.app

app = create_app()