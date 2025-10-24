from app.prompts import PromptManager
from dataclasses import dataclass, field
from openai import AsyncOpenAI, AsyncStream
from openai.types.completion import Completion
from typing import Dict, Literal, Optional, List, Any
import os

# ----------- CONFIGURATION -----------
@dataclass(frozen=True)
class ModelConfig:
    base_url: str = os.getenv("VLLM_BASE_URL", "http://vllm:4001/v1")
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

# ----------- LLM WRAPPER -----------
class LLMClient:
    def __init__(self, cfg: ModelConfig):
        self.client = AsyncOpenAI(base_url=cfg.base_url, api_key="EMPTY")
        self.model = cfg.model_name

    async def stream_completion(self, prompt: str, **kwargs) -> AsyncStream[Completion]:
        return await self.client.completions.create(
            model=self.model, prompt=prompt, stream=True, **kwargs
        )

    async def completion_once(self, prompt: str, **kwargs) -> str:
        resp = await self.client.completions.create(
            model=self.model, prompt=prompt, **kwargs
        )
        return (resp.choices[0].text or "").strip()
    
# ----------- PROMPT BUILDER -----------
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

    def build_twitter_prompt(self, character: Dict, previous_mentions: str) -> str:
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

        pre = self.pm.build_twitter_preamble(card)
        mentions = (previous_mentions or "").strip()

        return (
            "<start_of_turn>user\n"
            f"{pre}\n\n"
            "## Previous Mentions\n"
            f"{mentions}\n"
            "## Reply\n"
            "Compose your in-character reply to the latest mention only.\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

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

# ----------- HISTORY SANITIZER -----------
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
            expect: Literal['assistant'] | Literal['user'] = "assistant" if expect == "user" else "user"

    # must end with assistant
    if out and out[-1]["role"] != "assistant":
        out = out[:-1]

    # keep only last N pairs
    if len(out) > 2 * max_pairs:
        out = out[-2 * max_pairs :]

    return out
