import socketio

from auth import Auth
from prompts import PromptManager
from ai import GenDefaults, LLMClient, ModelConfig, RPFormatter, sanitize_history

auth = Auth()

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
        self.app: socketio.ASGIApp[socketio.AsyncServer] = socketio.ASGIApp(self.sio)
        self.llm: LLMClient = llm
        self.fmt: RPFormatter = formatter
        self.defaults: GenDefaults = defaults
        self._register_events()

    def _register_events(self):
        @self.sio.event
        async def connect(sid, data, authData):
            print(f"[SIO] TRIES TO CONNECT: {sid}", flush=True)
            response = auth.authenticate(sid, authData)

            if response.get("status") is False:
                return response

            print("Client connected:", sid)
            print("Successfully authenticated")

            return response

        @self.sio.on("rp_start")
        @auth.isAuthenticated
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
        @auth.isAuthenticated
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

def create_app() -> socketio.ASGIApp:
    cfg = ModelConfig()

    llm = LLMClient(cfg)
    defaults = GenDefaults()
    formatter = RPFormatter(PromptManager())

    server = RPServer(llm, formatter, defaults)
    return server.app

app = create_app()