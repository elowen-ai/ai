# RP test without authentication

import asyncio
import socketio
import uuid

GPU_SIO_URL = "http://localhost:8001"

USER_INPUTS = [
    "What the hell is your name?",
    "So, Stark, how fast can that suit actually fly?",
    "If I borrow the suit, what’s the worst that could happen?",
    "Do you ever turn the sarcasm off, Tony?",
    "Arc reactor aside, what's your biggest fear?",
    "How many suits have you blown up this month?",
    "Level with me: are you more genius or more showman?",
    "Pepper says hi—are you behaving?",
    "Tell me the wildest upgrade idea you haven’t built yet.",
    "If I’m in danger, what’s the first thing you do?"
]

CHARACTER = {
    "name": "Iron Man",
    "description": "As a charismatic and highly intelligent inventor, Tony Stark, aka Iron Man, is a billionaire genius with a flair for the dramatic and a sharp sense of humor. After being kidnapped and injured, he built a high-tech suit to escape, which ultimately led him to become the iconic superhero, vowing to protect the world. With his quick wit and inventive mind, Iron Man combines intelligence, humor, and heroism to save the day.",
    "personality": "Witty, confident, sometimes reckless; passionate about technology and using it for good.",
    "speakingStyle": "Fast-paced, humorous, filled with sarcasm, and marked by clever one-liners.",
    "samples": [
        "I am Iron Man.",
        "Q: What's your secret? \nA: I never play this game to lose.",
        "Genius, billionaire, playboy, philanthropist.",
        "Q: How do you feel about being a hero?\nA: It's not about how much you have; it's about what you do with what you have.",
        "You’re seriously asking me for advice? Alright, fine. Step one—be a genius. Step two—be ridiculously good-looking. Step three—tilts head, smirks—wing it. Works every time."
    ]
}

async def main():
    sio = socketio.AsyncClient()

    buffers = {}
    done_events = {}
    errors = {}

    @sio.event
    async def connect():
        print("[client] connected:", sio.sid)

    @sio.event
    async def disconnect():
        print("[client] disconnected")

    @sio.on("gpu_connected")
    async def on_gpu_connected(msg):
        print("[gpu] connected:", msg)

    @sio.on("rp_token")
    async def on_rp_token(msg):
        rid = msg["request_id"]
        tok = msg["token"]
        buffers[rid].append(tok)

    @sio.on("rp_done")
    async def on_rp_done(msg):
        rid = msg["request_id"]
        text = msg["text"]
        buffers[rid] = [text]  # final text is consistent check
        if rid in done_events and not done_events[rid].is_set():
            done_events[rid].set()

    @sio.on("rp_error")
    async def on_rp_error(msg):
        rid = msg.get("request_id", "unknown")
        errors[rid] = msg.get("message", "unknown error")
        if rid in done_events and not done_events[rid].is_set():
            done_events[rid].set()

    await sio.connect(GPU_SIO_URL)

    tasks = []
    for idx, user_input in enumerate(USER_INPUTS):
        rid = f"test-{idx}-{uuid.uuid4().hex[:6]}"
        buffers[rid] = []
        done_events[rid] = asyncio.Event()

        payload = {
            "request_id": rid,
            "character": CHARACTER,
            "user_input": user_input,
            "history": [
                {"role" : "user", "content": ""},
                {"role" : "assistant", "content": ""}
            ]
        }

        # emit and wait in parallel
        async def run_one(_rid=rid, _payload=payload):
            await sio.emit("rp_start", _payload)
            await done_events[_rid].wait()

        tasks.append(asyncio.create_task(run_one()))

    # wait for all to finish
    await asyncio.gather(*tasks)
    await sio.disconnect()

    print("\n=== RESULTS ===")
    for idx, user_input in enumerate(USER_INPUTS):
        rid_prefix = f"test-{idx}-"
        rid = next((r for r in buffers.keys() if r.startswith(rid_prefix)), None)
        if not rid:
            print(f"\n[{rid_prefix}] MISSING")
            continue
        if rid in errors:
            print(f"\n[{rid}] ERROR: {errors[rid]}")
        else:
            text = "".join(buffers[rid]).strip()
            print(f"\n[{rid}] USER: {user_input}\nREPLY: {text}")

if __name__ == "__main__":
    asyncio.run(main())