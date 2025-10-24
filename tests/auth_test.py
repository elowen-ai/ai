import os
import asyncio
import threading
import socketio
from flask import Flask
from dotenv import load_dotenv

load_dotenv()

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from auth import Auth
auth = Auth()

shutdown_event = asyncio.Event()

async def server():
    sio = socketio.Server(async_mode='threading')
    app = Flask(__name__)
    app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)
    print("Server started")

    @sio.event
    def connect(sid, environ, authData):
        print(f"[SIO] TRIES TO CONNECT: {sid}", flush=True)
        response = auth.authenticate(sid, authData)
        print(f"[SIO] AUTHENTICATED: {response}", flush=True)

        if response.get("status") is False:
            raise ConnectionRefusedError("Unauthorized")

        print("Client connected:", sid)
        print("Successfully authenticated")

        return response

    @sio.on("ping")
    @auth.isAuthenticated
    def ping(sid, data):
        sio.emit("pong", { "message": "pong", "data": data }, to=sid)
        print(f"[SIO] SENT PONG: {sid}", flush=True)

    def run_flask():
        app.run(host="127.0.0.1", port=5005, use_reloader=False)

    threading.Thread(target=run_flask, daemon=True).start()
    await asyncio.sleep(0.5)
    await shutdown_event.wait()

async def client():
    print("Client started")
    sio = socketio.AsyncClient()
    got_pong = asyncio.Event()

    @sio.event
    async def connect():
        print("[client] connected:", sio.sid)

    print("Sending ping")
    @sio.event
    async def pong(msg: dict[str, str]):
        print(f"[client] received pong: {msg}", flush=True)
        got_pong.set()

    await sio.connect("http://localhost:5005", auth={"token": os.getenv("TEST_JWT_TOKEN")})
    await sio.emit("ping", { "extraMessage": "Ping!!" })
    await got_pong.wait()

async def main():
    server_task = asyncio.create_task(server())
    client_task = asyncio.create_task(client())
    await asyncio.gather(server_task, client_task)

if __name__ == "__main__":
    asyncio.run(main())