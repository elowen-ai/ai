import socketio
from ai import AI
from auth import Auth
from flask import Flask

ai = AI()

sio = socketio.Server(async_mode='threading')
app = Flask(__name__)
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)

auth = Auth()

@sio.event
def connect(sid, data, authData):
    response = auth.authenticate(sid, authData)

    if response.get("status") is False:
        return response

    print("Client connected:", sid)
    print("Successfully authenticated")

    return response

@sio.event
@auth.isAuthenticated
def stop(sid):
    AI.stop(sid)

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=6000)