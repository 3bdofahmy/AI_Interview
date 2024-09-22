from fastapi import FastAPI
from routes import base, question 
from helpers.config import get_settings
from pyngrok import ngrok



app = FastAPI()
settings = get_settings()
app.include_router(base.base_router)
app.include_router(question.question_router)



NGROK_AUTH = settings.NGROK_AUTH
port = settings.NGROK_port
ngrok.set_auth_token(NGROK_AUTH)
tunnel = ngrok.connect(port ,domain="closing-insect-tough.ngrok-free.app")
print("puplic Ip :" ,tunnel.public_url)