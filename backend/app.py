
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/test")
def test():
    return {
        "message": "Servidor funcionando correctamente"
    }