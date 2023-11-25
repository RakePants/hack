from fastapi import FastAPI
import uvicorn
from starlette.staticfiles import StaticFiles
from app.routers.app import app_router

app = FastAPI()
app.include_router(app_router)
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
