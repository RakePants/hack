from fastapi import APIRouter, UploadFile
from fastapi.responses import FileResponse
import app.utils.utils as utils

app_router = APIRouter()


@app_router.get("/")
async def main_page() -> FileResponse:
    return FileResponse("templates/index.html")


@app_router.post("/video")
async def video(file: UploadFile):
    result = await utils.main(file)
    return result



