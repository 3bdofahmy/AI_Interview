from fastapi import BackgroundTasks, FastAPI, APIRouter , UploadFile,File ,HTTPException
from fastapi.responses import JSONResponse
from helpers.config import get_settings, Settings
from controllers import QuestionsController, UserController 
from controllers.BaseController import BaseController 
from routes.schema.Question import QuestionsRequest
from models import ResponseSignal
import csv
import logging
from controllers.ProcessController.video_processor import FileController
import shutil
import os

logger = logging.getLogger('uvicorn.error')

question_router = APIRouter(
    prefix="/api/v2/data",
    tags=["api_v2", "data"],
)

@question_router.post("/get_question/{username}")
async def questions_endpoint(username: str, questions_request: QuestionsRequest):

    user_controller = UserController()
    questions_controller = QuestionsController()
    baseController = BaseController()
    data = baseController.data_dir

    user_dir = user_controller.get_project_path(username)
    csv_file = user_controller.create_user_csv(username, user_dir)

    questions_controller.read_data(data)

    selected_questions = questions_controller.get_questions(
        track = questions_request.track, 
        difficulty = questions_request.difficulty
    )

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        for value in selected_questions:
            question_id = value['id']
            question_text = value['question']
            True_answer = value['True_answer']
            writer.writerow([question_id, question_text,True_answer, "", "" , ""])

    response_data = [{'id': value['id'], 'question': value['question']} for value in selected_questions]

    return JSONResponse(
            content={
                "Questions" : response_data ,
                "signal": ResponseSignal.SUCCESS.value
            }
        )

@question_router.post("/process_video/{username}/{question_id}")
async def process_video_endpoint(
    username: str, 
    question_id: str, 
    background_tasks: BackgroundTasks, 
    video: UploadFile = File(...)
):
    video_content = await video.read()

    file_controller = FileController(model_file='detect.pkl')

    background_tasks.add_task(background_process_video, file_controller, username, question_id, video_content)

    return JSONResponse(content={"message": "Video uploaded successfully. Processing started in the background."})


async def background_process_video(file_controller, username, question_id, video_content: bytes):
    videos_dir = file_controller.get_videos_path(username)
    os.makedirs(videos_dir, exist_ok=True)
    video_path = os.path.join(videos_dir, f"{question_id}.mp4")

    with open(video_path, "wb") as buffer:
        buffer.write(video_content)

    cv_score = file_controller.process_video(video_path)

    csv_file = file_controller.get_user_csv_path(username)
    transcribed_text, nlp_score = file_controller.process_audio(username, video_path, csv_file, question_id)

    file_controller.update_csv(csv_file, question_id, cv_score, transcribed_text, nlp_score)

    return {"message": "Video processed and scores updated successfully"}

file_controller = FileController(model_file='detect.pkl')

@question_router.get("/process_csv/{username}")
async def process_csv_endpoint(username: str):
    file_controller = FileController(model_file='detect.pkl')

    try:
        result = file_controller.process_csv(username)
        return result  
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="User CSV file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
