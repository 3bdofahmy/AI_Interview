from fastapi import APIRouter, UploadFile, Form
import os
import csv

# Assuming you have the base controller for general functionalities
from .BaseController import BaseController

data_router = APIRouter()

class UserController(BaseController):
    
    def __init__(self):
        super().__init__()
    
    def get_project_path(self, username: str):
        user_dir = os.path.join(self.files_dir, username)

        if not os.path.exists(user_dir):
            os.makedirs(user_dir)

        videos_dir = os.path.join(user_dir, "videos")
        audio_dir = os.path.join(user_dir, "audio")
        
        os.makedirs(videos_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)

        return user_dir
    
    def get_videos_path(self, username: str):
        user_dir = os.path.join(self.files_dir, username)
        videos_dir = os.path.join(user_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        return videos_dir
    
    def get_audio_path(self, username: str):
        user_dir = os.path.join(self.files_dir, username)
        audio_dir = os.path.join(user_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        return audio_dir
    
    def get_user_csv_path(self, username: str):
        user_dir = os.path.join(self.files_dir, username)
        return os.path.join(user_dir, f"{username}.csv")

    def create_user_csv(self, username: str, user_dir: str):
        csv_file = os.path.join(user_dir, f"{username}.csv")
        
        if not os.path.exists(csv_file):
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["id", "Question", "True_answer", "user_answer", "nlp_score", "cv_score"])

        return csv_file

    
