import os
import cv2
import csv
import shutil
import numpy as np
import pandas as pd
import torch
import pickle
import whisper
import moviepy.editor as mp_editor
from sentence_transformers import SentenceTransformer
import mediapipe as mp
from ..BaseController import BaseController
from .process_utils.landmarks import landmark
from fastapi import UploadFile
import time
import requests



class FileController(BaseController):
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
    headers = {"Authorization": "Bearer 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"}
    
    def __init__(self, model_file, sentence_model='all-MiniLM-L6-v2'):
        super().__init__()
        API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
        headers = {"Authorization": "Bearer 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"}
        self.model = self.load_model(model_file)
        self.api_url = API_URL
        self.headers = headers
        self.whisper_model = whisper.load_model("base")
        self.sentence_model = SentenceTransformer(sentence_model)
        self.mp_holistic = mp.solutions.holistic
        self.landmark_columns = landmark
        self.frame_counter = 0
        self.fps = 1

    def load_model(self, model_file):
        with open(model_file, 'rb') as f:
            return pickle.load(f)

    def ensure_directory_exists(self, path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_videos_path(self, username):
        user_dir = os.path.join(self.files_dir, username)
        videos_dir = os.path.join(user_dir, "videos")
        self.ensure_directory_exists(videos_dir)
        return videos_dir

    def get_audio_path(self, username):
        user_dir = os.path.join(self.files_dir, username)
        audio_dir = os.path.join(user_dir, "audio")
        self.ensure_directory_exists(audio_dir)
        return audio_dir

    def get_user_csv_path(self, username):
        user_dir = os.path.join(self.files_dir, username)
        return os.path.join(user_dir, f"{username}.csv")

    def extract_landmarks(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            results = holistic.process(image)
        
        right_row = []
        face_row = []
        left_row = []
        pose_row = []

        if results.right_hand_landmarks:
            right_row = list(np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in results.right_hand_landmarks.landmark]).flatten())
        else:
            right_row = [0] * 84

        if results.face_landmarks:
            face_row = list(np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in results.face_landmarks.landmark]).flatten())
        else:
            face_row = [0] * 1872

        if results.left_hand_landmarks:
            left_row = list(np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in results.left_hand_landmarks.landmark]).flatten())
        else:
            left_row = [0] * 84

        if results.pose_landmarks:
            pose_row = list(np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in results.pose_landmarks.landmark]).flatten())
        else:
            pose_row = [0] * 132

        return right_row + face_row + left_row + pose_row

    def predict_frame(self, frame, video_fps):
        frame_interval = int(video_fps) 

        current_time_s = self.frame_counter / video_fps

        print(f"Frame counter: {self.frame_counter}, Video FPS: {video_fps}, Current Time (s): {current_time_s}")

        if self.frame_counter % frame_interval == 0:
            row = self.extract_landmarks(frame)  #
            X = pd.DataFrame([row], columns=self.landmark_columns)
            
            body_language_class = self.model.predict(X)[0]
            body_language_prob = self.model.predict_proba(X)[0]

           
            print(f"Processing second: {int(current_time_s)} | Frame: {self.frame_counter}")

            return body_language_class, body_language_prob

        return None, None

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.frame_counter = 0  
        frame_results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_interval = 15  
            if self.frame_counter % frame_interval == 0:
                body_language_class, body_language_prob = self.predict_frame(frame, video_fps)
                if body_language_class is not None:
                    frame_results.append({
                        'frame': self.frame_counter,
                        'time': self.frame_counter / video_fps,
                        'class': body_language_class,
                        'probability': body_language_prob.tolist()
                    })

            self.frame_counter += 1  

        cap.release()
        cv2.destroyAllWindows()

        return self.calculate_score(frame_results)

    def calculate_score(self, frame_results):
        if not frame_results:
            return 0
        
        good_count = sum(1 for result in frame_results if result["class"] == "good")
        bad_count = sum(1 for result in frame_results if result["class"] == "bad")
        total = good_count + bad_count
        
        return (good_count / total) * 100 if total > 0 else 0

    def extract_audio(self, video_path, output_path):
        print(f"Extracting audio from: {video_path} to {output_path}")
        try:
            self.ensure_directory_exists(output_path)
            video = mp_editor.VideoFileClip(video_path)
            audio = video.audio
            if not output_path.endswith(".wav"):
                output_path = output_path.replace(".mp3", ".wav")
            audio.write_audiofile(output_path, codec='pcm_s16le') 
            print(f"Successfully extracted audio to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error extracting audio: {e}")
            raise

    def query(self, filename):
        with open(filename, "rb") as f:
            data = f.read()
        response = requests.post(self.API_URL, headers=self.headers, data=data)
        
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code}, {response.text}")

        return response.json()

    def transcribe_audio(self, audio_path):
        print(f"Transcribing audio from: {audio_path}")
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            output = self.query(audio_path)
            transcribed_text = output.get("text", "")
            print("Transcription successful")
            return transcribed_text
        except Exception as e:
            print(f"Error during transcription: {e}")
            raise

    def calculate_similarity(self, text1, text2):
        embedding1 = self.sentence_model.encode(text1, convert_to_tensor=True)
        embedding2 = self.sentence_model.encode(text2, convert_to_tensor=True)
        similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)

        scaled_score = (similarity.item() + 1) / 2  
        
        return scaled_score


    def get_true_answer(self, csv_file, question_id):
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['id'] == question_id:
                    return row['True_answer']
        return None

    def update_csv(self, csv_file, question_id, cv_score, transcribed_text, nlp_score):
        temp_file = csv_file + '.tmp'
        with open(csv_file, 'r') as file, open(temp_file, 'w', newline='') as temp:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames
            writer = csv.DictWriter(temp, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                if row['id'] == question_id:
                    row['cv_score'] = cv_score
                    row['user_answer'] = transcribed_text
                    row['nlp_score'] = nlp_score
                writer.writerow(row)
        
        os.replace(temp_file, csv_file)

    def process_audio(self, username, video_path, csv_file, question_id):
        audio_dir = self.get_audio_path(username)
        audio_file = os.path.join(audio_dir, f"{question_id}.wav")
        
        self.extract_audio(video_path, audio_file)
        transcribed_text = self.transcribe_audio(audio_file)
        true_answer = self.get_true_answer(csv_file, question_id)
        if not true_answer:
            raise ValueError(f"True answer for question ID {question_id} not found in CSV.")
        
        nlp_score = self.calculate_similarity(true_answer, transcribed_text)
        return transcribed_text, nlp_score



    async def process_uploaded_video(self, username, question_id, video: UploadFile):
        videos_dir = self.get_videos_path(username)
        os.makedirs(videos_dir, exist_ok=True)
        
        video_path = os.path.join(videos_dir, f"{question_id}.mp4")
        
        with open(video_path, "wb") as buffer:
            video_data = await video.read()  
            buffer.write(video_data)  
        
        cv_score = self.process_video(video_path)
        
        transcribed_text, nlp_score = self.process_audio(username, video_path, csv_file, question_id)
        
        csv_file = self.get_user_csv_path(username)
        self.update_csv(csv_file, question_id, cv_score, transcribed_text, nlp_score)
        
        return {"message": "Video processed and scores updated successfully"}
    
    def process_csv(self , username):
        questions = []
        nlp_scores = []
        cv_scores = []

        file_path = self.get_user_csv_path(username)
        
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                questions.append({
                    "id": row['id'],
                    "question": row['Question'],
                    "true_answer": row['True_answer'],
                    "user_answer": row['user_answer']
                })
                
                nlp_scores.append(float(row['nlp_score']))
                cv_scores.append(float(row['cv_score']))
        
        avg_nlp_score = sum(nlp_scores) / len(nlp_scores) * 100
        avg_cv_score = sum(cv_scores) / len(cv_scores) * 100
        
        total_score = (avg_nlp_score + avg_cv_score) / 2
        
        result = {
            "Questions": questions,
            "NLP Score": f"{avg_nlp_score:.2f}%",
            "CV Score": f"{avg_cv_score:.2f}%",
            "Total Score": f"{total_score:.2f}%"
        }
        
        return result

