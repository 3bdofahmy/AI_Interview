import csv
import random
from models import ResponseSignal
from .BaseController import BaseController




class QuestionsController(BaseController):
    def __init__(self):
        super().__init__()
        self.questions = []

    def read_data(self, file_path="./dataset/technical.csv"):
        try:
            with open(file_path, mode='r', encoding='utf-8') as file:  
                csv_reader = csv.DictReader(file)
                self.questions = [row for row in csv_reader]
            return ResponseSignal.FILE_FOUND.value
      
        except FileNotFoundError:
            return ResponseSignal.FILE_NOT_FOUND.value
       
        except UnicodeDecodeError:
            return ResponseSignal.UnicodeDecodeError.value


    def get_questions(self, track: str, difficulty: str  ):
        
        filtered_questions = [q for q in self.questions if q['track'] == track and q['difficulty'] == difficulty]
        
        selected_questions = random.sample(filtered_questions, min(3, len(filtered_questions)))

        return  [{'id': q['id'], 'question': q['question'], 'True_answer': q['answer']} for i, q in enumerate(selected_questions)]


