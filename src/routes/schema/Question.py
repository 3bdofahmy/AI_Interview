from pydantic import BaseModel

class QuestionsRequest(BaseModel):
    
    track: str
    difficulty : str 