from fastapi import APIRouter,HTTPException
from pydantic import BaseModel
from nirf_chatbot.chat_service import process_question
router = APIRouter(prefix="/api", tags=["Chatbot"])
class ChatRequest(BaseModel):
    question:str
    session_id:str|None=None
@router.post("/chat")
def chat(request:ChatRequest):
    try:
        response=process_question(request.question)
        return {
            "success": True,
            "question": request.question,
            "answer": response["answer"],
            "generated_sql": response["generated_sql"],
            "results": response["results"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )