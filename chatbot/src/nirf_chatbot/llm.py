from langchain_sarvam import ChatSarvam
from .Settings import gemini


llm = ChatSarvam(
    model="sarvam-105b",
    api_key=gemini(),
    temperature=0,
    max_retries=5,
    # This budget also covers the model's reasoning. At the default of 2048,
    # harder questions can run out before any answer is written.
    max_tokens=8192,
)
