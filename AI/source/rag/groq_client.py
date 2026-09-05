import os
from groq import Groq

_client = None

def get_groq_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            _client = Groq(api_key=api_key)
    return _client
