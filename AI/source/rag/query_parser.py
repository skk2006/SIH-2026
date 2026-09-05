import json
import logging
from .groq_client import get_groq_client

# Supported intents:
# LIST_WATCHLIST
# SEARCH_PERSON_DETECTIONS
# LATEST_PERSON_DETECTION
# COUNT_PERSON_DETECTIONS
# RECENT_DETECTIONS
# DETECTIONS_BY_SOURCE
# DETECTIONS_BY_TIME_RANGE
# WATCHLIST_DETECTED_TODAY
# LATEST_DETECTION
# UNSUPPORTED

SYSTEM_PROMPT = """You are a surveillance database query parser.
Your ONLY task is to convert the operator's question into one supported structured intent.

Supported intents:
- LIST_WATCHLIST (returns all enrolled redlist/watchlist suspects)
- SEARCH_PERSON_DETECTIONS (search detection history for specific name)
- LATEST_PERSON_DETECTION (last time a specific person was seen)
- COUNT_PERSON_DETECTIONS (number of times a person was seen)
- RECENT_DETECTIONS (general recent camera detections)
- DETECTIONS_BY_SOURCE (filter by camera/video name)
- DETECTIONS_BY_TIME_RANGE (filter between times)
- WATCHLIST_DETECTED_TODAY (list everyone caught today)
- LATEST_DETECTION (the absolute last record)
- UNSUPPORTED (fallback)

You must NOT answer the question.
You must NOT generate SQL.

You must NOT invent names, detections, cameras, times,
or surveillance records.

Available information:
- watchlist/redlist people
- stored face detection events

Unsupported:
- vehicle search
- ANPR
- number plates
- weapon retrieval
- fight retrieval
- intrusion
- general chatbot questions

Return valid JSON only.

Required JSON structure:
{
  "intent": "...",
  "person_name": "..." or null,
  "date_reference": "today" | "yesterday" | null,
  "start_time": "..." or null,
  "end_time": "..." or null,
  "source": "..." or null,
  "limit": 10
}
All fields must always exist. If intent is unknown/unsupported, use "UNSUPPORTED".
Limit must be an integer between 1 and 50.
"""

def parse_query(question: str) -> dict:
    client = get_groq_client()
    if not client:
        # Fallback or indicate error
        raise RuntimeError("Groq client is not configured")

    try:
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        
        # Validate structure
        intent = parsed.get("intent", "UNSUPPORTED")
        valid_intents = {
            "LIST_WATCHLIST", "SEARCH_PERSON_DETECTIONS", "LATEST_PERSON_DETECTION",
            "COUNT_PERSON_DETECTIONS", "RECENT_DETECTIONS", "DETECTIONS_BY_SOURCE",
            "DETECTIONS_BY_TIME_RANGE", "WATCHLIST_DETECTED_TODAY", "LATEST_DETECTION",
            "UNSUPPORTED"
        }
        if intent not in valid_intents:
            intent = "UNSUPPORTED"
            
        limit = parsed.get("limit", 10)
        if not isinstance(limit, int):
            limit = 10
        limit = max(1, min(50, limit))

        return {
            "intent": intent,
            "person_name": parsed.get("person_name"),
            "date_reference": parsed.get("date_reference"),
            "start_time": parsed.get("start_time"),
            "end_time": parsed.get("end_time"),
            "source": parsed.get("source"),
            "limit": limit
        }
    except Exception as e:
        logging.error(f"Error parsing query: {e}")
        return fall_back_parsing(question)

def fall_back_parsing(question: str) -> dict:
    q = question.lower()
    intent = "UNSUPPORTED"
    person_name = null = None
    limit = 10
    
    if "redlist" in q or "watchlist" in q:
        intent = "LIST_WATCHLIST"
        limit = 50
    elif "recent detections" in q:
        intent = "RECENT_DETECTIONS"
    
    return {
        "intent": intent,
        "person_name": person_name,
        "date_reference": None,
        "start_time": None,
        "end_time": None,
        "source": None,
        "limit": limit
    }
