import logging
import json
from .query_parser import parse_query
from . import retrieval
from .groq_client import get_groq_client

ANSWER_PROMPT = """You are a surveillance retrieval assistant.

Answer ONLY using the provided PostgreSQL records.

Do not use outside knowledge.
Do not invent detections.
Do not invent locations.
Do not invent confidence values.
Do not invent names.

If there are no matching records, clearly say:
"No matching stored detection events were found."

Do not claim physical absence merely because no stored detection exists.

Be concise and factual.

Question: {question}

Records:
{records}

"""

def process_query(question: str) -> dict:
    # 1. Parse Query
    intent_data = parse_query(question)
    intent = intent_data["intent"]
    
    if intent == "UNSUPPORTED":
        return {
            "success": True,
            "question": question,
            "intent": intent,
            "answer": "This surveillance query is currently unsupported. This version searches only watchlist people and stored face detection events.",
            "result_count": 0,
            "results": []
        }

    # 2. Retrieve Data
    results = []
    if intent == "LIST_WATCHLIST":
        results = retrieval.list_watchlist(intent_data["limit"])
    elif intent == "SEARCH_PERSON_DETECTIONS":
        results = retrieval.search_person_detections(
            intent_data["person_name"], 
            intent_data["date_reference"], 
            intent_data["start_time"], 
            intent_data["end_time"], 
            intent_data["source"], 
            intent_data["limit"]
        )
    elif intent == "LATEST_PERSON_DETECTION":
        res = retrieval.get_latest_person_detection(intent_data["person_name"])
        if res:
            results = [res]
    elif intent == "COUNT_PERSON_DETECTIONS":
        count = retrieval.count_person_detections(
            intent_data["person_name"], 
            intent_data["date_reference"]
        )
        return {
            "success": True,
            "question": question,
            "intent": intent,
            "answer": f"Stored detection events found: {count}",
            "result_count": count,
            "results": []
        }
    elif intent == "RECENT_DETECTIONS":
        results = retrieval.get_recent_detections(intent_data["limit"])
    elif intent == "DETECTIONS_BY_SOURCE":
        results = retrieval.get_detections_by_source(intent_data["source"], intent_data["limit"])
    elif intent == "DETECTIONS_BY_TIME_RANGE":
        results = retrieval.get_detections_by_time_range(
            intent_data["date_reference"],
            intent_data["start_time"],
            intent_data["end_time"],
            intent_data["limit"]
        )
    elif intent == "WATCHLIST_DETECTED_TODAY":
        results = retrieval.get_watchlist_detected_today(intent_data["limit"])
    elif intent == "LATEST_DETECTION":
        res = retrieval.get_latest_detection()
        if res:
            results = [res]

    # Clean records for Groq (remove image URLs)
    clean_records = []
    for r in results:
        cr = dict(r)
        cr.pop("reference_image_url", None)
        cr.pop("evidence_image_url", None)
        clean_records.append(cr)

    # 3. Generate Answer
    answer = "No matching stored records were found."
    if results:
        client = get_groq_client()
        if client:
            try:
                system_content = ANSWER_PROMPT.format(
                    question=question,
                    records=json.dumps(clean_records, indent=2)
                )
                response = client.chat.completions.create(
                    model="qwen/qwen3.8-27b",
                    messages=[
                        {"role": "system", "content": "You are a helpful surveillance assistant."},
                        {"role": "user", "content": system_content}
                    ],
                    temperature=0
                )
                answer = response.choices[0].message.content
            except Exception as e:
                logging.error(f"Error generating answer: {e}")
                
                # Basic fallback text
                if intent == "LIST_WATCHLIST":
                    answer = f"There are {len(results)} people currently enrolled in the watchlist."
                else:
                    answer = f"Found {len(results)} matching records."
    else:
        if intent == "SEARCH_PERSON_DETECTIONS" or intent == "LATEST_PERSON_DETECTION":
            if intent_data["person_name"]:
                answer = f"No matching stored detection events were found for {intent_data['person_name']}."
        
    return {
        "success": True,
        "question": question,
        "intent": intent,
        "answer": answer,
        "result_count": len(results),
        "results": results
    }
