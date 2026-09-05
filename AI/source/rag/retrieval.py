import db_pg
from datetime import datetime, timezone, timedelta

def get_db_conn():
    return db_pg.get_conn()

def _dt_to_iso(dt):
    return dt.isoformat() if dt else None

def list_watchlist(limit=50):
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, category, created_at 
                FROM watchlist_people 
                ORDER BY created_at DESC 
                LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
        conn.close()
        return [
            {
                "person_id": r[0],
                "person_name": r[1],
                "category": r[2],
                "created_at": _dt_to_iso(r[3]),
                "reference_image_url": f"/api/watchlist/{r[0]}/image"
            } for r in rows
        ]
    except Exception as e:
        print(f"[RAG] list_watchlist failed: {e}")
        return []

def search_person_detections(person_name, date_reference=None, start_time=None, end_time=None, source=None, limit=10):
    if not person_name:
        return []
    
    combined_results = []
    
    # Check if they exist in the Watchlist Database first
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, category, created_at FROM watchlist_people WHERE name ILIKE %s", (f"%{person_name}%",))
            w_rows = cur.fetchall()
            for r in w_rows:
                combined_results.append({
                    "person_id": r[0],
                    "person_name": r[1],
                    "category": r[2],
                    "created_at": _dt_to_iso(r[3]),
                    "reference_image_url": f"/api/watchlist/{r[0]}/image",
                    "evidence_image_url": None,
                    "source": "WATCHLIST_ENROLLMENT"
                })
        conn.close()
    except Exception as e:
        print(f"[RAG] watchlist search failed: {e}")

    # Now search for actual detection events
    query = """
        SELECT id, person_id, person_name, source, confidence, detected_at
        FROM detection_events
        WHERE person_name ILIKE %s
    """
    params = [f"%{person_name}%"]
    
    # Date filtering
    if date_reference:
        now = datetime.now(timezone.utc)
        if date_reference.lower() == "today":
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            query += " AND detected_at >= %s"
            params.append(start_of_day)
        elif date_reference.lower() == "yesterday":
            start_of_yesterday = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_yesterday = now.replace(hour=0, minute=0, second=0, microsecond=0)
            query += " AND detected_at >= %s AND detected_at < %s"
            params.extend([start_of_yesterday, end_of_yesterday])
            
    if source:
        query += " AND source = %s"
        params.append(source)
        
    query += " ORDER BY detected_at DESC LIMIT %s"
    params.append(limit)
    
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
        conn.close()
        for r in rows:
            combined_results.append({
                "event_id": r[0],
                "person_id": r[1],
                "person_name": r[2],
                "source": r[3],
                "confidence": r[4],
                "detected_at": _dt_to_iso(r[5]),
                "reference_image_url": f"/api/watchlist/{r[1]}/image" if r[1] else None,
                "evidence_image_url": f"/api/events/{r[0]}/frame"
            })
    except Exception as e:
        print(f"[RAG] search_person_detections failed: {e}")
        
    return combined_results

def get_latest_person_detection(person_name):
    # Same as search but limit 1
    res = search_person_detections(person_name, limit=1)
    if res:
        return res[0]
    return None

def count_person_detections(person_name, date_reference=None):
    if not person_name:
        return 0
    query = """
        SELECT COUNT(*)
        FROM detection_events
        WHERE person_name ILIKE %s
    """
    params = [f"%{person_name}%"]
    if date_reference:
        now = datetime.now(timezone.utc)
        if date_reference.lower() == "today":
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            query += " AND detected_at >= %s"
            params.append(start_of_day)
        elif date_reference.lower() == "yesterday":
            start_of_yesterday = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_yesterday = now.replace(hour=0, minute=0, second=0, microsecond=0)
            query += " AND detected_at >= %s AND detected_at < %s"
            params.extend([start_of_yesterday, end_of_yesterday])

    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute(query, tuple(params))
            row = cur.fetchone()
        conn.close()
        return row[0] if row else 0
    except Exception as e:
        print(f"[RAG] count_person_detections failed: {e}")
        return 0

def get_recent_detections(limit=10):
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, person_id, person_name, source, confidence, detected_at
                FROM detection_events
                ORDER BY detected_at DESC
                LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
        conn.close()
        return [
            {
                "event_id": r[0],
                "person_id": r[1],
                "person_name": r[2],
                "source": r[3],
                "confidence": r[4],
                "detected_at": _dt_to_iso(r[5]),
                "reference_image_url": f"/api/watchlist/{r[1]}/image" if r[1] else None,
                "evidence_image_url": f"/api/events/{r[0]}/frame"
            } for r in rows
        ]
    except Exception as e:
        print(f"[RAG] get_recent_detections failed: {e}")
        return []

def get_detections_by_source(source, limit=10):
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, person_id, person_name, source, confidence, detected_at
                FROM detection_events
                WHERE source = %s
                ORDER BY detected_at DESC
                LIMIT %s
            """, (source, limit))
            rows = cur.fetchall()
        conn.close()
        return [
            {
                "event_id": r[0],
                "person_id": r[1],
                "person_name": r[2],
                "source": r[3],
                "confidence": r[4],
                "detected_at": _dt_to_iso(r[5]),
                "reference_image_url": f"/api/watchlist/{r[1]}/image" if r[1] else None,
                "evidence_image_url": f"/api/events/{r[0]}/frame"
            } for r in rows
        ]
    except Exception as e:
        print(f"[RAG] get_detections_by_source failed: {e}")
        return []

def get_detections_by_time_range(date_reference, start_time_str, end_time_str, limit=50):
    # Retrieve local time assuming +05:30 IST based on app's configuration
    local_tz = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(local_tz)
    target_date = now.date()
    
    if date_reference and date_reference.lower() == "yesterday":
        target_date = (now - timedelta(days=1)).date()
        
    try:
        start_t = datetime.strptime(start_time_str, "%H:%M").time() if start_time_str else None
        end_t = datetime.strptime(end_time_str, "%H:%M").time() if end_time_str else None
    except:
        start_t = None
        end_t = None

    if start_t and end_t:
        dt_start = datetime.combine(target_date, start_t).replace(tzinfo=local_tz)
        dt_end = datetime.combine(target_date, end_t).replace(tzinfo=local_tz)
    else:
        dt_start = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=local_tz)
        dt_end = dt_start + timedelta(days=1)
        
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, person_id, person_name, source, confidence, detected_at
                FROM detection_events
                WHERE detected_at >= %s AND detected_at <= %s
                ORDER BY detected_at DESC
                LIMIT %s
            """, (dt_start, dt_end, limit))
            rows = cur.fetchall()
        conn.close()
        return [
            {
                "event_id": r[0],
                "person_id": r[1],
                "person_name": r[2],
                "source": r[3],
                "confidence": r[4],
                "detected_at": _dt_to_iso(r[5]),
                "reference_image_url": f"/api/watchlist/{r[1]}/image" if r[1] else None,
                "evidence_image_url": f"/api/events/{r[0]}/frame"
            } for r in rows
        ]
    except Exception as e:
        print(f"[RAG] get_detections_by_time_range failed: {e}")
        return []

def get_watchlist_detected_today(limit=50):
    now = datetime.now(timezone.utc)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, person_id, person_name, source, confidence, detected_at
                FROM detection_events
                WHERE detected_at >= %s
                ORDER BY detected_at DESC
                LIMIT %s
            """, (start_of_day, limit))
            rows = cur.fetchall()
        conn.close()
        return [
            {
                "event_id": r[0],
                "person_id": r[1],
                "person_name": r[2],
                "source": r[3],
                "confidence": r[4],
                "detected_at": _dt_to_iso(r[5]),
                "reference_image_url": f"/api/watchlist/{r[1]}/image" if r[1] else None,
                "evidence_image_url": f"/api/events/{r[0]}/frame"
            } for r in rows
        ]
    except Exception as e:
        print(f"[RAG] get_watchlist_detected_today failed: {e}")
        return []

def get_latest_detection():
    res = get_recent_detections(limit=1)
    if res:
        return res[0]
    return None
