import pickle
import psycopg2
from datetime import datetime, timezone
import config

DATABASE_URL = config.DATABASE_URL


def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is not set.")
    return psycopg2.connect(DATABASE_URL)


def init_tables():
    ddl = """
    CREATE TABLE IF NOT EXISTS watchlist_people (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL DEFAULT 'REDLIST',
        face_embedding BYTEA,
        reference_image_path TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    
    ALTER TABLE watchlist_people ADD COLUMN IF NOT EXISTS reference_image BYTEA;

    CREATE TABLE IF NOT EXISTS detection_events (
        id SERIAL PRIMARY KEY,
        person_id INTEGER REFERENCES watchlist_people(id) ON DELETE SET NULL,
        person_name TEXT NOT NULL,
        source TEXT NOT NULL,
        confidence DOUBLE PRECISION,
        frame_image BYTEA,
        detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """

    try:
        conn = get_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
        conn.close()
        print("[PG] Tables created / verified.")
        return True
    except Exception as e:
        print(f"[PG] init_tables failed: {e}")
        return False


def insert_watchlist(name, category, embedding, reference_image_path=None, reference_image=None):
    try:
        emb_bytes = pickle.dumps(embedding) if embedding is not None else None

        conn = get_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO watchlist_people
                    (name, category, face_embedding, reference_image_path, reference_image, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        name,
                        category,
                        psycopg2.Binary(emb_bytes) if emb_bytes else None,
                        reference_image_path,
                        psycopg2.Binary(reference_image) if reference_image else None,
                        datetime.now(timezone.utc)
                    )
                )
                row = cur.fetchone()

        conn.close()
        return row[0] if row else None

    except Exception as e:
        print(f"[PG] insert_watchlist failed: {e}")
        return None


def get_watchlist_id(name):
    try:
        conn = get_conn()

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id
                FROM watchlist_people
                WHERE name=%s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (name,)
            )

            row = cur.fetchone()

        conn.close()
        return row[0] if row else None

    except Exception as e:
        print(f"[PG] get_watchlist_id failed: {e}")
        return None


def get_watchlist_image(person_id):
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT reference_image, reference_image_path
                FROM watchlist_people
                WHERE id=%s
                """,
                (person_id,)
            )
            row = cur.fetchone()
        conn.close()
        if row:
            if row[0]: # reference_image BYTEA
                return bytes(row[0]), None 
            else: # fallback to reference_image_path
                return None, row[1]
        return None, None
    except Exception as e:
        print(f"[PG] get_watchlist_image failed: {e}")
        return None, None

def insert_detection(
    person_name,
    source,
    confidence,
    frame_jpeg_bytes,
    person_id=None
):
    if person_id is None:
        person_id = get_watchlist_id(person_name)

    try:
        conn = get_conn()

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO detection_events
                    (
                        person_id,
                        person_name,
                        source,
                        confidence,
                        frame_image,
                        detected_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        person_id,
                        person_name,
                        source,
                        confidence if confidence is not None else 0.0,
                        psycopg2.Binary(frame_jpeg_bytes)
                        if frame_jpeg_bytes else None,
                        datetime.now(timezone.utc)
                    )
                )

                row = cur.fetchone()

        conn.close()
        return row[0] if row else None

    except Exception as e:
        print(f"[PG] insert_detection failed: {e}")
        return None


def get_frame_image(event_id):
    try:
        conn = get_conn()

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT frame_image
                FROM detection_events
                WHERE id=%s
                """,
                (event_id,)
            )

            row = cur.fetchone()

        conn.close()

        if row and row[0]:
            return bytes(row[0])

        return None

    except Exception as e:
        print(f"[PG] get_frame_image failed: {e}")
        return None


def get_recent_detections(limit=100):
    try:
        conn = get_conn()

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    person_id,
                    person_name,
                    source,
                    confidence,
                    detected_at,
                    octet_length(frame_image)
                FROM detection_events
                ORDER BY detected_at DESC
                LIMIT %s
                """,
                (limit,)
            )

            rows = cur.fetchall()

        conn.close()

        return [
            {
                "id": r[0],
                "person_id": r[1],
                "person_name": r[2],
                "source": r[3],
                "confidence": r[4],
                "detected_at": r[5].isoformat() if r[5] else None,
                "image_size": r[6],
                "frame_url": f"/api/events/{r[0]}/frame"
            }
            for r in rows
        ]

    except Exception as e:
        print(f"[PG] get_recent_detections failed: {e}")
        return []