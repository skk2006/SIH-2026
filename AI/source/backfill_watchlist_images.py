import os
import cv2
import psycopg2
import config

DATABASE_URL = config.DATABASE_URL

def backfill():
    if not DATABASE_URL:
        print("DATABASE_URL not set.")
        return
        
    conn = psycopg2.connect(DATABASE_URL)
    with conn:
        with conn.cursor() as cur:
            # Find rows where reference_image IS NULL but we have a path
            cur.execute("""
                SELECT id, name, reference_image_path
                FROM watchlist_people
                WHERE reference_image IS NULL AND reference_image_path IS NOT NULL
            """)
            rows = cur.fetchall()
            
            updated_count = 0
            for row in rows:
                pid, name, path = row
                if os.path.exists(path):
                    img = cv2.imread(path)
                    if img is not None:
                        ok, enc = cv2.imencode('.jpg', img)
                        if ok:
                            img_bytes = enc.tobytes()
                            cur.execute("""
                                UPDATE watchlist_people
                                SET reference_image = %s
                                WHERE id = %s
                            """, (psycopg2.Binary(img_bytes), pid))
                            updated_count += 1
                            print(f"Updated {name} (ID: {pid}) with {len(img_bytes)} bytes.")
                    else:
                        print(f"Failed to read image for {name} at {path}")
                else:
                    print(f"File not found for {name} at {path}")
            
    conn.close()
    print(f"Backfill complete. Updated {updated_count} records.")

if __name__ == "__main__":
    backfill()
