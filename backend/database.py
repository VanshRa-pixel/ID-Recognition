import os
import psycopg #type:ignore
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

conn = psycopg.connect(DATABASE_URL)

def save_verification_session(
    live_image: bytes,
    card_image: bytes,
    similarity: float,
    verified: bool,
    extracted_data: dict
):
    with conn.cursor() as cur:

        cur.execute(
            """
            INSERT INTO verification_sessions
            (
                live_image,
                card_image,
                similarity,
                verified,
                extracted_data
            )
            VALUES (%s,%s,%s,%s,%s)
            """,
            (
                live_image,
                card_image,
                similarity,
                verified,
                psycopg.types.json.Jsonb(extracted_data)
            )
        )

        conn.commit()