import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text

from app.db import Base, SessionLocal, engine
from app import models
from app.ingestion.pipeline import ingest_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the FastAPI docs into Postgres")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)

    started = time.perf_counter()
    with SessionLocal() as session:
        report = ingest_directory(session, limit=args.limit)
    elapsed = time.perf_counter() - started
    print(
        f"documents={report.documents} chunks={report.chunks} "
        f"skipped={report.skipped} seconds={elapsed:.1f}"
    )


if __name__ == "__main__":
    main()
