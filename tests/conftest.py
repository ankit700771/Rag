import os
from pathlib import Path

import psycopg
import pytest
from dotenv import dotenv_values
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
TEST_DB_NAME = "rag_test"


def _build_test_url() -> str:
    explicit = os.environ.get("TEST_DATABASE_URL")
    if explicit:
        return explicit
    base = dotenv_values(ROOT / ".env")["DATABASE_URL"]
    return make_url(base).set(database=TEST_DB_NAME).render_as_string(hide_password=False)


TEST_URL = _build_test_url()
os.environ["DATABASE_URL"] = TEST_URL


def _ensure_test_database() -> None:
    url = make_url(TEST_URL)
    admin = url.set(database="postgres", drivername="postgresql")
    with psycopg.connect(admin.render_as_string(hide_password=False), autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (url.database,)
        ).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{url.database}"')


_ensure_test_database()


@pytest.fixture(scope="session")
def engine():
    from sqlalchemy import text

    from app import models
    from app.db import Base, engine as app_engine

    with app_engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.drop_all(app_engine)
    Base.metadata.create_all(app_engine)
    yield app_engine
    Base.metadata.drop_all(app_engine)


@pytest.fixture()
def session(engine):
    from sqlalchemy import text

    from app.db import SessionLocal

    with SessionLocal() as s:
        s.execute(text("TRUNCATE documents RESTART IDENTITY CASCADE"))
        s.commit()
        yield s
        s.rollback()


@pytest.fixture()
def docs_tree(tmp_path):
    docs = tmp_path / "docs"
    (docs / "tutorial").mkdir(parents=True)
    (docs / "reference").mkdir()
    (docs / "tutorial" / "background-tasks.md").write_text(
        "# Background Tasks { #background-tasks }\n\n"
        "You can define background tasks to be run after returning a response.\n\n"
        "## Using BackgroundTasks { #using-backgroundtasks }\n\n"
        "Import BackgroundTasks and define a parameter in your path operation function. "
        "FastAPI creates the object and passes it in, then runs add_task functions after "
        "the response has been sent to the client.\n",
        encoding="utf-8",
    )
    (docs / "tutorial" / "cors.md").write_text(
        "# CORS (Cross-Origin Resource Sharing) { #cors }\n\n"
        "CORS refers to situations when a frontend running in a browser has JavaScript code "
        "that communicates with a backend in a different origin. Use CORSMiddleware with "
        "allow_origins to permit requests from other domains.\n",
        encoding="utf-8",
    )
    (docs / "tutorial" / "static-files.md").write_text(
        "# Static Files { #static-files }\n\n"
        "You can serve static files automatically from a directory using StaticFiles "
        "mounted at a path with app.mount.\n",
        encoding="utf-8",
    )
    (docs / "reference" / "fastapi.md").write_text("# Reference\n\nignored page\n", encoding="utf-8")
    (docs / "release-notes.md").write_text("# Release Notes\n\nignored page\n", encoding="utf-8")
    return docs
