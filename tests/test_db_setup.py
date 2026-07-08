"""Tests for database migrations and seeding script."""

import os
import subprocess

import pytest
from sqlalchemy import create_engine, inspect, text

from scripts.seed import seed_db
from src.config.settings import settings

# Use the real test DB URL from environment or fallback to a separate test DB
TEST_DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/engageiq_dev"
)

# Connect to the default 'postgres' database to create/drop the test database
ADMIN_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/postgres"
TEST_DB_NAME = "engageiq_test_migration"


import sys


def run_alembic_command(command: str) -> None:
    """Run an Alembic CLI command."""
    env = os.environ.copy()
    # Override the DB URL for Alembic
    env["DATABASE_URL"] = TEST_DATABASE_URL.replace("engageiq_dev", TEST_DB_NAME)
    subprocess.run(
        [sys.executable, "-m", "alembic"] + command.split(),
        env=env,
        check=True,
        capture_output=True,
    )


@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    """Create a fresh test database for migration testing, drop it after."""
    engine = create_engine(ADMIN_DATABASE_URL, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        conn.execute(text(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}"))
        conn.execute(text(f"CREATE DATABASE {TEST_DB_NAME}"))

    yield

    with engine.connect() as conn:
        # Terminate any remaining connections to the test database
        conn.execute(text(f"""
            SELECT pg_terminate_backend(pg_stat_activity.pid)
            FROM pg_stat_activity
            WHERE pg_stat_activity.datname = '{TEST_DB_NAME}'
              AND pid <> pg_backend_pid();
        """))
        conn.execute(text(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}"))


@pytest.fixture
def engine():
    """Provides a SQLAlchemy engine connected to the test DB."""
    test_db_url = TEST_DATABASE_URL.replace("engageiq_dev", TEST_DB_NAME)
    eng = create_engine(test_db_url)
    yield eng
    eng.dispose()


class TestMigrations:
    def test_alembic_upgrade_and_downgrade(self, engine):
        """Test that migrations can apply and rollback cleanly."""

        # 1. Start with an empty database
        inspector = inspect(engine)
        assert len(inspector.get_table_names()) == 0

        # 2. Upgrade to head
        run_alembic_command("upgrade head")

        # Verify tables are created
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        expected_tables = {
            "alembic_version",
            "users",
            "courses",
            "sessions",
            "engagement_logs",
            "nudges",
            "reports",
        }
        assert expected_tables.issubset(set(tables))

        # 3. Downgrade to base
        run_alembic_command("downgrade base")

        # Verify all tables are removed (except possibly alembic_version)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert len([t for t in tables if t != "alembic_version"]) == 0


class TestSeedData:
    def test_seed_script_idempotency_and_counts(self, engine, monkeypatch):
        """Test that the seed script populates correct counts and is idempotent."""
        # Upgrade to head first
        run_alembic_command("upgrade head")

        # Override the setting inside the seed script to use test DB
        test_db_url = TEST_DATABASE_URL.replace("engageiq_dev", TEST_DB_NAME)
        monkeypatch.setattr(settings, "database_url", test_db_url)

        # 1. Run seed script once
        seed_db()

        # Verify counts
        with engine.connect() as conn:
            users_count = conn.execute(text("SELECT count(*) FROM users")).scalar()
            courses_count = conn.execute(text("SELECT count(*) FROM courses")).scalar()
            sessions_count = conn.execute(
                text("SELECT count(*) FROM sessions")
            ).scalar()

            # Acceptance criteria: 2 teachers + 10 students = 12 users
            assert users_count == 12
            assert courses_count == 3
            assert sessions_count == 5

        # 2. Run seed script again to test idempotency
        seed_db()

        # Counts should not change
        with engine.connect() as conn:
            assert (
                users_count == conn.execute(text("SELECT count(*) FROM users")).scalar()
            )
            assert (
                courses_count
                == conn.execute(text("SELECT count(*) FROM courses")).scalar()
            )
            assert (
                sessions_count
                == conn.execute(text("SELECT count(*) FROM sessions")).scalar()
            )
