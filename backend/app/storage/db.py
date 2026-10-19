"""Database engine and sessions."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.storage.models import Base


class Database:
    def __init__(self, url: str) -> None:
        parsed = make_url(url)
        kwargs: dict[str, Any] = {}
        if parsed.get_backend_name() == "sqlite":
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
            if parsed.database in (None, "", ":memory:"):
                # One shared connection, so every session sees the same in-memory database.
                kwargs["poolclass"] = StaticPool
            else:
                Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
        self.engine: Engine = create_engine(url, **kwargs)
        if parsed.get_backend_name() == "sqlite":
            event.listen(self.engine, "connect", _sqlite_pragmas)
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)
        self._schema_ready = False

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)
        self._schema_ready = True

    def ensure_schema(self) -> None:
        """Create the tables on first use (the engine itself connects lazily)."""
        if not self._schema_ready:
            self.create_all()

    @contextmanager
    def session(self) -> Iterator[Session]:
        """A session that commits on success and rolls back on error."""
        session = self._sessions()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        self.engine.dispose()


def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    # SQLite ignores foreign keys (and so ON DELETE CASCADE) unless this is on.
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()
