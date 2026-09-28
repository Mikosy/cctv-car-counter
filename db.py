"""
Database layer. Kept deliberately framework-agnostic (plain SQLAlchemy) so
this exact module can be imported by:
  - main.py (the offline/live CLI pipeline), and later
  - a FastAPI app (routes just import get_session() and the models below)
without any changes.
"""

from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, text
from sqlalchemy.orm import declarative_base, sessionmaker

import config

Base = declarative_base()


class VehicleEvent(Base):
    """One row per logged entry or exit."""

    __tablename__ = "vehicle_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    plate_number = Column(String, nullable=False, index=True)
    vehicle_type = Column(String, nullable=True)          # "car" | "motorcycle" | "bus" | "truck"
    direction = Column(String, nullable=False)          # "entry" | "exit"
    confidence = Column(Float, nullable=False)           # 0.0-1.0, from OCR voting
    track_id = Column(Integer, nullable=False)           # tracker's internal id
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    snapshot_path = Column(String, nullable=True)         # saved frame/crop, for manual audit
    source = Column(String, nullable=True)                # "live" or the file name used

    def __repr__(self):
        return (
            f"<VehicleEvent {self.plate_number} {self.direction} "
            f"conf={self.confidence:.2f} at {self.timestamp}>"
        )


_engine = create_engine(config.DB_URL, echo=False, future=True)
_SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)


def init_db():
    """Create tables if they don't exist yet. Safe to call every run."""
    Base.metadata.create_all(_engine)
    _ensure_vehicle_type_column()


def _ensure_vehicle_type_column():
    # create_all() only creates missing TABLES, not missing COLUMNS on a
    # table that already exists -- this patches an older anpr.db (from
    # before vehicle_type existed) in place. SQLite-specific; if you move to
    # Postgres later, use a real migration tool (e.g. Alembic) instead.
    with _engine.connect() as conn:
        columns = [row[1] for row in conn.execute(text("PRAGMA table_info(vehicle_events)"))]
        if "vehicle_type" not in columns:
            conn.execute(text("ALTER TABLE vehicle_events ADD COLUMN vehicle_type TEXT"))
            conn.commit()


def get_session():
    """Yields a session; use as a context manager or call .close() yourself."""
    return _SessionLocal()