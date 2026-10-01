"""database.py - Moteur SQLAlchemy et sessions.

Chaque transaction reçoit `app.user_id` (set_config local) : les triggers d'audit
PostgreSQL l'utilisent pour attribuer chaque modification à son auteur.
"""
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config import get_settings

_settings = get_settings()

engine = create_engine(_settings.database_url, pool_pre_ping=True, pool_size=20, max_overflow=20)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


@event.listens_for(SessionLocal, "after_begin")
def _contexte_transaction(session, transaction, connection):
    """Pose l'auteur de la transaction (vide pour les requêtes anonymes)."""
    connection.execute(
        text("SELECT set_config('app.user_id', :v, true)"),
        {"v": session.info.get("user_id", "")},
    )


def poser_secret_qr(db) -> None:
    """Expose la clé HMAC des QR à la transaction courante, uniquement quand c'est nécessaire
    (génération de convocations) pour limiter sa présence dans les échanges avec la base."""
    db.execute(text("SELECT set_config('app.qr_secret', :v, true)"), {"v": _settings.qr_secret})
