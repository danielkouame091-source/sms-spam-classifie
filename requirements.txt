"""config.py - Configuration par variables d'environnement.

Aucun secret n'a de valeur par défaut : l'application refuse de démarrer si
JWT_SECRET ou QR_SECRET manquent (32 caractères minimum, et distincts).
"""
import os
from dataclasses import dataclass
from functools import lru_cache


class ConfigurationManquante(RuntimeError):
    pass


def _requis(nom: str, min_len: int = 1) -> str:
    valeur = os.environ.get(nom, "")
    if len(valeur) < min_len:
        raise ConfigurationManquante(
            f"Variable d'environnement {nom} absente ou trop courte (minimum {min_len} caractères)"
        )
    return valeur


@dataclass(frozen=True)
class Settings:
    database_url: str
    jwt_secret: str
    qr_secret: str
    jwt_expiration_min: int = 30
    tolerance_horloge_s: int = 300          # un terminal ne peut pas dater dans le futur au-delà de 5 min
    max_echecs_connexion: int = 5
    duree_verrouillage_min: int = 15


@lru_cache
def get_settings() -> Settings:
    jwt_secret = _requis("JWT_SECRET", 32)
    qr_secret = _requis("QR_SECRET", 32)
    if jwt_secret == qr_secret:
        raise ConfigurationManquante("JWT_SECRET et QR_SECRET doivent être différents")
    return Settings(
        database_url=_requis("DATABASE_URL"),   # ex. postgresql+psycopg://app_api:***@host/ecole_ci
        jwt_secret=jwt_secret,
        qr_secret=qr_secret,
        jwt_expiration_min=int(os.environ.get("JWT_EXPIRATION_MIN", "30")),
    )
