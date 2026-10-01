"""assiduite.py - Algorithme du seuil d'assiduité (85 %) et QR Codes infalsifiables.

Règles (aucune exception, aucune dérogation politique ou administrative) :
  * Seules les séances réellement tenues (DISPENSE / REMPLACE) comptent : un cours non
    dispensé ne pénalise pas l'élève.
  * taux = présences / séances comptées x 100, arrondi à 2 décimales (demi supérieur).
  * bloque = (taux < seuil) : strictement inférieur ; 85.00 % exactement n'est PAS bloqué.
  * Le calcul d'autorité est exécuté par la base (trigger `convocation_calculer`) à l'insertion,
    puis figé : aucune route, aucune fonction ici ne peut modifier `bloque`, le taux ou le hash.
  * Ce module recalcule le hash côté Python et le compare à celui de la base : toute divergence
    (trigger altéré, clé différente, donnée falsifiée) lève IncoherenceConvocation.
Le hash est un HMAC-SHA256 (clé QR_SECRET) : impossible à forger sans la clé.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Iterator

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

import models as m
from config import get_settings
from database import poser_secret_qr
from enums import STATUTS_SEANCE_COMPTES, StatutPresence

PREFIXE_QR = "CIEDU1"          # version du format du QR : CIEDU1:<uuid convocation>:<hash hex>
_DEUX_DECIMALES = Decimal("0.01")


class IncoherenceConvocation(RuntimeError):
    """La convocation stockée ne correspond pas à l'algorithme : incident de sécurité à investiguer."""


class ErreurAssiduite(ValueError):
    pass


# ---------------------------------------------------------------------------
# 1. Taux d'assiduité
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResultatAssiduite:
    seances_comptees: int
    presences: int
    taux: Decimal
    seuil: Decimal

    @property
    def bloque(self) -> bool:
        return est_bloque(self.taux, self.seuil)


def calculer_taux(presences: int, seances_comptees: int) -> Decimal:
    """Sans séance comptée, l'élève n'a aucune absence à son passif : 100 %."""
    if seances_comptees < 0 or presences < 0 or presences > seances_comptees:
        raise ErreurAssiduite("Compteurs d'assiduité incohérents")
    if seances_comptees == 0:
        return Decimal("100.00")
    return (Decimal(100) * presences / seances_comptees).quantize(_DEUX_DECIMALES, rounding=ROUND_HALF_UP)


def est_bloque(taux: Decimal, seuil: Decimal) -> bool:
    """Règle unique du blocage : strictement sous le seuil."""
    return taux < seuil


def lire_seuil(db: Session) -> Decimal:
    valeur = db.scalar(select(m.ParametreSysteme.valeur).where(m.ParametreSysteme.cle == "seuil_assiduite_pct"))
    if valeur is None:
        raise ErreurAssiduite("Paramètre seuil_assiduite_pct manquant")
    return Decimal(valeur).quantize(_DEUX_DECIMALES)


def calculer_assiduite(db: Session, inscription_id: uuid.UUID) -> ResultatAssiduite:
    """Recalcule depuis les données sources (jamais depuis un compteur en cache)."""
    tenue = m.Seance.statut.in_(STATUTS_SEANCE_COMPTES)
    total, presents = db.execute(
        select(
            func.count(m.PresenceEleve.id).filter(tenue),
            func.count(m.PresenceEleve.id).filter(and_(tenue, m.PresenceEleve.statut == StatutPresence.PRESENT)),
        )
        .select_from(m.PresenceEleve)
        .join(m.Seance, and_(m.Seance.id == m.PresenceEleve.seance_id,
                             m.Seance.date_seance == m.PresenceEleve.date_seance))
        .where(m.PresenceEleve.inscription_id == inscription_id)
    ).one()
    return ResultatAssiduite(total, presents, calculer_taux(presents, total), lire_seuil(db))


# ---------------------------------------------------------------------------
# 2. Hash HMAC et QR Codes
# ---------------------------------------------------------------------------
def message_hash(conv_id, roll_number: str, session_id, taux: Decimal, bloque: bool, genere_le: datetime) -> str:
    """Message canonique : DOIT rester identique à la concaténation du trigger SQL convocation_calculer()."""
    horodatage = genere_le.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
    return "|".join([
        str(conv_id), roll_number, str(session_id),
        f"{Decimal(taux).quantize(_DEUX_DECIMALES):.2f}",
        "true" if bloque else "false",
        horodatage,
    ])


def calculer_hash_qr(secret: str, conv_id, roll_number, session_id, taux, bloque, genere_le) -> str:
    message = message_hash(conv_id, roll_number, session_id, taux, bloque, genere_le)
    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def _roll_number(db: Session, inscription_id) -> str:
    roll = db.scalar(
        select(m.Eleve.roll_number).join(m.Inscription, m.Inscription.eleve_id == m.Eleve.id)
        .where(m.Inscription.id == inscription_id)
    )
    if roll is None:
        raise ErreurAssiduite("Inscription introuvable")
    return roll


def verifier_integrite(db: Session, conv: m.Convocation, roll_number: str | None = None) -> None:
    """Contrôle croisé Python/SQL d'une convocation stockée."""
    roll = roll_number or _roll_number(db, conv.inscription_id)
    attendu = calculer_hash_qr(get_settings().qr_secret, conv.id, roll, conv.session_id,
                               conv.taux_assiduite_fige, conv.bloque, conv.genere_le)
    if not hmac.compare_digest(attendu, conv.hash_qr):
        raise IncoherenceConvocation(f"Hash divergent pour la convocation {conv.id}")
    if conv.bloque != est_bloque(conv.taux_assiduite_fige, conv.seuil_applique):
        raise IncoherenceConvocation(f"Statut de blocage incohérent pour la convocation {conv.id}")


def payload_qr(conv: m.Convocation) -> str:
    return f"{PREFIXE_QR}:{conv.id}:{conv.hash_qr}"


def generer_image_qr(payload: str) -> bytes:
    """PNG du QR Code (bibliothèque `qrcode`, pur Python). Correction d'erreur M : lisible même abîmé."""
    import qrcode
    image = qrcode.make(payload, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=8, border=3)
    tampon = io.BytesIO()
    image.save(tampon, format="PNG")
    return tampon.getvalue()


# ---------------------------------------------------------------------------
# 3. Génération des convocations
# ---------------------------------------------------------------------------
def generer_convocation(db: Session, session_id: uuid.UUID, inscription_id: uuid.UUID) -> tuple[m.Convocation, bool]:
    """Crée (ou retrouve) la convocation d'un élève. Retourne (convocation, creee).

    L'appelant commit. Une fois insérée, la ligne est verrouillée par trigger : aucune
    régénération ni mise à jour possible ; il n'existe volontairement aucune fonction de déblocage.
    """
    existante = db.scalar(select(m.Convocation).where(
        m.Convocation.session_id == session_id, m.Convocation.inscription_id == inscription_id))
    if existante is not None:
        return existante, False
    if db.get(m.SessionExamen, session_id) is None:
        raise ErreurAssiduite("Session d'examen introuvable")
    if db.get(m.Inscription, inscription_id) is None:
        raise ErreurAssiduite("Inscription introuvable")

    poser_secret_qr(db)
    conv = m.Convocation(session_id=session_id, inscription_id=inscription_id)
    db.add(conv)
    db.flush()                  # le trigger calcule taux, seuil, bloque, hash ; eager_defaults les relit
    verifier_integrite(db, conv)
    return conv, True


def generer_convocations_session(db: Session, session_id: uuid.UUID) -> tuple[int, list[uuid.UUID]]:
    """Génération en masse (SQL set-based) puis contrôle d'intégrité Python de toute la session."""
    if db.get(m.SessionExamen, session_id) is None:
        raise ErreurAssiduite("Session d'examen introuvable")
    poser_secret_qr(db)
    crees = db.scalar(select(func.generer_convocations(session_id))) or 0
    db.flush()
    return crees, controler_session(db, session_id)


def controler_session(db: Session, session_id: uuid.UUID) -> list[uuid.UUID]:
    """Retourne les identifiants des convocations dont l'intégrité est douteuse."""
    anomalies: list[uuid.UUID] = []
    requete = (
        select(m.Convocation, m.Eleve.roll_number)
        .join(m.Inscription, m.Inscription.id == m.Convocation.inscription_id)
        .join(m.Eleve, m.Eleve.id == m.Inscription.eleve_id)
        .where(m.Convocation.session_id == session_id)
        .execution_options(yield_per=2000)
    )
    for conv, roll in db.execute(requete):
        try:
            verifier_integrite(db, conv, roll)
        except IncoherenceConvocation:
            anomalies.append(conv.id)
    return anomalies


# ---------------------------------------------------------------------------
# 4. Validation d'un QR Code (contrôle à l'entrée de la salle d'examen)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResultatQR:
    valide: bool
    bloque: bool | None = None
    roll_number: str | None = None
    nom: str | None = None
    prenoms: str | None = None
    raison: str | None = None

    @property
    def autorise_entree(self) -> bool:
        """Entrée autorisée uniquement si le QR est authentique ET la convocation non bloquée."""
        return self.valide and self.bloque is False


def valider_qr(db: Session, payload: str) -> ResultatQR:
    parties = payload.strip().split(":")
    if len(parties) != 3 or parties[0] != PREFIXE_QR:
        return ResultatQR(False, raison="FORMAT_INVALIDE")
    try:
        conv_id = uuid.UUID(parties[1])
    except ValueError:
        return ResultatQR(False, raison="FORMAT_INVALIDE")
    hash_recu = parties[2].lower()

    ligne = db.execute(
        select(m.Convocation, m.Eleve.roll_number, m.Eleve.nom, m.Eleve.prenoms)
        .join(m.Inscription, m.Inscription.id == m.Convocation.inscription_id)
        .join(m.Eleve, m.Eleve.id == m.Inscription.eleve_id)
        .where(m.Convocation.id == conv_id)
    ).first()
    # Même réponse si la convocation n'existe pas ou si le hash est faux : pas d'oracle pour un attaquant
    if ligne is None or not hmac.compare_digest(ligne[0].hash_qr, hash_recu):
        return ResultatQR(False, raison="QR_INVALIDE")
    conv, roll, nom, prenoms = ligne
    try:
        verifier_integrite(db, conv, roll)
    except IncoherenceConvocation:
        return ResultatQR(False, raison="INTEGRITE_COMPROMISE")
    return ResultatQR(True, conv.bloque, roll, nom, prenoms,
                      raison="CONVOCATION_BLOQUEE_ASSIDUITE" if conv.bloque else None)
