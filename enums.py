"""crud.py - Logique métier : pointage des professeurs, service fait, impact salarial,
synchronisation des appels, périmètres d'accès.

Principes :
  * Les règles d'intégrité sont appliquées par PostgreSQL (triggers) ET contrôlées ici pour
    renvoyer des erreurs claires. La base a toujours le dernier mot.
  * Chaque fonction qui écrit gère son commit et traduit les erreurs de la base en ErreurMetier.
  * Aucune fonction ne permet de modifier un blocage de convocation.
"""
from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import exists, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session

import assiduite
import models as m
import schemas as s
import security
from config import get_settings
from enums import RoleUtilisateur, StatutPresence, StatutRetenue, StatutSeance

ABIDJAN = ZoneInfo("Africa/Abidjan")


class ErreurMetier(Exception):
    def __init__(self, message: str, statut_http: int = 422, code: str = "ERREUR_METIER"):
        super().__init__(message)
        self.message, self.statut_http, self.code = message, statut_http, code


def maintenant() -> datetime:
    return datetime.now(timezone.utc)


def _message_pg(e: DBAPIError) -> str:
    diag = getattr(e.orig, "diag", None)
    return (getattr(diag, "message_primary", None) or str(e.orig)).strip()


@contextmanager
def traduire_erreurs_db(db: Session):
    """Convertit les erreurs PostgreSQL (contraintes, RAISE EXCEPTION des triggers) en ErreurMetier."""
    try:
        yield
    except IntegrityError as e:
        db.rollback()
        raise ErreurMetier("Conflit d'intégrité : donnée déjà existante ou référence invalide", 409, "INTEGRITE") from e
    except DBAPIError as e:
        db.rollback()
        sqlstate = getattr(e.orig, "sqlstate", None) or getattr(e.orig, "pgcode", None)
        if sqlstate == "P0001":                      # RAISE EXCEPTION d'un trigger / d'une fonction
            raise ErreurMetier(_message_pg(e), 422, "REGLE_METIER_DB") from e
        raise


# ---------------------------------------------------------------------------
# Périmètres d'accès
# ---------------------------------------------------------------------------
def _etab_sous_structure(db: Session, etablissement_id: uuid.UUID, structure_racine: uuid.UUID) -> bool:
    arbre = select(m.StructureAdministrative.id).where(m.StructureAdministrative.id == structure_racine).cte(recursive=True)
    arbre = arbre.union_all(select(m.StructureAdministrative.id).where(m.StructureAdministrative.parent_id == arbre.c.id))
    return bool(db.scalar(select(exists().where(
        m.Etablissement.id == etablissement_id, m.Etablissement.structure_id.in_(select(arbre.c.id))))))


def verifier_portee(db: Session, claims: security.Claims, etablissement_id: uuid.UUID) -> None:
    if claims.role == RoleUtilisateur.ADMIN:
        return
    if claims.role == RoleUtilisateur.INSPECTEUR and claims.struct:
        if _etab_sous_structure(db, etablissement_id, claims.struct):
            return
    elif claims.etab is not None and claims.etab == etablissement_id:
        return
    raise ErreurMetier("Établissement hors de votre périmètre", 403, "HORS_PERIMETRE")


def _etab_de_classe(db: Session, classe_id: uuid.UUID) -> uuid.UUID:
    etab = db.scalar(select(m.Classe.etablissement_id).where(m.Classe.id == classe_id))
    if etab is None:
        raise ErreurMetier("Classe introuvable", 404, "INTROUVABLE")
    return etab


def professeur_de(db: Session, claims: security.Claims) -> m.Professeur | None:
    return db.scalar(select(m.Professeur).where(m.Professeur.utilisateur_id == claims.sub))


# ---------------------------------------------------------------------------
# Authentification
# ---------------------------------------------------------------------------
def authentifier(db: Session, identifiant: str, mot_de_passe: str) -> m.Utilisateur:
    cfg = get_settings()
    cle = identifiant.strip()
    u = db.scalar(select(m.Utilisateur).where(
        (func.lower(m.Utilisateur.email) == cle.lower()) | (m.Utilisateur.telephone == cle)))
    if u is None:
        security.verifier_factice(mot_de_passe)
        raise ErreurMetier("Identifiants invalides", 401, "AUTH")
    if u.verrouille_jusqua and u.verrouille_jusqua > maintenant():
        raise ErreurMetier("Compte temporairement verrouillé", 423, "VERROUILLE")
    if not security.verifier(mot_de_passe, u.mot_de_passe_hash) or not u.actif:
        u.tentatives_echec += 1
        if u.tentatives_echec >= cfg.max_echecs_connexion:
            u.verrouille_jusqua = maintenant() + timedelta(minutes=cfg.duree_verrouillage_min)
            u.tentatives_echec = 0
        db.commit()
        raise ErreurMetier("Identifiants invalides", 401, "AUTH")
    u.tentatives_echec, u.verrouille_jusqua, u.derniere_connexion = 0, None, maintenant()
    if security.doit_rehacher(u.mot_de_passe_hash):
        u.mot_de_passe_hash = security.hacher(mot_de_passe)
    db.commit()
    return u


# ---------------------------------------------------------------------------
# Référentiel : utilisateurs, établissements, professeurs, élèves
# ---------------------------------------------------------------------------
def creer_utilisateur(db: Session, data: s.UtilisateurCreate) -> m.Utilisateur:
    u = m.Utilisateur(
        role=data.role, etablissement_id=data.etablissement_id, structure_id=data.structure_id,
        email=str(data.email).lower() if data.email else None, telephone=data.telephone,
        mot_de_passe_hash=security.hacher(data.mot_de_passe))
    with traduire_erreurs_db(db):
        db.add(u)
        db.commit()
    return u


def creer_etablissement(db: Session, data: s.EtablissementCreate) -> m.Etablissement:
    e = m.Etablissement(**data.model_dump())
    with traduire_erreurs_db(db):
        db.add(e)
        db.commit()
    return e


def creer_professeur(db: Session, data: s.ProfesseurCreate, claims: security.Claims) -> m.Professeur:
    verifier_portee(db, claims, data.etablissement_id)
    p = m.Professeur(**data.model_dump())
    with traduire_erreurs_db(db):
        db.add(p)
        db.commit()
    return p


def creer_eleve(db: Session, data: s.EleveCreate, claims: security.Claims) -> m.Eleve:
    champs = data.model_dump(exclude={"classe_id"})
    classe = None
    if data.classe_id:
        classe = db.get(m.Classe, data.classe_id)
        if classe is None:
            raise ErreurMetier("Classe introuvable", 404, "INTROUVABLE")
        verifier_portee(db, claims, classe.etablissement_id)
    elif claims.role != RoleUtilisateur.ADMIN and claims.role != RoleUtilisateur.DIRECTEUR:
        raise ErreurMetier("Précisez la classe d'inscription", 422)
    eleve = m.Eleve(**champs)
    with traduire_erreurs_db(db):
        db.add(eleve)
        db.flush()                     # le Roll Number est attribué par la base (séquence + clé de contrôle)
        if classe:
            db.add(m.Inscription(eleve_id=eleve.id, classe_id=classe.id, annee_id=classe.annee_id))
        db.commit()
    return eleve


# ---------------------------------------------------------------------------
# Séances et remplacements
# ---------------------------------------------------------------------------
def seances_du_jour(db: Session, professeur_id: uuid.UUID, jour: date, claims: security.Claims) -> list[m.Seance]:
    prof = db.get(m.Professeur, professeur_id)
    if prof is None:
        raise ErreurMetier("Professeur introuvable", 404, "INTROUVABLE")
    if claims.role == RoleUtilisateur.PROFESSEUR:
        moi = professeur_de(db, claims)
        if moi is None or moi.id != professeur_id:
            raise ErreurMetier("Vous ne pouvez consulter que votre propre emploi du temps", 403, "HORS_PERIMETRE")
    else:
        verifier_portee(db, claims, prof.etablissement_id)
    return list(db.scalars(select(m.Seance).where(
        m.Seance.date_seance == jour,
        (m.Seance.professeur_prevu_id == professeur_id) | (m.Seance.professeur_effectif_id == professeur_id),
    ).order_by(m.Seance.debut)))


def declarer_remplacement(db: Session, data: s.RemplacementCreate, claims: security.Claims) -> m.Remplacement:
    """Déclare un remplacement. La date (horloge serveur) et la conformité du préavis de 24 h sont
    décidées par la base : impossible de les antidater ou de les modifier ensuite."""
    seance = db.get(m.Seance, data.seance_id)
    if seance is None:
        raise ErreurMetier("Séance introuvable", 404, "INTROUVABLE")
    etab = _etab_de_classe(db, seance.classe_id)
    if claims.role == RoleUtilisateur.PROFESSEUR:
        moi = professeur_de(db, claims)
        if moi is None or moi.id != seance.professeur_prevu_id:
            raise ErreurMetier("Seul le professeur prévu ou sa direction peut déclarer un remplacement", 403, "HORS_PERIMETRE")
    else:
        verifier_portee(db, claims, etab)
    remplacant = db.get(m.Professeur, data.professeur_remplacant_id)
    if remplacant is None or not remplacant.actif:
        raise ErreurMetier("Remplaçant introuvable ou inactif", 404, "INTROUVABLE")
    if remplacant.etablissement_id != etab:
        raise ErreurMetier("Le remplaçant doit appartenir au même établissement", 422, "REMPLACANT_HORS_ETAB")

    r = m.Remplacement(seance_id=seance.id, professeur_remplacant_id=remplacant.id, declare_par=claims.sub)
    with traduire_erreurs_db(db):
        db.add(r)
        db.commit()
    return r


# ---------------------------------------------------------------------------
# Pointage des professeurs (service fait)
# ---------------------------------------------------------------------------
def enregistrer_pointage(db: Session, data: s.PointageCreate, claims: security.Claims) -> tuple[m.PointageProfesseur, bool, m.Seance]:
    """Enregistre un pointage biométrique. Idempotent (un terminal hors ligne peut renvoyer le même).
    Le trigger `pointage_appliquer` fait passer la séance en DISPENSE / REMPLACE si le pointage est
    valide (professeur prévu, ou remplaçant déclaré avec préavis >= 24 h)."""
    prof = db.get(m.Professeur, data.professeur_id)
    seance = db.get(m.Seance, data.seance_id)
    if prof is None or not prof.actif or seance is None:
        raise ErreurMetier("Professeur ou séance introuvable", 404, "INTROUVABLE")
    etab = _etab_de_classe(db, seance.classe_id)
    verifier_portee(db, claims, etab)
    if prof.etablissement_id != etab:
        raise ErreurMetier("Ce professeur n'appartient pas à l'établissement de la séance", 422, "PROF_HORS_ETAB")
    if data.horodatage > maintenant() + timedelta(seconds=get_settings().tolerance_horloge_s):
        raise ErreurMetier("Horodatage dans le futur : horloge du terminal à vérifier", 422, "HORODATAGE_FUTUR")
    if data.terminal_id:
        terminal = db.get(m.TerminalBiometrique, data.terminal_id)
        if terminal is None or not terminal.actif or terminal.etablissement_id != etab:
            raise ErreurMetier("Terminal inconnu, inactif ou d'un autre établissement", 422, "TERMINAL_INVALIDE")

    existant = db.scalar(select(m.PointageProfesseur).where(
        m.PointageProfesseur.seance_id == seance.id,
        m.PointageProfesseur.professeur_id == prof.id,
        m.PointageProfesseur.type == data.type))
    if existant is not None:
        return existant, True, seance

    p = m.PointageProfesseur(
        id=data.id or uuid.uuid4(), professeur_id=prof.id, seance_id=seance.id, type=data.type,
        horodatage=data.horodatage, terminal_id=data.terminal_id,
        score_correspondance=data.score_correspondance, signature_terminal=data.signature_terminal)
    with traduire_erreurs_db(db):
        db.add(p)
        db.commit()
    db.refresh(seance)                 # relit le statut décidé par le trigger
    return p, False, seance


# ---------------------------------------------------------------------------
# Détection des cours non dispensés et impact salarial
# ---------------------------------------------------------------------------
_SQL_GENERER_RETENUES = text("""
    INSERT INTO retenues_salaire
        (id, professeur_id, seance_id, duree_minutes, taux_horaire_fcfa, montant_fcfa, secteur, contestable_jusqua)
    SELECT gen_random_uuid(), p.id, sc.id,
           (extract(epoch FROM sc.fin - sc.debut) / 60)::int,
           p.taux_horaire_fcfa,
           round((p.taux_horaire_fcfa * extract(epoch FROM sc.fin - sc.debut) / 3600)::numeric, 2),
           p.secteur,
           now() + make_interval(days => param('delai_contestation_j')::int)
      FROM seances_cours sc
      JOIN professeurs p ON p.id = sc.professeur_prevu_id
     WHERE sc.statut = 'NON_DISPENSE' AND p.taux_horaire_fcfa IS NOT NULL
    ON CONFLICT (seance_id) DO NOTHING
    RETURNING id
""")

_SQL_SANS_TAUX = text("""
    SELECT count(*) FROM seances_cours sc JOIN professeurs p ON p.id = sc.professeur_prevu_id
     WHERE sc.statut = 'NON_DISPENSE' AND p.taux_horaire_fcfa IS NULL
""")

# Une séance régularisée (pointage reçu tardivement après synchronisation hors ligne) annule la retenue
# tant qu'elle n'est pas finalisée.
_SQL_ANNULER_OBSOLETES = text("""
    UPDATE retenues_salaire r
       SET statut = 'ANNULEE', decide_le = now(),
           motif_decision = 'Séance régularisée : pointage ou statut officiel reçu ultérieurement'
      FROM seances_cours sc
     WHERE sc.id = r.seance_id AND r.statut IN ('EN_CONTESTATION', 'CONTESTEE')
       AND sc.statut IN ('DISPENSE', 'REMPLACE', 'ANNULE_OFFICIEL')
    RETURNING r.id
""")

_SQL_VALIDER_ECHUES = text("""
    UPDATE retenues_salaire
       SET statut = 'VALIDEE', decide_le = now(),
           motif_decision = 'Validation automatique : délai de contestation écoulé'
     WHERE statut = 'EN_CONTESTATION' AND contestable_jusqua < now()
    RETURNING id
""")


def evaluer_et_generer_retenues(db: Session) -> s.EvaluationResultat:
    """Tâche planifiée (toutes les heures) :
    1. séances terminées depuis > délai sans pointage valide => NON_DISPENSE ;
    2. création des retenues (fenêtre de contestation ouverte) ;
    3. annulation des retenues devenues sans objet ;
    4. validation automatique des retenues non contestées à l'échéance."""
    with traduire_erreurs_db(db):
        non_dispensees = db.scalar(text("SELECT evaluer_seances_terminees()")) or 0
        creees = len(db.execute(_SQL_GENERER_RETENUES).all())
        annulees = len(db.execute(_SQL_ANNULER_OBSOLETES).all())
        validees = len(db.execute(_SQL_VALIDER_ECHUES).all())
        sans_taux = db.scalar(_SQL_SANS_TAUX) or 0
        db.commit()
    return s.EvaluationResultat(seances_non_dispensees=non_dispensees, retenues_creees=creees,
                                retenues_annulees=annulees, retenues_validees=validees, sans_taux_horaire=sans_taux)


def lister_retenues(db: Session, claims: security.Claims, statut: StatutRetenue | None = None,
                    professeur_id: uuid.UUID | None = None) -> list[m.RetenueSalaire]:
    requete = select(m.RetenueSalaire).join(m.Professeur, m.Professeur.id == m.RetenueSalaire.professeur_id)
    if claims.role == RoleUtilisateur.PROFESSEUR:
        moi = professeur_de(db, claims)
        if moi is None:
            return []
        requete = requete.where(m.RetenueSalaire.professeur_id == moi.id)
    elif claims.role in (RoleUtilisateur.DIRECTEUR,):
        requete = requete.where(m.Professeur.etablissement_id == claims.etab)
    elif claims.role == RoleUtilisateur.INSPECTEUR:
        raise ErreurMetier("Lecture des retenues non autorisée pour ce rôle", 403, "HORS_PERIMETRE")
    if statut:
        requete = requete.where(m.RetenueSalaire.statut == statut)
    if professeur_id:
        requete = requete.where(m.RetenueSalaire.professeur_id == professeur_id)
    return list(db.scalars(requete.order_by(m.RetenueSalaire.cree_le.desc()).limit(500)))


def contester_retenue(db: Session, retenue_id: uuid.UUID, motif: str, claims: security.Claims) -> m.RetenueSalaire:
    r = db.get(m.RetenueSalaire, retenue_id)
    moi = professeur_de(db, claims)
    if r is None or moi is None or r.professeur_id != moi.id:
        raise ErreurMetier("Retenue introuvable", 404, "INTROUVABLE")
    if r.statut != StatutRetenue.EN_CONTESTATION or r.contestable_jusqua < maintenant():
        raise ErreurMetier("Le délai de contestation est clos ou la retenue est déjà traitée", 409, "DELAI_CLOS")
    r.statut, r.motif_contestation = StatutRetenue.CONTESTEE, motif
    with traduire_erreurs_db(db):
        db.commit()
    return r


def trancher_retenue(db: Session, retenue_id: uuid.UUID, data: s.DecisionRetenueIn, claims: security.Claims) -> m.RetenueSalaire:
    """Décision humaine sur une retenue contestée (direction de l'établissement ou administration)."""
    r = db.get(m.RetenueSalaire, retenue_id)
    if r is None:
        raise ErreurMetier("Retenue introuvable", 404, "INTROUVABLE")
    prof = db.get(m.Professeur, r.professeur_id)
    verifier_portee(db, claims, prof.etablissement_id)
    if r.statut != StatutRetenue.CONTESTEE:
        raise ErreurMetier("Seule une retenue contestée peut être tranchée", 409, "STATUT_INVALIDE")
    r.statut = StatutRetenue.VALIDEE if data.accepter else StatutRetenue.ANNULEE
    r.motif_decision, r.decide_par, r.decide_le = data.motif, claims.sub, maintenant()
    with traduire_erreurs_db(db):
        db.commit()
    return r


# ---------------------------------------------------------------------------
# Appel élèves : liste de classe (téléchargement hors ligne) et synchronisation
# ---------------------------------------------------------------------------
def _professeur_autorise_sur_seance(db: Session, prof: m.Professeur | None, seance: m.Seance) -> bool:
    if prof is None:
        return False
    if prof.id in (seance.professeur_prevu_id, seance.professeur_effectif_id):
        return True
    return bool(db.scalar(select(exists().where(
        m.Remplacement.seance_id == seance.id, m.Remplacement.professeur_remplacant_id == prof.id))))


def _controler_acces_seance(db: Session, seance: m.Seance, claims: security.Claims) -> None:
    if claims.role == RoleUtilisateur.PROFESSEUR:
        if not _professeur_autorise_sur_seance(db, professeur_de(db, claims), seance):
            raise ErreurMetier("Vous n'êtes pas le professeur de cette séance", 403, "HORS_PERIMETRE")
    else:
        verifier_portee(db, claims, _etab_de_classe(db, seance.classe_id))


def liste_appel(db: Session, seance_id: uuid.UUID, claims: security.Claims) -> s.ListeAppelOut:
    seance = db.get(m.Seance, seance_id)
    if seance is None:
        raise ErreurMetier("Séance introuvable", 404, "INTROUVABLE")
    _controler_acces_seance(db, seance, claims)
    lignes = db.execute(
        select(m.Inscription.id, m.Eleve.roll_number, m.Eleve.nom, m.Eleve.prenoms)
        .join(m.Eleve, m.Eleve.id == m.Inscription.eleve_id)
        .where(m.Inscription.classe_id == seance.classe_id,
               (m.Inscription.date_sortie.is_(None)) | (m.Inscription.date_sortie > seance.date_seance))
        .order_by(m.Eleve.nom, m.Eleve.prenoms)).all()
    return s.ListeAppelOut(
        seance_id=seance.id, date_seance=seance.date_seance, classe_id=seance.classe_id,
        eleves=[s.EleveAppelOut(inscription_id=i, roll_number=r, nom=n, prenoms=p) for i, r, n, p in lignes])


def enregistrer_appels_sync(db: Session, req: s.AppelSyncRequest, claims: security.Claims) -> s.AppelSyncResponse:
    """Reçoit un lot d'appels créés hors ligne. Chaque ligne est validée indépendamment et le lot est
    IDEMPOTENT : renvoyer une ligne déjà reçue (même id, ou même statut) est accepté sans doublon ;
    un statut contradictoire déjà enregistré est rejeté en CONFLIT (jamais écrasé en silence)."""
    tolerance = timedelta(seconds=get_settings().tolerance_horloge_s)
    prof = professeur_de(db, claims) if claims.role == RoleUtilisateur.PROFESSEUR else None

    seances = {x.id: x for x in db.scalars(select(m.Seance).where(m.Seance.id.in_(list({l.seance_id for l in req.lignes}))))}
    inscriptions = {
        i.id: (i.classe_id, i.date_sortie, roll)
        for i, roll in db.execute(
            select(m.Inscription, m.Eleve.roll_number).join(m.Eleve, m.Eleve.id == m.Inscription.eleve_id)
            .where(m.Inscription.id.in_(list({l.inscription_id for l in req.lignes}))))
    }
    acces: dict[uuid.UUID, bool] = {}

    acceptes: list[uuid.UUID] = []
    rejetes: list[s.LigneRejetee] = []
    classes_touchees: set[uuid.UUID] = set()

    def rejeter(ligne: s.AppelLigneSync, code: str, raison: str):
        rejetes.append(s.LigneRejetee(id=ligne.id, code=code, raison=raison))

    with traduire_erreurs_db(db):
        for l in req.lignes:
            seance = seances.get(l.seance_id)
            if seance is None or seance.date_seance != l.date_seance:
                rejeter(l, "SEANCE_INCONNUE", "Séance introuvable ou date incohérente"); continue
            if seance.id not in acces:
                try:
                    _controler_acces_seance(db, seance, claims); acces[seance.id] = True
                except ErreurMetier:
                    acces[seance.id] = False
            if not acces[seance.id]:
                rejeter(l, "NON_AUTORISE", "Vous n'êtes pas habilité à saisir l'appel de cette séance"); continue
            if seance.statut == StatutSeance.ANNULE_OFFICIEL:
                rejeter(l, "SEANCE_ANNULEE", "Séance officiellement annulée"); continue
            insc = inscriptions.get(l.inscription_id)
            if insc is None or insc[0] != seance.classe_id:
                rejeter(l, "ELEVE_HORS_CLASSE", "Inscription inconnue ou d'une autre classe"); continue
            if insc[2] != l.roll_number:
                rejeter(l, "ROLL_INCOHERENT", "Le Roll Number ne correspond pas à l'inscription"); continue
            if insc[1] is not None and insc[1] <= seance.date_seance:
                rejeter(l, "ELEVE_SORTI", "Élève sorti de l'établissement avant cette séance"); continue
            if l.horodatage > maintenant() + tolerance:
                rejeter(l, "HORODATAGE_FUTUR", "Horodatage dans le futur : horloge de l'appareil à vérifier"); continue
            if l.horodatage.astimezone(ABIDJAN).date() != seance.date_seance:
                rejeter(l, "HORODATAGE_INCOHERENT", "L'appel n'a pas été fait le jour de la séance"); continue

            inseree = db.execute(
                pg_insert(m.PresenceEleve).values(
                    id=l.id, seance_id=seance.id, date_seance=seance.date_seance,
                    inscription_id=l.inscription_id, ordre_appel=l.ordre_appel,
                    statut=l.statut, horodatage=l.horodatage, saisi_par=claims.sub)
                .on_conflict_do_nothing().returning(m.PresenceEleve.id)).first()
            if inseree is None:                                     # déjà reçu : idempotent ou conflit
                existant = db.execute(select(m.PresenceEleve.id, m.PresenceEleve.statut).where(
                    m.PresenceEleve.seance_id == seance.id, m.PresenceEleve.date_seance == seance.date_seance,
                    m.PresenceEleve.inscription_id == l.inscription_id)).first()
                if existant is None or (existant.id != l.id and existant.statut != l.statut):
                    rejeter(l, "CONFLIT", f"Appel déjà enregistré avec le statut {existant.statut.value if existant else '?'}")
                    continue
            acceptes.append(l.id)
            classes_touchees.add(seance.classe_id)
        db.commit()

        # Met à jour le suivi d'assiduité (compteurs et statut) des classes concernées
        for classe_id in classes_touchees:
            db.execute(text("SELECT recalculer_assiduite(p_classe => CAST(:c AS uuid))"), {"c": str(classe_id)})
        db.commit()
    return s.AppelSyncResponse(lot_id=req.lot_id, acceptes=acceptes, rejetes=rejetes)


# ---------------------------------------------------------------------------
# Assiduité et convocations
# ---------------------------------------------------------------------------
def assiduite_inscription(db: Session, inscription_id: uuid.UUID, claims: security.Claims) -> s.AssiduiteOut:
    insc = db.get(m.Inscription, inscription_id)
    if insc is None:
        raise ErreurMetier("Inscription introuvable", 404, "INTROUVABLE")
    verifier_portee(db, claims, _etab_de_classe(db, insc.classe_id))
    r = assiduite.calculer_assiduite(db, inscription_id)
    return s.AssiduiteOut(
        inscription_id=inscription_id, seances_comptees=r.seances_comptees, presences=r.presences,
        taux=r.taux, seuil=r.seuil, bloque_si_convoque_maintenant=r.bloque, statut=insc.statut_assiduite)


def creer_convocation(db: Session, data: s.ConvocationCreate, claims: security.Claims) -> tuple[m.Convocation, bool]:
    insc = db.get(m.Inscription, data.inscription_id)
    if insc is None:
        raise ErreurMetier("Inscription introuvable", 404, "INTROUVABLE")
    verifier_portee(db, claims, _etab_de_classe(db, insc.classe_id))
    try:
        with traduire_erreurs_db(db):
            conv, creee = assiduite.generer_convocation(db, data.session_id, data.inscription_id)
            db.commit()
    except assiduite.ErreurAssiduite as e:
        db.rollback()
        raise ErreurMetier(str(e), 404, "INTROUVABLE") from e
    except assiduite.IncoherenceConvocation as e:
        db.rollback()
        raise ErreurMetier("Incohérence d'intégrité détectée : opération annulée", 500, "INTEGRITE_CONVOCATION") from e
    return conv, creee


def generer_convocations_session(db: Session, session_id: uuid.UUID) -> s.GenerationSessionOut:
    try:
        with traduire_erreurs_db(db):
            crees, anomalies = assiduite.generer_convocations_session(db, session_id)
            db.commit()
    except assiduite.ErreurAssiduite as e:
        db.rollback()
        raise ErreurMetier(str(e), 404, "INTROUVABLE") from e
    return s.GenerationSessionOut(convocations_creees=crees, anomalies_integrite=anomalies)


def image_qr_convocation(db: Session, convocation_id: uuid.UUID, claims: security.Claims) -> bytes:
    """Fournit le QR imprimable. Refusé (403) pour toute convocation bloquée : aucun document
    utilisable ne peut être émis pour un élève sous le seuil."""
    conv = db.get(m.Convocation, convocation_id)
    if conv is None:
        raise ErreurMetier("Convocation introuvable", 404, "INTROUVABLE")
    insc = db.get(m.Inscription, conv.inscription_id)
    verifier_portee(db, claims, _etab_de_classe(db, insc.classe_id))
    if conv.bloque:
        raise ErreurMetier(
            f"Convocation bloquée : assiduité {conv.taux_assiduite_fige} % < seuil {conv.seuil_applique} %",
            403, "CONVOCATION_BLOQUEE")
    assiduite.verifier_integrite(db, conv)
    return assiduite.generer_image_qr(assiduite.payload_qr(conv))
