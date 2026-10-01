"""enums.py - Énumérations partagées (miroir exact des types ENUM PostgreSQL)."""
from enum import Enum


class RoleUtilisateur(str, Enum):
    ADMIN = "ADMIN"
    INSPECTEUR = "INSPECTEUR"
    DIRECTEUR = "DIRECTEUR"
    PROFESSEUR = "PROFESSEUR"
    ELEVE = "ELEVE"


class SecteurEtablissement(str, Enum):
    PUBLIC = "PUBLIC"
    PRIVE = "PRIVE"


class NiveauEnseignement(str, Enum):
    PRESCOLAIRE = "PRESCOLAIRE"
    PRIMAIRE = "PRIMAIRE"
    COLLEGE = "COLLEGE"
    LYCEE = "LYCEE"
    TECHNIQUE = "TECHNIQUE"


class StatutEmploi(str, Enum):
    FONCTIONNAIRE = "FONCTIONNAIRE"
    CONTRACTUEL_ETAT = "CONTRACTUEL_ETAT"
    VACATAIRE = "VACATAIRE"
    ENSEIGNANT_PRIVE = "ENSEIGNANT_PRIVE"


class StatutSeance(str, Enum):
    PLANIFIE = "PLANIFIE"
    DISPENSE = "DISPENSE"
    REMPLACE = "REMPLACE"
    NON_DISPENSE = "NON_DISPENSE"
    ANNULE_OFFICIEL = "ANNULE_OFFICIEL"


class TypePointage(str, Enum):
    ARRIVEE = "ARRIVEE"
    DEPART = "DEPART"


class StatutPresence(str, Enum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"


class StatutAssiduite(str, Enum):
    REGULIER = "REGULIER"
    A_RISQUE = "A_RISQUE"
    INSUFFISANT = "INSUFFISANT"


class StatutRetenue(str, Enum):
    EN_CONTESTATION = "EN_CONTESTATION"   # fenêtre de contestation ouverte
    CONTESTEE = "CONTESTEE"               # en attente de décision humaine
    VALIDEE = "VALIDEE"                   # définitive : exportable vers la paie
    ANNULEE = "ANNULEE"                   # définitive : sans effet sur le salaire


# Séances réellement tenues : seules elles comptent dans l'assiduité des élèves
STATUTS_SEANCE_COMPTES = (StatutSeance.DISPENSE, StatutSeance.REMPLACE)
