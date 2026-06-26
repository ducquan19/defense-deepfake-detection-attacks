"""Adversarial attack implementations for authorized robustness evaluation."""

from attacks.fgsm import FGSMAttack, fgsm_attack
from attacks.ifgsm import IFGSMAttack, ifgsm_attack
from attacks.pgd import PGDAttack, pgd_attack

# Registry: config name → class, for config-driven instantiation
ATTACK_REGISTRY: dict[str, type] = {
    "fgsm": FGSMAttack,
    "ifgsm": IFGSMAttack,
    "pgd": PGDAttack,
}

__all__ = [
    "FGSMAttack",
    "fgsm_attack",
    "IFGSMAttack",
    "ifgsm_attack",
    "PGDAttack",
    "pgd_attack",
    "ATTACK_REGISTRY",
]
