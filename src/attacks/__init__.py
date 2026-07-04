"""Adversarial attack implementations for authorized robustness evaluation."""

from attacks.fgsm import FGSMAttack, fgsm_attack
from attacks.ifgsm import IFGSMAttack, ifgsm_attack
from attacks.pgd import PGDAttack, pgd_attack
from attacks.mifgsm import MIFGSMAttack
from attacks.nifgsm import NIFGSMAttack
from attacks.autoattack import AutoAttackWrapper
from attacks.deepfool import DeepFoolAttack
from attacks.cw import CWAttack
from attacks.square import SquareAttackWrapper

# Registry: config name → class, for config-driven instantiation
ATTACK_REGISTRY: dict[str, type] = {
    "fgsm": FGSMAttack,
    "ifgsm": IFGSMAttack,
    "pgd": PGDAttack,
    "mifgsm": MIFGSMAttack,
    "nifgsm": NIFGSMAttack,
    "autoattack": AutoAttackWrapper,
    "deepfool": DeepFoolAttack,
    "cw": CWAttack,
    "square": SquareAttackWrapper,
}

__all__ = [
    "FGSMAttack",
    "fgsm_attack",
    "IFGSMAttack",
    "ifgsm_attack",
    "PGDAttack",
    "pgd_attack",
    "MIFGSMAttack",
    "NIFGSMAttack",
    "AutoAttackWrapper",
    "DeepFoolAttack",
    "CWAttack",
    "SquareAttackWrapper",
    "ATTACK_REGISTRY",
]
