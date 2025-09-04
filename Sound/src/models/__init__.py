from typing import Any, Dict

from .sed_crnn import CRNN


def build_model(cfg: Dict[str, Any]) -> CRNN:
    """Instantiate the model defined by ``cfg``.

    If ``cfg['model']['variant']`` is ``"conformer"``, the configuration
    is adjusted to use a transformer encoder (i.e. a Conformer-style
    architecture) before instantiating :class:`CRNN`.
    """
    variant = cfg.get("model", {}).get("variant", "crnn")
    if variant == "conformer":
        # Configure the CRNN to use the transformer-based encoder
        cfg.setdefault("model", {}).setdefault("rnn", {})["type"] = "transformer"
    return CRNN(cfg)
