# evaluation/__init__.py
from .bleu import Bleu
from .rouge import Rouge
from .cider import Cider
from .tokenizer import PTBTokenizer

import warnings

# 尝试可选的 SPICE（依赖 Java 与 pycocoevalcap/spice）
try:
    from .spice import Spice  # noqa: F401
    _SPICE_AVAILABLE = True
except Exception:
    Spice = None  # type: ignore
    _SPICE_AVAILABLE = False


def compute_scores(gts, gen):
    import os
    metrics = [Bleu(), Rouge(), Cider()]  # 默认只跑三项，稳定且快
    if os.getenv("FSGR_USE_METEOR", "0") == "1":
        try:
            from .meteor import Meteor
            metrics.append(Meteor())
        except Exception:
            pass

    all_score, all_scores = {}, {}
    for metric in metrics:
        try:
            score, scores = metric.compute_score(gts, gen)
        except Exception:
            # 避免单个指标挂掉导致整体失败
            score, scores = 0.0, []
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores
    return all_score, all_scores


__all__ = ['Bleu', 'Rouge', 'Cider', 'PTBTokenizer'] + (['Spice'] if _SPICE_AVAILABLE else [])
