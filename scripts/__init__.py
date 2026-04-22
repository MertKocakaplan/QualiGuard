"""
scripts/ — Batch / long-running CLI scriptleri.

PLAN §2.1.1: bu dosyalar argparse ile arguman alir, checkpoint yazar ve
terminal'de uzun sureli calisir. `# %%` cell markers YOKTUR.

Calistirma:
    python -m scripts.collect --target 1000
    python -m scripts.train_final --tasks commit,bug,smell
"""
from __future__ import annotations
