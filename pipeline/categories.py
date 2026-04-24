"""
categories.py — proje kategorisi atama (AI/ML, Web, Data, vs.).

V1 `filter_categorize.ipynb` icindeki keyword tabanli siniflandirmanin
yeniden kullanilabilir, test edilebilir bir surumudur. Proje adi +
(opsiyonel) GitHub topic listesi + description uzerinde regex tabanli
esleme yapar.

Her proje **birden fazla** kategoriye atanabilir — "birincil" kategori
CATEGORY_KEYWORDS iterasyon sirasinda ilk eslesendir. Hicbir keyword
eslesmezse tek kategori `"Diger"` doner.

PLAN §4.2 ("agresif filtre kaldirilir") ile uyumlu: burada sadece
kategori atanir, filtrelemeye karar interactive hucrede verilir.
"""
from __future__ import annotations

import re
from typing import Final, Iterable

# ── Kategori -> keyword listesi ───────────────────────────────────
# Iterasyon sirasi "birincil" kategoriyi belirler — bu siralamayi
# degistirmeden onek "Diger" olmaktan kurtulan projeler ayni kategoriye
# dusmeye devam eder.
CATEGORY_KEYWORDS: Final[dict[str, tuple[str, ...]]] = {
    "AI/ML": (
        "machine-learning", "deep-learning", "artificial-intelligence",
        "neural-network", "neural-networks", "ai", "ml", "generative-ai",
        "foundation-model",
        "nlp", "natural-language-processing", "computer-vision",
        "reinforcement-learning", "reinforcement", "supervised-learning",
        "unsupervised-learning", "transfer-learning", "federated-learning",
        "self-supervised", "contrastive-learning",
        "time-series", "anomaly-detection", "object-detection",
        "image-segmentation", "image-classification", "text-classification",
        "sentiment-analysis", "named-entity-recognition", "question-answering",
        "summarization", "translation", "speech-recognition", "text-to-speech",
        "audio", "multimodal",
        "tensorflow", "pytorch", "keras", "jax", "flax", "paddle", "mxnet",
        "scikit-learn", "sklearn", "xgboost", "lightgbm", "catboost",
        "huggingface", "transformers", "diffusers", "accelerate", "peft",
        "langchain", "llamaindex", "llama-index", "openai", "anthropic",
        "llm", "gpt", "bert", "gpt2", "gpt3", "gpt4", "llama", "mistral",
        "stable-diffusion", "diffusion", "vae", "gan", "autoencoder",
        "attention", "transformer", "encoder", "decoder",
        "embedding", "rag", "retrieval", "vector-database", "vector-search",
        "chatbot", "chat", "recommendation", "recommender", "ranking",
        "ocr", "pose-estimation", "depth-estimation", "3d",
        "mlops", "mlflow", "wandb", "experiment-tracking", "model-serving",
        "model-deployment", "inference", "onnx", "triton", "bentoml",
    ),
    "Web": (
        "django", "flask", "fastapi", "aiohttp", "tornado", "starlette",
        "sanic", "bottle", "pyramid", "falcon", "litestar", "quart",
        "web-framework", "web-server", "wsgi", "asgi",
        "rest-api", "restful", "graphql", "grpc", "rpc", "api", "openapi",
        "swagger", "webhook", "microservice", "microservices",
        "http", "https", "websocket", "webrtc", "sse", "mqtt",
        "html", "css", "javascript", "typescript", "react", "vue", "svelte",
        "jinja", "template", "frontend", "backend", "fullstack",
        "scraping", "web-scraping", "crawler", "crawling", "spider",
        "beautifulsoup", "scrapy", "playwright", "selenium", "puppeteer",
        "oauth", "oauth2", "jwt", "saml", "sso",
        "requests", "httpx", "urllib",
        "web", "website", "webapp", "cms", "blog", "e-commerce", "ecommerce",
        "proxy", "reverse-proxy", "load-balancer", "cdn",
    ),
    "Data": (
        "data-science", "data-analysis", "data-analytics", "data-engineering",
        "data-pipeline", "data-processing", "data-wrangling", "data-cleaning",
        "exploratory-data-analysis", "eda", "statistics", "statistical",
        "pandas", "numpy", "scipy", "polars", "dask", "vaex", "modin",
        "spark", "pyspark", "hadoop", "hive", "flink", "kafka", "kinesis",
        "big-data", "distributed", "streaming", "batch-processing",
        "etl", "elt", "airflow", "prefect", "dagster", "luigi", "dbt",
        "data-warehouse", "lakehouse", "delta-lake",
        "database", "sql", "postgresql", "postgres", "mysql", "sqlite",
        "mongodb", "redis", "elasticsearch", "cassandra", "clickhouse",
        "dynamodb", "neo4j", "graph-database", "time-series-database",
        "orm", "sqlalchemy", "alembic",
        "visualization", "matplotlib", "seaborn", "plotly", "bokeh",
        "dash", "streamlit", "gradio", "panel", "altair",
        "jupyter", "notebook", "colab", "jupyterlab",
    ),
    "DevOps/CLI": (
        "cli", "command-line", "command-line-tool", "terminal", "shell",
        "bash", "zsh", "fish", "tui", "curses", "rich", "typer", "click",
        "argparse", "tqdm",
        "docker", "container", "kubernetes", "k8s", "helm", "podman",
        "docker-compose", "compose",
        "terraform", "ansible", "puppet", "chef", "saltstack",
        "infrastructure", "infrastructure-as-code", "iac",
        "ci-cd", "cicd", "github-actions", "gitlab-ci", "jenkins", "circleci",
        "continuous-integration", "continuous-deployment", "pipeline",
        "devops", "sre", "platform-engineering", "gitops", "argocd",
        "monitoring", "observability", "logging", "tracing", "metrics",
        "prometheus", "grafana", "alerting", "opentelemetry",
        "deploy", "deployment", "serverless", "lambda", "cloud",
        "aws", "gcp", "azure",
        "automation", "tool", "utility", "script", "helper",
        "package-manager", "build-tool", "linter", "formatter",
        "git", "version-control",
    ),
    "Security": (
        "security", "cybersecurity", "infosec", "appsec",
        "cryptography", "crypto", "encryption", "decryption", "hashing",
        "tls", "ssl", "certificate", "pki",
        "vulnerability", "exploit", "cve", "poc",
        "penetration-testing", "pentest", "red-team", "blue-team",
        "ctf", "capture-the-flag", "wargame",
        "malware", "ransomware", "virus", "trojan",
        "forensics", "reverse-engineering", "disassembler", "decompiler",
        "binary-analysis", "fuzzing", "fuzzer",
        "firewall", "ids", "ips", "network-security", "packet", "wireshark",
        "nmap", "scanner", "port-scanner",
        "authentication", "authorization", "access-control", "rbac",
        "password", "secrets", "vault", "keystore",
        "privacy", "anonymization", "gdpr", "tor", "vpn",
        "sast", "dast", "code-analysis", "static-analysis", "audit",
    ),
    "Desktop": (
        "gui", "desktop", "desktop-app", "desktop-application",
        "tkinter", "pyqt", "pyqt5", "pyqt6", "pyside", "pyside2", "pyside6",
        "wxpython", "wx", "gtk", "gtk3", "gtk4", "pygobject",
        "kivy", "kivymd", "toga", "dearpygui",
        "cross-platform", "electron", "tauri", "qt",
        "windows", "windows-app", "macos", "mac-app", "linux-desktop",
        "system-tray", "taskbar", "notification",
        "game", "game-engine", "pygame", "arcade", "2d", "opengl",
        "graphics", "rendering",
    ),
    "Mobile": (
        "mobile", "mobile-app", "mobile-development",
        "android", "ios", "iphone", "ipad",
        "react-native", "flutter",
        "beeware", "briefcase",
        "push-notification", "mobile-ui",
    ),
}

OTHER_CATEGORY: Final[str] = "Diger"


def _compile_patterns() -> dict[str, tuple[re.Pattern[str], ...]]:
    """Her kategori icin 'kw' -> compiled regex; tire+bosluk esnek."""
    out: dict[str, tuple[re.Pattern[str], ...]] = {}
    for cat, kws in CATEGORY_KEYWORDS.items():
        compiled = []
        for kw in kws:
            # 'machine-learning' hem 'machine-learning' hem 'machine learning'
            # ile eslesir. \b word boundary numeric-leading keyword'leri
            # (2d, 3d, gpt2) icin de calisir.
            body = re.escape(kw).replace(r"\-", r"[\- ]")
            compiled.append(re.compile(rf"\b{body}\b", re.IGNORECASE))
        out[cat] = tuple(compiled)
    return out


_COMPILED: Final[dict[str, tuple[re.Pattern[str], ...]]] = _compile_patterns()


def assign_categories(
    full_name: str,
    topics: Iterable[str] = (),
    description: str = "",
) -> list[str]:
    """
    Bir projeye kategori(ler) ata.

    Arama metni: ``topics + description + full_name`` (hepsi lowercase).
    Bir kategori icin bir keyword eslesmesi yeterli. Hicbir eslesme yoksa
    donus ``["Diger"]``.

    Args:
        full_name: GitHub "user/repo" formatinda.
        topics: repo topic etiketleri (GitHub meta).
        description: repo kisa aciklamasi.

    Returns:
        Eslesen kategori isimleri (CATEGORY_KEYWORDS sirasinda), yoksa
        ``["Diger"]``.
    """
    topic_blob = " ".join(str(t).lower() for t in topics if t)
    haystack = f"{topic_blob} {description or ''} {full_name or ''}".lower()

    matched: list[str] = []
    for cat, patterns in _COMPILED.items():
        for pat in patterns:
            if pat.search(haystack):
                matched.append(cat)
                break
    return matched if matched else [OTHER_CATEGORY]


def primary_category(categories: list[str]) -> str:
    """assign_categories donusunden birincil kategoriyi al."""
    return categories[0] if categories else OTHER_CATEGORY
