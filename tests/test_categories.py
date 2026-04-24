"""
test_categories.py - pipeline.categories keyword tabanli atama logic'i.
"""
from __future__ import annotations

import pytest

from pipeline.categories import (
    CATEGORY_KEYWORDS,
    OTHER_CATEGORY,
    assign_categories,
    primary_category,
)


def test_assigns_other_when_nothing_matches():
    cats = assign_categories("alice/xyz123abc")
    assert cats == [OTHER_CATEGORY]
    assert primary_category(cats) == OTHER_CATEGORY


def test_matches_category_from_project_name_only():
    # 'pytorch' keyword'u proje adinda bagimsiz bir kelime (word boundary ile).
    cats = assign_categories("alice/pytorch-utils")
    assert "AI/ML" in cats
    assert primary_category(cats) == "AI/ML"


def test_matches_multiple_categories():
    # description hem 'django' (Web) hem 'postgres' (Data) iceriyor
    cats = assign_categories(
        "alice/proj",
        description="A Django app with Postgres backend",
    )
    assert "Web"  in cats
    assert "Data" in cats


def test_topics_list_is_used():
    # proje adi belirsiz; topics 'security' iceriyor
    cats = assign_categories(
        "bob/repo", topics=["security", "vulnerability"],
    )
    assert "Security" in cats


def test_hyphen_and_space_variants_match():
    # "machine-learning" hem tireli hem bosluklu varyant
    a = assign_categories("x/y", description="machine-learning library")
    b = assign_categories("x/y", description="machine learning library")
    assert "AI/ML" in a
    assert "AI/ML" in b


def test_case_insensitive():
    cats = assign_categories("x/y", description="A DJANGO API server")
    assert "Web" in cats


def test_word_boundary_prevents_false_match():
    # 'api' keyword'u 'api' oldugunda eslesir; 'rapid' veya 'capital' olmamali.
    assert "Web" not in assign_categories("x/y", description="rapidly evolving")
    assert "Web" not in assign_categories("x/y", description="capital strategy")
    # Beklenen: 'api' bagimsiz kelime olarak gecerse eslesir
    assert "Web" in assign_categories("x/y", description="REST api wrapper")


def test_primary_category_is_first_match_in_keyword_order():
    # CATEGORY_KEYWORDS sirasi: AI/ML, Web, Data, DevOps/CLI, ...
    # hem 'django' (Web) hem 'pytorch' (AI/ML) olsa da AI/ML once geldigi icin
    # primary 'AI/ML'.
    cats = assign_categories(
        "x/y", description="Django service using pytorch models",
    )
    assert cats.index("AI/ML") < cats.index("Web")
    assert primary_category(cats) == "AI/ML"


def test_empty_inputs_return_other():
    assert assign_categories("") == [OTHER_CATEGORY]
    assert assign_categories("", topics=[], description="") == [OTHER_CATEGORY]


def test_all_configured_categories_reachable():
    """CATEGORY_KEYWORDS'te tanimli her kategori en az 1 keyword ile eslesir."""
    for cat, kws in CATEGORY_KEYWORDS.items():
        assert kws, f"{cat} kategorisi bos"
        # Ilk keyword'u prob olarak kullan (ornegin 'machine-learning')
        sample_kw = kws[0]
        cats = assign_categories("x/y", description=sample_kw.replace("-", " "))
        assert cat in cats, f"{cat} kategorisi '{sample_kw}' ile eslesmedi"


def test_non_string_topic_elements_handled():
    # topics parametresi karisik tip dondurebilir — str cast'i uygulanir
    cats = assign_categories("x/y", topics=["django", None, 42])  # type: ignore[list-item]
    assert "Web" in cats


@pytest.mark.parametrize("name,expected", [
    ("apache/airflow",          "Data"),     # 'airflow' Data keyword'u
    ("django/django",           "Web"),      # 'django'
    ("kubernetes/kubernetes",   "DevOps/CLI"),
    ("pytorch/pytorch",         "AI/ML"),
    ("pallets/flask",           "Web"),
])
def test_known_real_projects_primary_category(name: str, expected: str):
    cats = assign_categories(name)
    assert primary_category(cats) == expected, f"{name} -> {cats}"
