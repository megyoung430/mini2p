"""Assembling the code-map data from extracted records, descriptions and overviews."""

import pytest

from src.code_map.assemble import (
    Overviews,
    build_map,
    coverage,
    definition_entry,
    overviews_from_toml,
    package_of,
    parse_descriptions,
)
from src.code_map.extract import module_record, notebook_record

SOURCE = '''"""Toy analysis."""


def fit(trials, n_starts: int = 5) -> dict:
    """Fit one mouse.

    Args:
        trials (pd.DataFrame): One row per trial.
        seed (int): Passed through ``**kwargs``.

    Returns:
        dict: The fit.
    """
    return {}


def _save(path) -> None:
    pass


def _undescribed(x):
    return x
'''

OVERVIEWS = Overviews(
    intro="The repo.",
    sections=(("src/", "The library."),),
    packages={"src/behavior_analysis": "Pure analyses.", "src/agents": "Agents."},
    modules={},
    notebooks={"scripts/nb.ipynb": "A notebook."},
)


def toy_module():
    return module_record("src/behavior_analysis/toy.py", SOURCE)


def test_inputs_join_signature_types_with_documented_descriptions():
    fit = definition_entry(toy_module().defs[0], "src/behavior_analysis/toy.py", {})
    assert fit["i"] == [
        ["trials", "pd.DataFrame", "One row per trial."],
        ["n_starts", "int", ""],
        ["seed", "int", "Passed through ``**kwargs``."],
    ]
    assert fit["o"] == "dict: The fit." and fit["d"] == "Fit one mouse."


def test_a_written_description_fills_a_missing_docstring_and_is_flagged():
    descriptions = {"src/behavior_analysis/toy.py::_save": "Writes the fit."}
    save = definition_entry(toy_module().defs[1], "src/behavior_analysis/toy.py", descriptions)
    assert save["d"] == "Writes the fit." and save["u"] == 1
    assert save["o"] == "None (acts by side effect)"


def test_coverage_lists_undescribed_definitions_and_stale_descriptions():
    descriptions = {
        "src/behavior_analysis/toy.py::_save": "Writes the fit.",
        "src/gone.py::old": "No longer exists.",
    }
    missing, stale = coverage([toy_module()], descriptions)
    assert missing == ["src/behavior_analysis/toy.py::_undescribed"]
    assert stale == ["src/gone.py::old"]


def test_package_is_the_folder_a_file_is_in_or_the_top_level():
    assert package_of("src/agents/q.py") == "src/agents"
    assert package_of("src/settings.py") == "src (top level)"
    assert package_of("scripts/full_pipelines/run.py") == "scripts/full_pipelines"
    assert package_of("src/behavior_analysis/glm_hmm/fit.py") == "src/behavior_analysis/glm_hmm"


def test_build_map_orders_packages_as_the_overviews_list_them_and_appends_unknown_ones():
    modules = [
        module_record("src/agents/q.py", '"""Q."""\n'),
        module_record("src/new_pkg/x.py", '"""X."""\n'),
        toy_module(),
    ]
    notebook = notebook_record("scripts/nb.ipynb", {"cells": []})
    data = build_map(modules, [notebook], OVERVIEWS, {}, "main @ abc1234")
    assert [p["name"] for p in data["pkgs"]] == [
        "src/behavior_analysis", "src/agents", "src/new_pkg", "scripts (top level)",
    ]  # fmt: skip
    assert data["pkgs"][2]["about"] == ""
    assert data["meta"] == "main @ abc1234"
    assert data["pkgs"][3]["mods"][0]["a"] == "A notebook."


def test_module_summary_prefers_the_written_overview_over_the_docstring():
    overviews = OVERVIEWS._replace(modules={"src/agents/q.py": "Written."})
    data = build_map([module_record("src/agents/q.py", '"""Q."""\n')], [], overviews, {}, "")
    assert data["pkgs"][0]["mods"][0]["a"] == "Written."


def test_descriptions_are_tab_separated_key_and_text_with_comments_allowed():
    text = "# path::qualname<TAB>description\nsrc/a.py::f\tDoes f.\n\nsrc/a.py::g\tDoes g.\n"
    assert parse_descriptions(text) == {"src/a.py::f": "Does f.", "src/a.py::g": "Does g."}


@pytest.mark.parametrize(
    "text, message",
    [
        ("src/a.py::f\tOne.\nsrc/a.py::f\tTwo.\n", "twice"),
        ("src/a.py::f Does f.\n", "line 1"),
        ("src/a.py\tNo qualname.\n", "line 1"),
    ],
)
def test_bad_description_lines_fail_loudly(text, message):
    with pytest.raises(ValueError, match=message):
        parse_descriptions(text)


def test_overviews_from_toml_keeps_order_and_requires_the_intro():
    parsed = {
        "intro": "Hi.",
        "section": [{"name": "src/", "text": "Lib."}],
        "package": {"src/b": "B.", "src/a": "A."},
    }
    ov = overviews_from_toml(parsed)
    assert list(ov.packages) == ["src/b", "src/a"] and ov.sections == (("src/", "Lib."),)
    assert ov.modules == {} and ov.notebooks == {}
    with pytest.raises(ValueError, match="intro"):
        overviews_from_toml({"package": {}})


def test_topics_keep_their_order_and_reach_the_page():
    parsed = {
        "intro": "Hi.",
        "topic": [{"title": "Flow", "text": "<p>a</p>", "svg": "<svg></svg>", "caption": "c"},
                  {"title": "Layout", "text": "<p>b</p>"}],
    }  # fmt: skip
    overviews = overviews_from_toml(parsed)
    data = build_map([], [], overviews, {}, "main @ abc, 2026-11-23")
    assert [t["title"] for t in data["topics"]] == ["Flow", "Layout"]
    assert data["topics"][0]["svg"] == "<svg></svg>"


def test_a_topic_without_a_title_or_with_an_unknown_key_fails_loudly():
    with pytest.raises(ValueError, match="title"):
        overviews_from_toml({"intro": "Hi.", "topic": [{"text": "no title"}]})
    with pytest.raises(ValueError, match="title"):
        overviews_from_toml({"intro": "Hi.", "topic": [{"title": "T", "picture": "x"}]})


def test_groups_list_files_under_a_named_section_in_their_order():
    parsed = {
        "intro": "Hi.",
        "group": [{"name": "Second", "about": "b", "files": ["src/pkg/z.py", "src/pkg/b.py"]},
                  {"name": "First", "about": "a", "files": ["src/pkg/a.py", "notebooks/n.ipynb"]}],
    }  # fmt: skip
    overviews = overviews_from_toml(parsed)
    paths = ("src/pkg/a.py", "src/pkg/b.py", "src/pkg/c.py", "src/pkg/z.py")
    modules = [module_record(p, '"""Doc."""\n') for p in paths]
    notebooks = [notebook_record("notebooks/n.ipynb", {"cells": []})]
    data = build_map(modules, notebooks, overviews, {}, "main @ abc, 2026-11-23")
    assert [(p["name"], [m["p"] for m in p["mods"]]) for p in data["pkgs"]] == [
        ("Second", ["src/pkg/z.py", "src/pkg/b.py"]),
        ("First", ["src/pkg/a.py", "notebooks/n.ipynb"]),
        ("src/pkg", ["src/pkg/c.py"]),
    ]


def test_a_file_in_two_groups_or_a_group_named_like_a_folder_fails_loudly():
    from src.code_map.assemble import stale_groups

    with pytest.raises(ValueError, match="two groups"):
        overviews_from_toml({"intro": "Hi.", "group": [
            {"name": "A", "files": ["x.py"]}, {"name": "B", "files": ["x.py"]}]})  # fmt: skip
    with pytest.raises(ValueError, match="two sections"):
        overviews_from_toml(
            {"intro": "Hi.", "package": {"A": "folder"}, "group": [{"name": "A", "files": []}]}
        )
    overviews = overviews_from_toml(
        {"intro": "Hi.", "group": [{"name": "A", "files": ["gone.py"]}]}
    )
    assert stale_groups([], [], overviews) == ["gone.py"]


def test_a_caption_without_a_schematic_or_a_schematic_path_fails_loudly():
    with pytest.raises(ValueError, match="caption"):
        overviews_from_toml({"intro": "Hi.", "topic": [{"title": "T", "caption": "c"}]})
    with pytest.raises(ValueError, match="markup"):
        overviews_from_toml({"intro": "Hi.", "topic": [{"title": "T", "svg": "flow.svg"}]})
