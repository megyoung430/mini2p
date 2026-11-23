"""Reading the repo's sources and the hand-written overview files of the code map."""

import pytest

from src.code_map.files import load_overviews, source_paths


def test_schematics_are_read_from_the_files_topics_name(tmp_path):
    (tmp_path / "flow.svg").write_text("<svg viewBox='0 0 1 1'></svg>\n")
    (tmp_path / "bad.svg").write_text("<p>not svg</p>")
    toml = tmp_path / "overviews.toml"
    toml.write_text('intro = "Hi."\n[[topic]]\ntitle = "Flow"\nsvg = "flow.svg"\n')
    assert load_overviews(toml).topics[0]["svg"] == "<svg viewBox='0 0 1 1'></svg>"
    toml.write_text('intro = "Hi."\n[[topic]]\ntitle = "Flow"\nsvg = "bad.svg"\n')
    with pytest.raises(ValueError, match="svg"):
        load_overviews(toml)
    toml.write_text('intro = "Hi."\n[[topic]]\ntitle = "Flow"\nsvg = "missing.svg"\n')
    with pytest.raises(FileNotFoundError):
        load_overviews(toml)


def test_a_missing_source_folder_is_skipped(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("")
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks" / "n.ipynb").write_text("{}")
    python, notebooks = source_paths(tmp_path)  # no scripts/ folder
    assert [p.name for p in python] == ["a.py"] and [p.name for p in notebooks] == ["n.ipynb"]
