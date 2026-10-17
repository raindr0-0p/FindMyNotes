"""Unit tests for the presentation helpers (no search, indexing or history logic)."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import findmynotes_ui as ui  # noqa: E402


def _result(text="plain text", filename="notes.pdf", page=3, similarity=0.42):
    return {"filename": filename, "page": page, "similarity": similarity, "text": text}


def test_results_html_escapes_untrusted_text_and_filename():
    nasty = '<script>alert(1)</script> & <b>bold</b> "quoted"'
    out = ui.results_html([_result(text=nasty, filename='a"><img src=x>.pdf')], "anything")
    assert "<script" not in out and "<img" not in out and "<b>bold</b>" not in out
    assert "&lt;script&gt;" in out and "&lt;b&gt;bold&lt;/b&gt;" in out and "&amp;" in out
    assert "a&quot;&gt;&lt;img src=x&gt;.pdf" in out  # filename is escaped too


def test_blank_lines_in_passages_cannot_break_the_markup():
    out = ui.results_html([_result(text="first paragraph\n\n# not a heading\n\n* not a list")], "q")
    assert out.count("<li") == 1 and out.count("</li>") == 1
    assert out.count("<p>") == 3


def test_matching_query_words_are_marked_and_markup_stays_escaped():
    out = ui.results_html([_result(text="Gradient descent <uses> gradients & momentum.")], "gradient descent")
    assert "<mark>Gradient</mark> <mark>descent</mark>" in out
    assert "<mark>gradients</mark>" in out  # simple plural
    assert "&lt;uses&gt;" in out and "&amp; momentum" in out


def test_stopwords_and_short_words_are_not_marked():
    assert ui.query_terms("how does the model of a to be") == ["model"]
    out = ui.results_html([_result(text="the model and the data")], "how does the model")
    assert out.count("<mark>") == 1


def test_no_marks_when_query_has_no_content_words():
    out = ui.results_html([_result(text="the of and")], "the of and")
    assert "<mark>" not in out


def test_query_with_regex_metacharacters_is_safe():
    out = ui.results_html([_result(text="C++ (templates) [x] a.b")], "c++ (templates) [x] a.b ?*")
    assert out.count("<li") == 1  # no exception, no broken pattern


def test_reflow_joins_wrapped_lines_but_keeps_paragraphs_and_bullets():
    text = "Gradient descent moves\nagainst the gradient.\n\nSecond paragraph\nwraps too.\n- item one\n- item two"
    assert ui.reflow_passage(text) == [
        "Gradient descent moves against the gradient.",
        "Second paragraph wraps too.",
        "- item one",
        "- item two",
    ]


def test_reflow_never_returns_empty_for_nonempty_text():
    assert ui.reflow_passage("x") == ["x"]


def test_similarity_shown_with_four_decimals_and_no_percent():
    out = ui.results_html([_result(similarity=0.5)], "q")
    assert "0.5000" in out and "%" not in out


def test_results_are_an_ordered_list_in_rank_order():
    out = ui.results_html([_result(filename="a.pdf"), _result(filename="b.pdf"), _result(filename="c.pdf")], "q")
    assert out.startswith("<ol") and out.endswith("</ol>")
    assert [m for m in re.findall(r'class="fm-sr">Rank </span>(\d+)', out)] == ["1", "2", "3"]


def test_headings_pluralise_and_escape():
    assert "1 result<" in ui.results_head_html(1, "q", None)
    assert "2 results<" in ui.results_head_html(2, "q", "a.pdf")
    assert "&lt;i&gt;" in ui.results_head_html(1, "<i>", "a.pdf")
    head = ui.documents_head_html(["a.pdf"], 1)
    assert "1</b> document indexed" in head and "1</b> passage<" in head


def test_documents_head_truncates_long_lists():
    names = [f"f{i}.pdf" for i in range(9)]
    assert "and 3 more" in ui.documents_head_html(names, 90, max_names=6)


def test_both_themes_define_the_same_tokens_and_differ():
    assert ui.THEMES["light"].keys() == ui.THEMES["dark"].keys()
    assert ui.THEMES["light"]["--fm-bg"] != ui.THEMES["dark"]["--fm-bg"]


def test_css_declares_scheme_and_makes_no_external_requests():
    light, dark = ui.build_css(False), ui.build_css(True)
    assert "color-scheme: light" in light and "color-scheme: dark" in dark
    for css in (light, dark):
        assert "@import" not in css and "http" not in css  # data URIs are base64-encoded


def test_compat_drops_unsupported_keyword_arguments():
    def fn(a, b=1):
        return a, b

    assert ui.compat(fn, 1, b=2, nonsense=3) == (1, 2)
