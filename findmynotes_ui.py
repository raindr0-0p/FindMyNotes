"""Presentation layer for FindMyNotes.

Design tokens for both themes, the stylesheet, and small HTML builders.
Nothing here touches indexing, search, ranking, CSV export or history logic;
``app.py`` owns all behaviour and calls into this module only to draw.

Design notes
------------
* Identity: ballpoint ink (the interface accent) plus a highlighter (used only
  for the logo and for query words that appear in a passage).
* No external requests of any kind. The font stack uses whatever good sans the
  operating system already has, which suits a local-first tool.
* Styling hooks are ``data-testid`` wrappers, keyed containers (``.st-key-*``)
  and plain elements. Library-internal class names are never relied upon, so a
  Streamlit upgrade degrades styling gracefully instead of breaking layout.
"""

from __future__ import annotations

import base64
import html
import inspect
import re

import streamlit as st

# --------------------------------------------------------------------------- #
# Design tokens
# --------------------------------------------------------------------------- #

FONT_STACK = (
    '"Inter", "Inter Variable", ui-sans-serif, system-ui, -apple-system, '
    '"Segoe UI Variable Text", "Segoe UI", Roboto, "Helvetica Neue", Arial, '
    '"Noto Sans", sans-serif'
)

THEMES: dict[str, dict[str, str]] = {
    # Paper: cool, quiet grey-white with ballpoint-blue ink.
    "light": {
        "--fm-bg": "#F3F5F7",
        "--fm-surface": "#FFFFFF",
        "--fm-sunken": "#ECEFF3",
        "--fm-border": "#DAE0E6",
        "--fm-border-strong": "#BEC7D0",
        "--fm-control-border": "#848F9C",  # >= 3:1 against surfaces (WCAG 1.4.11)
        "--fm-text": "#151C24",
        "--fm-text-2": "#4A5662",
        "--fm-text-3": "#626E7A",
        "--fm-accent": "#27479E",
        "--fm-accent-hover": "#1D3786",
        "--fm-accent-soft": "#E6EBF8",
        "--fm-on-accent": "#FFFFFF",
        "--fm-mark": "#FFE680",
        "--fm-mark-text": "#151C24",
        "--fm-shadow": "0 1px 2px rgba(21, 28, 36, .05), 0 12px 28px -16px rgba(21, 28, 36, .22)",
        "--fm-success-bg": "#E6F4EC",
        "--fm-success-fg": "#17673B",
        "--fm-warning-bg": "#FCF1D6",
        "--fm-warning-fg": "#7A4E00",
        "--fm-error-bg": "#FBE8E6",
        "--fm-error-fg": "#A3231B",
        "--fm-info-bg": "#E6EBF8",
        "--fm-info-fg": "#27479E",
        "--fm-focus": "#27479E",
    },
    # Night desk: blue-black surfaces, softer sky-blue ink. Designed, not inverted.
    "dark": {
        "--fm-bg": "#0E1217",
        "--fm-surface": "#161B22",
        "--fm-sunken": "#0B0F13",
        "--fm-border": "#27303A",
        "--fm-border-strong": "#3A4654",
        "--fm-control-border": "#5F6E80",  # >= 3:1 against surfaces (WCAG 1.4.11)
        "--fm-text": "#E7EBF0",
        "--fm-text-2": "#A9B3BF",
        "--fm-text-3": "#8793A1",
        "--fm-accent": "#8DB8F6",
        "--fm-accent-hover": "#AECCF9",
        "--fm-accent-soft": "#17263A",
        "--fm-on-accent": "#0A1220",
        "--fm-mark": "rgba(255, 206, 46, .30)",
        "--fm-mark-text": "#FFF3C4",
        "--fm-shadow": "0 0 0 1px rgba(255, 255, 255, .02)",
        "--fm-success-bg": "#12271C",
        "--fm-success-fg": "#78D6A3",
        "--fm-warning-bg": "#2A2210",
        "--fm-warning-fg": "#F0C562",
        "--fm-error-bg": "#2E1816",
        "--fm-error-fg": "#FF9A90",
        "--fm-info-bg": "#17263A",
        "--fm-info-fg": "#8DB8F6",
        "--fm-focus": "#8DB8F6",
    },
}

# --------------------------------------------------------------------------- #
# Small compatibility helpers (older Streamlit versions lack some options)
# --------------------------------------------------------------------------- #


def compat(fn, *args, **kwargs):
    """Call ``fn`` and silently drop optional keyword arguments it doesn't support."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return fn(*args, **kwargs)
    accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    usable = {k: v for k, v in kwargs.items() if accepts_any or k in params}
    return fn(*args, **usable)


def html_block(markup: str) -> None:
    """Render trusted, pre-escaped HTML without Markdown processing when possible."""
    if hasattr(st, "html"):
        st.html(markup)
    else:  # pragma: no cover - very old Streamlit
        st.markdown(markup, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Stylesheet
# --------------------------------------------------------------------------- #

_CSS = """
:root, html, body { background: var(--fm-bg); }
html { -webkit-text-size-adjust: 100%; }

/* ---------- Base ---------- */
.stApp, [data-testid="stApp"] {
  background: var(--fm-bg);
  color: var(--fm-text);
  font-size: 15px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}
.stApp,
.stApp :is(p, label, button, input, textarea, li, h1, h2, h3, h4, div, span, summary, a):not([data-testid="stIconMaterial"]):not(.material-icons):not(.material-symbols-rounded) {
  font-family: var(--fm-font) !important;
}
.stApp :is(p, li, label, span, div) { color: inherit; }

/* ---------- Page chrome: keep only the product ---------- */
#MainMenu, footer,
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stAppDeployButton"] {
  display: none !important;
}
[data-testid="stHeader"] {
  background: transparent !important;
  height: 0 !important; min-height: 0 !important; overflow: hidden;
}
[data-testid="stMainBlockContainer"], .stMainBlockContainer, .block-container {
  max-width: 800px !important;
  padding: 40px 24px 112px !important;
}
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"],
.block-container > [data-testid="stVerticalBlock"] { gap: 0 !important; }
/* Streamlit wraps each element; collapse the empty wrapper left by <style>. */
[data-testid="stElementContainer"]:has(> [data-testid="stHtml"] > style:only-child),
[data-testid="stElementContainer"]:has(> .stHtml > style:only-child) { display: none; }

/* ---------- Header ---------- */
.st-key-fm_header { margin: 0 0 28px; }
.st-key-fm_header [data-testid="stHorizontalBlock"] {
  flex-wrap: nowrap !important; align-items: center; gap: 12px;
}
.st-key-fm_header [data-testid="stColumn"]:first-child { flex: 1 1 auto !important; min-width: 0 !important; width: auto !important; }
.st-key-fm_header [data-testid="stColumn"]:last-child  { flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; }
.fm-brand { display: flex; align-items: center; gap: 12px; min-width: 0; }
.fm-logo { flex: none; display: block; width: 30px; height: 30px; border-radius: 8px; }
.fm-ico { display: inline-block; flex: none; width: 14px; height: 14px; background-color: currentColor;
  -webkit-mask: var(--fm-doc-icon) center / contain no-repeat; mask: var(--fm-doc-icon) center / contain no-repeat; }
.fm-brand-text { min-width: 0; }
.fm-wordmark { margin: 0; font-size: 18px; font-weight: 650; letter-spacing: -.015em; line-height: 1.2; color: var(--fm-text); }
.fm-tagline { margin: 2px 0 0; font-size: 13.5px; color: var(--fm-text-2); line-height: 1.4; }

/* ---------- Section headings ---------- */
.fm-section-head { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; flex-wrap: wrap; margin: 0 0 10px; }
.fm-section-title { margin: 0; font-size: 15px; font-weight: 650; letter-spacing: -.005em; color: var(--fm-text); }
.fm-stats { display: flex; gap: 16px; font-size: 13px; color: var(--fm-text-2); font-variant-numeric: tabular-nums; }
.fm-stats b { color: var(--fm-text); font-weight: 650; }
.stApp ul.fm-doclist { margin: -2px 0 0 !important; padding: 0 !important; list-style: none; display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--fm-text-2); }
.stApp ul.fm-doclist li { margin: 0 !important; padding: 0 !important; display: inline-flex; align-items: center; gap: 6px; min-width: 0; max-width: 100%; }
.fm-doclist .fm-docname { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 30ch; }
.fm-doclist svg { flex: none; color: var(--fm-text-3); }
.fm-help { margin: 0 0 12px; font-size: 13.5px; color: var(--fm-text-2); line-height: 1.55; max-width: 62ch; }
.st-key-fm_docs { margin-bottom: 28px; }

/* ---------- Widget labels ---------- */
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {
  color: var(--fm-text-2) !important; font-size: 12.5px !important; font-weight: 550 !important; letter-spacing: 0;
}

/* ---------- Text, number and select controls ----------
   One outer box per control owns the fill and border. Every wrapper inside it is
   made transparent, so versions that nest extra native wrappers still match. */
[data-testid="stTextInputRootElement"] :is([data-baseweb], input), [data-testid="stNumberInputContainer"] :is([data-baseweb], input) {
  background: transparent !important; border-color: transparent !important; box-shadow: none !important;
}
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {
  color: var(--fm-text) !important; caret-color: var(--fm-accent); font-size: 15px !important; border: 0 !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--fm-text-3) !important; opacity: 1; }
[data-testid="stTextInputRootElement"], [data-testid="stNumberInputContainer"],
[data-testid="stSelectbox"] [role="group"], [data-testid="stSelectbox"] [data-baseweb="select"] > div {
  background: var(--fm-sunken) !important; border: 1px solid var(--fm-control-border) !important; border-radius: 10px !important;
  min-height: 44px; box-shadow: none !important; transition: border-color .12s ease, box-shadow .12s ease;
}
[data-testid="stSelectbox"] :is([role="group"], [data-baseweb="select"] > div) * { background: transparent !important; border-color: transparent !important; color: var(--fm-text) !important; box-shadow: none !important; }
[data-testid="stSelectbox"] input[role="combobox"] { border: 0 !important; background: transparent !important; box-shadow: none !important; font-size: 15px !important; }
[data-testid="stTextInputRootElement"]:hover, [data-testid="stNumberInputContainer"]:hover,
[data-testid="stSelectbox"] [role="group"]:hover, [data-testid="stSelectbox"] [data-baseweb="select"] > div:hover { border-color: var(--fm-text-2) !important; }
[data-testid="stTextInputRootElement"]:focus-within, [data-testid="stNumberInputContainer"]:focus-within,
[data-testid="stSelectbox"] [role="group"]:focus-within, [data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
  border-color: var(--fm-focus) !important; box-shadow: 0 0 0 3px color-mix(in srgb, var(--fm-focus) 28%, transparent) !important;
}
/* The focus ring above replaces the generic outline on the inner field. */
.stApp :is([data-testid="stTextInput"], [data-testid="stNumberInput"], [data-testid="stSelectbox"]) input:focus-visible { outline: none !important; }
[data-testid="stSelectbox"] input::selection { background: transparent; color: inherit; }
/* Streamlit's built-in "Press Enter to submit form" hint: Enter still submits. */
[data-testid="InputInstructions"] { display: none !important; }
[data-testid="stSelectbox"] svg, [data-testid="stNumberInput"] svg { fill: var(--fm-text-2) !important; color: var(--fm-text-2) !important; }
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {
  background: transparent !important; color: var(--fm-text-2) !important; border: 0 !important; min-height: 0;
}
[data-testid="stNumberInputStepUp"]:hover, [data-testid="stNumberInputStepDown"]:hover { background: var(--fm-border) !important; }
/* Dropdown list (rendered in a portal outside the app tree). */
[data-testid="stSelectboxVirtualDropdown"], [data-testid="stSelectboxVirtualDropdown"] ul,
ul[role="listbox"], div[role="listbox"] {
  background: var(--fm-surface) !important; color: var(--fm-text) !important;
  border: 1px solid var(--fm-border);
  border-radius: 10px; box-shadow: var(--fm-shadow);
}
[role="listbox"] [role="option"], [data-testid="stSelectboxVirtualDropdown"] li {
  background: transparent !important; color: var(--fm-text) !important;
}
[role="listbox"] [role="option"]:hover, [role="listbox"] [role="option"][aria-selected="true"],
[data-testid="stSelectboxVirtualDropdown"] li:hover {
  background: var(--fm-accent-soft) !important;
}
div[data-baseweb="popover"] > div, [data-testid="stSelectboxVirtualDropdown"] { background: var(--fm-surface) !important; }

/* ---------- Buttons ---------- */
.stApp button[kind], [data-testid^="stBaseButton-"] {
  border-radius: 10px; font-weight: 600; font-size: 14px; min-height: 44px; padding: 0 18px;
  transition: background-color .12s ease, border-color .12s ease, color .12s ease;
}
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
  background: var(--fm-accent) !important; color: var(--fm-on-accent) !important; border: 1px solid var(--fm-accent) !important;
}
[data-testid="stBaseButton-primary"] *, [data-testid="stBaseButton-primaryFormSubmit"] * { color: var(--fm-on-accent) !important; }
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover {
  background: var(--fm-accent-hover) !important; border-color: var(--fm-accent-hover) !important;
}
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondaryFormSubmit"], [data-testid="stBaseButton-secondaryDownload"], .stDownloadButton button {
  background: var(--fm-surface) !important; color: var(--fm-text) !important; border: 1px solid var(--fm-border-strong) !important;
}
[data-testid="stBaseButton-secondary"] *, [data-testid="stBaseButton-secondaryFormSubmit"] *, [data-testid="stBaseButton-secondaryDownload"] * { color: var(--fm-text) !important; }
[data-testid="stBaseButton-secondary"]:hover, [data-testid="stBaseButton-secondaryFormSubmit"]:hover, [data-testid="stBaseButton-secondaryDownload"]:hover {
  background: var(--fm-sunken) !important; border-color: var(--fm-text-3) !important;
}
[data-testid="stBaseButton-tertiary"] { background: transparent !important; border: 1px solid transparent !important; color: var(--fm-text-2) !important; }
[data-testid="stBaseButton-tertiary"] * { color: var(--fm-text-2) !important; }
[data-testid="stBaseButton-tertiary"]:hover { background: var(--fm-sunken) !important; }
[data-testid^="stBaseButton-"]:disabled { opacity: .5; cursor: not-allowed; }
.stApp :is(button, input, textarea, summary, [role="switch"], [role="combobox"]):focus-visible {
  outline: 2px solid var(--fm-focus) !important; outline-offset: 2px;
}

/* ---------- File uploader ---------- */
[data-testid="stFileUploaderDropzone"] {
  background: var(--fm-sunken) !important; border: 1px dashed var(--fm-control-border) !important; border-radius: 12px !important;
  padding: 14px 16px; color: var(--fm-text-2) !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--fm-accent) !important; }
[data-testid="stFileUploaderDropzone"] :is(span, small, p, div, svg) { color: var(--fm-text-2) !important; }
[data-testid="stFileUploaderDropzone"] button { min-height: 36px; background: var(--fm-surface) !important; color: var(--fm-text) !important; border: 1px solid var(--fm-border-strong) !important; }
[data-testid="stFileUploaderDropzone"] button * { color: var(--fm-text) !important; }
[data-testid="stFileChips"] { gap: 8px; }
[data-testid="stFileChip"] { background: var(--fm-surface) !important; border: 1px solid var(--fm-border) !important; border-radius: 10px !important; color: var(--fm-text) !important; }
[data-testid="stFileChip"] > div:first-child { background: var(--fm-accent-soft) !important; color: var(--fm-accent) !important; border-radius: 8px !important; }
[data-testid="stFileChip"] > div:first-child svg { color: var(--fm-accent) !important; fill: var(--fm-accent) !important; }
[data-testid="stFileChipName"] { color: var(--fm-text) !important; font-weight: 550; }
[data-testid="stFileChip"] > div:nth-child(2) > div:not([data-testid]) { color: var(--fm-text-3) !important; }
[data-testid="stFileChipDeleteBtn"] button, [data-testid="stBaseButton-minimal"], [data-testid="stBaseButton-borderlessIcon"] {
  min-height: 28px !important; padding: 2px 6px !important; border: 0 !important; background: transparent !important; color: var(--fm-text-2) !important;
}
[data-testid="stFileChipDeleteBtn"] button:hover, [data-testid="stBaseButton-borderlessIcon"]:hover { background: var(--fm-border) !important; }
[data-testid="stFileChipDeleteBtn"] svg, [data-testid="stBaseButton-borderlessIcon"] svg { color: var(--fm-text-2) !important; fill: var(--fm-text-2) !important; }
[data-testid="stFileUploader"] small { color: var(--fm-text-3) !important; }

/* ---------- Alerts: tinted fill, status glyph + text (never colour alone) ---------- */
[data-testid="stAlert"] { border: 0 !important; margin: 0; background: transparent !important; }
[data-testid="stAlertContainer"] {
  display: flex !important; align-items: flex-start; gap: 10px; padding: 12px 14px !important;
  border: 0 !important; border-radius: 10px !important; font-size: 14px; line-height: 1.5;
  background: var(--fm-info-bg) !important; color: var(--fm-info-fg) !important;
}
[data-testid="stAlertContainer"]::before {
  content: ""; flex: none; width: 18px; height: 18px; margin-top: 1px; background-color: currentColor;
  -webkit-mask: var(--fm-i-info) center / contain no-repeat; mask: var(--fm-i-info) center / contain no-repeat;
}
[data-testid="stAlertContainer"] * { color: inherit !important; font-size: 14px; }
[data-testid="stAlertContainer"] p { margin: 0; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) { background: var(--fm-success-bg) !important; color: var(--fm-success-fg) !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"])::before { -webkit-mask-image: var(--fm-i-success); mask-image: var(--fm-i-success); }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) { background: var(--fm-warning-bg) !important; color: var(--fm-warning-fg) !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"])::before { -webkit-mask-image: var(--fm-i-warning); mask-image: var(--fm-i-warning); }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) { background: var(--fm-error-bg) !important; color: var(--fm-error-fg) !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"])::before { -webkit-mask-image: var(--fm-i-error); mask-image: var(--fm-i-error); }

/* ---------- Expander ---------- */
[data-testid="stExpander"] { border: 1px solid var(--fm-border) !important; border-radius: 12px !important; background: var(--fm-surface) !important; overflow: hidden; }
[data-testid="stExpander"] details, [data-testid="stExpander"] details > div { background: transparent !important; border: 0 !important; }
[data-testid="stExpander"] summary { background: transparent !important; padding: 12px 16px; color: var(--fm-text) !important; font-weight: 600; font-size: 14px; }
[data-testid="stExpander"] summary:hover { background: var(--fm-sunken) !important; }
[data-testid="stExpander"] summary * { color: var(--fm-text) !important; }
[data-testid="stExpander"] summary svg { color: var(--fm-text-2) !important; fill: var(--fm-text-2) !important; }
[data-testid="stExpanderDetails"] { padding: 4px 16px 16px; }

/* ---------- Spinner ---------- */
[data-testid="stSpinner"], [data-testid="stSpinner"] * { color: var(--fm-text-2) !important; font-size: 14px; }

/* ---------- Toggle (theme) ---------- */
[data-testid="stCheckbox"] label { gap: 10px; }
[data-testid="stCheckbox"] [data-testid="stWidgetLabel"], [data-testid="stCheckbox"] [data-testid="stWidgetLabel"] p { color: var(--fm-text-2) !important; font-size: 13px !important; font-weight: 550 !important; }
[data-testid="stCheckbox"] label > div:first-of-type { background-color: var(--fm-border-strong) !important; }
[data-testid="stCheckbox"] label:has(input:checked) > div:first-of-type { background-color: var(--fm-accent) !important; }
[data-testid="stCheckbox"] label:has(input:focus-visible) > div:first-of-type { outline: 2px solid var(--fm-focus); outline-offset: 2px; }

/* ---------- Search panel: the one elevated surface ---------- */
.st-key-fm_search { margin-bottom: 12px; }
.st-key-fm_search_panel {
  background: var(--fm-surface); border: 1px solid var(--fm-border); border-radius: 16px;
  padding: 18px; box-shadow: var(--fm-shadow);
}
.st-key-fm_search_panel [data-testid="stForm"] { border: 0 !important; padding: 0 !important; background: transparent !important; }
.st-key-fm_search_panel [data-testid="stVerticalBlock"] { gap: 14px; }
.st-key-fm_search_panel [data-testid="stHorizontalBlock"] { gap: 12px; }
.st-key-fm_search_panel [data-testid="stTextInput"] input { font-size: 17px !important; padding: 0 14px; min-height: 50px; }
.st-key-fm_search_panel [data-testid="stTextInputRootElement"] { min-height: 52px; }
.st-key-search_button, .st-key-search_button > div, .st-key-search_button [data-testid="stFormSubmitButton"] { width: 100% !important; }
.st-key-search_button button { min-height: 52px; width: 100%; font-size: 15px; padding: 0 22px; }
.fm-options-rule { height: 1px; background: var(--fm-border); margin: 2px 0 0; }

/* ---------- History ---------- */
.st-key-fm_history { margin-bottom: 32px; }
.st-key-fm_history [data-testid="stExpander"] { border-radius: 12px !important; background: transparent !important; }
.st-key-fm_history [data-testid="stExpander"] summary { font-weight: 550; color: var(--fm-text-2) !important; }
.st-key-fm_history [data-testid="stExpander"] summary * { color: var(--fm-text-2) !important; }
.fm-quiet { margin: 0; font-size: 13.5px; color: var(--fm-text-2); line-height: 1.55; }

/* ---------- Results ---------- */
.st-key-fm_results_head { margin-bottom: 0; }
.st-key-fm_results_head [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
.st-key-fm_results_head [data-testid="stColumn"]:first-child { flex: 1 1 260px !important; min-width: 0 !important; width: auto !important; }
.st-key-fm_results_head [data-testid="stColumn"]:last-child { flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; }
.st-key-fm_results_head [data-testid="stDownloadButton"] button { min-height: 38px; font-size: 13.5px; }
.st-key-fm_results_head [data-testid="stHorizontalBlock"] { align-items: center; gap: 12px; }
.fm-results-title { margin: 0; font-size: 15px; font-weight: 650; color: var(--fm-text); }
.fm-results-sub { margin: 2px 0 0; font-size: 13px; color: var(--fm-text-2); overflow-wrap: anywhere; }
.fm-results-sub em { font-style: normal; color: var(--fm-text); font-weight: 550; }
.fm-note { margin: 8px 0 14px; font-size: 13px; color: var(--fm-text-2); }
.fm-note summary { cursor: pointer; display: inline-block; color: var(--fm-text-2); border-radius: 4px; text-decoration: underline dotted; text-underline-offset: 3px; }
.fm-note summary:hover { color: var(--fm-text); }
.stApp .fm-note p { margin: 8px 0 0; max-width: 62ch; font-size: 13px; line-height: 1.6; }
.stApp ol.fm-results { list-style: none; margin: 0 !important; padding: 0 !important; background: var(--fm-surface); border: 1px solid var(--fm-border); border-radius: 14px; }
.stApp ol.fm-results > li.fm-result { margin: 0 !important; display: grid; grid-template-columns: 36px minmax(0, 1fr); gap: 0 12px; padding: 20px 22px 20px 18px; border-top: 1px solid var(--fm-border); }
.stApp ol.fm-results > li.fm-result:first-child { border-top: 0; }
.fm-rank { font-size: 13px; font-weight: 650; color: var(--fm-text-3); font-variant-numeric: tabular-nums; text-align: right; padding-top: 2px; }
.fm-sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
.fm-result-main { min-width: 0; }
.fm-result-head { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 12px; margin-bottom: 8px; }
.fm-doc { display: inline-flex; align-items: center; gap: 7px; min-width: 0; font-size: 14px; font-weight: 650; color: var(--fm-text); }
.fm-doc svg { flex: none; color: var(--fm-text-3); }
.fm-doc-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 34ch; }
.fm-page { font-size: 12.5px; font-weight: 550; color: var(--fm-accent); background: var(--fm-accent-soft); border-radius: 6px; padding: 1px 8px; white-space: nowrap; }
.fm-score { margin-left: auto; font-size: 12.5px; color: var(--fm-text-3); white-space: nowrap; font-variant-numeric: tabular-nums; }
.fm-score b { color: var(--fm-text-2); font-weight: 650; margin-left: 2px; }
.fm-passage { max-width: 68ch; font-size: 15.5px; line-height: 1.68; color: var(--fm-text); overflow-wrap: anywhere; hyphens: manual; }
.fm-passage p { margin: 0 0 .7em; }
.fm-passage p:last-child { margin-bottom: 0; }
.fm-passage mark { background: var(--fm-mark); color: var(--fm-mark-text); border-radius: 3px; padding: 0 2px; margin: 0 -2px; -webkit-box-decoration-break: clone; box-decoration-break: clone; }

/* ---------- Empty states ---------- */
.fm-empty { padding: 20px 0 8px; max-width: 56ch; }
.fm-empty h3 { margin: 0 0 6px; font-size: 16px; font-weight: 650; color: var(--fm-text); letter-spacing: -.005em; }
.fm-empty p { margin: 0; font-size: 14px; color: var(--fm-text-2); line-height: 1.6; }

/* ---------- Small screens ---------- */
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"], .stMainBlockContainer, .block-container { padding: 24px 16px 80px !important; }
  .st-key-fm_search_panel { padding: 14px; border-radius: 14px; }
  .fm-result { grid-template-columns: 1fr; padding: 16px 16px 18px; }
  .fm-rank { text-align: left; padding: 0 0 4px; }
  .fm-rank::before { content: "Result "; font-weight: 500; }
  .fm-score { margin-left: 0; }
  .fm-passage { font-size: 15px; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
"""


def build_css(dark: bool) -> str:
    """Return the full ``<style>`` block for the requested theme."""
    tokens = dict(THEMES["dark" if dark else "light"])
    tokens["--fm-font"] = FONT_STACK
    tokens["--fm-doc-icon"] = f'url("{_DOC_MASK}")'
    for name, mask in _STATUS_MASKS.items():
        tokens[f"--fm-i-{name}"] = f'url("{mask}")'
    declarations = "\n".join(f"  {name}: {value};" for name, value in tokens.items())
    scheme = "dark" if dark else "light"
    return f"<style>\n:root {{\n  color-scheme: {scheme};\n{declarations}\n}}\n{_CSS}\n</style>"


def inject_theme(dark: bool) -> None:
    """Apply the active theme. Called first on every run so reruns never flash.

    ``st.html`` sanitises away <style>, so the stylesheet goes through
    ``st.markdown(..., unsafe_allow_html=True)``. Its content is plain CSS that
    this module generates; no user-supplied text ever reaches it.
    """
    st.markdown(build_css(dark), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Icons, logo, favicon
# --------------------------------------------------------------------------- #

# The document glyph is drawn by CSS (a mask on .fm-ico) because st.html strips
# inline <svg>. See the .fm-ico rule in the stylesheet.
_DOC_ICON = '<span class="fm-ico" aria-hidden="true"></span>'


def _logo_svg(accent: str, line: str) -> str:
    """A page with one highlighted line: the product in one mark."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 30 30">'
        f'<rect width="30" height="30" rx="8" fill="{accent}"/>'
        f'<rect x="7.5" y="8" width="15" height="2.2" rx="1.1" fill="{line}" opacity=".6"/>'
        '<rect x="5.5" y="12.4" width="19" height="5.4" rx="1.8" fill="#FFD84D"/>'
        f'<rect x="8" y="14.2" width="13" height="1.8" rx=".9" fill="{accent}"/>'
        f'<rect x="7.5" y="20.2" width="9" height="2.2" rx="1.1" fill="{line}" opacity=".6"/></svg>'
    )


def _data_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


# One logo per theme, embedded as a data URI: no file path, no network request.
_LOGO_URI = {
    False: _data_uri(_logo_svg("#27479E", "#FFFFFF")),
    True: _data_uri(_logo_svg("#8DB8F6", "#0A1220")),
}


_DOC_MASK = _data_uri(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="none" stroke="#000" '
    'stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M4 1.75h5.2L12.25 4.8v8.7a.75.75 0 0 1-.75.75h-7.5a.75.75 0 0 1-.75-.75v-10.9A.85.85 0 0 1 4 1.75Z"/>'
    '<path d="M9 1.9v3h3.1M5.6 8.4h4.8M5.6 10.9h3.4"/></svg>'
)


def _status_mask(inner: str) -> str:
    return _data_uri(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="none" stroke="#000" '
        f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{inner}</svg>'
    )


# Distinct shapes as well as colours: check-circle, triangle, cross-circle, i-circle.
_STATUS_MASKS = {
    "success": _status_mask('<circle cx="10" cy="10" r="8"/><path d="m6.5 10.2 2.4 2.4 4.6-5"/>'),
    "warning": _status_mask('<path d="M10 2.8 18 16.5H2L10 2.8Z"/><path d="M10 8v3.6M10 14.2v.1"/>'),
    "error": _status_mask('<circle cx="10" cy="10" r="8"/><path d="m7.2 7.2 5.6 5.6m0-5.6-5.6 5.6"/>'),
    "info": _status_mask('<circle cx="10" cy="10" r="8"/><path d="M10 9.2v4.6M10 6.2v.1"/>'),
}


def page_icon() -> str:
    """Favicon as an inline data URI (no file or network dependency)."""
    return _LOGO_URI[False]



# --------------------------------------------------------------------------- #
# HTML builders (every dynamic value is escaped here)
# --------------------------------------------------------------------------- #


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def brand_html(dark: bool = False) -> str:
    return (
        '<div class="fm-brand">'
        f'<img class="fm-logo" src="{_LOGO_URI[bool(dark)]}" width="30" height="30" alt="">'
        '<div class="fm-brand-text"><h1 class="fm-wordmark">FindMyNotes</h1>'
        '<p class="fm-tagline">Search your PDFs by meaning, not just exact words.</p></div></div>'
    )


def _plural(n: int, singular: str, plural: str | None = None) -> str:
    return singular if n == 1 else (plural or singular + "s")


def documents_head_html(indexed_files: list[str] | None, indexed_chunks: int | None, max_names: int = 6) -> str:
    """Section heading, plus counts and file names once something is indexed."""
    if not indexed_files or indexed_chunks is None:
        return '<div class="fm-section-head"><h2 class="fm-section-title">Documents</h2></div>'
    n_docs, n_chunks = len(indexed_files), int(indexed_chunks)
    shown = indexed_files[:max_names]
    items = "".join(
        f'<li>{_DOC_ICON}<span class="fm-docname" title="{esc(name)}">{esc(name)}</span></li>' for name in shown
    )
    if n_docs > len(shown):
        items += f"<li>and {n_docs - len(shown)} more</li>"
    return (
        '<div class="fm-section-head"><h2 class="fm-section-title">Documents</h2>'
        f'<div class="fm-stats"><span><b>{n_docs}</b> {_plural(n_docs, "document")} indexed</span>'
        f'<span><b>{n_chunks}</b> {_plural(n_chunks, "passage")}</span></div></div>'
        f'<ul class="fm-doclist" aria-label="Indexed documents">{items}</ul>'
    )


def upload_help_html() -> str:
    return (
        '<p class="fm-help">Choose one or more PDFs, then select <b>Index documents</b>. '
        "Text is read and indexed locally. Scanned, image-only pages can’t be searched.</p>"
    )


def empty_state_html(kind: str) -> str:
    if kind == "no_documents":
        title = "No documents indexed yet"
        body = "Add PDFs above and select Index documents. Once they’re indexed, you can search them here."
    else:
        title = "Ask in your own words"
        body = (
            "Describe an idea rather than a keyword, for example “why the method needs more data”. "
            "Matching passages appear here, ranked by similarity."
        )
    return f'<div class="fm-empty"><h3>{esc(title)}</h3><p>{esc(body)}</p></div>'


def history_empty_html(limit: int) -> str:
    return (
        f'<p class="fm-quiet">No searches yet. Your last {int(limit)} searches are kept here for this session '
        "so you can run them again.</p>"
    )


def results_head_html(count: int, query: str, source: str | None) -> str:
    q = query if len(query) <= 70 else query[:69].rstrip() + "…"
    scope = esc(source) if source else "All PDFs"
    return (
        f'<h2 class="fm-results-title">{count} {_plural(count, "result")}</h2>'
        f'<p class="fm-results-sub">for <em>“{esc(q)}”</em> in {scope}</p>'
    )


def score_note_html() -> str:
    return (
        '<details class="fm-note"><summary>How results are ranked and scored</summary>'
        "<p>Passages are ordered by similarity to your query, highest first. The similarity score is a "
        "relative measure of how close a passage’s meaning is to your query, so compare scores within one "
        "search. It isn’t a percentage or a probability.</p>"
        "<p>Highlighted words are words from your query that appear in the passage. "
        "A passage can be a good match without containing any of them.</p></details>"
    )


# ---- passage text ---------------------------------------------------------- #

_BULLET = re.compile(r"^\s*(?:[•◦▪‣∙·\-–—*]|\(?\d{1,3}[.)]|\(?[A-Za-z][.)])\s+")

_STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in into is it its
    me my of on or our so than that the their them then there these they this to was we were what when where
    which who why will with would you your about also not""".split()
)


def reflow_passage(text: str) -> list[str]:
    """Split extracted PDF text into readable paragraphs.

    PDF extraction usually breaks lines where the page did, which makes narrow,
    ragged text. Blank lines stay paragraph breaks and bullet or numbered lines
    keep their own line; other single line breaks become spaces. Display only:
    the stored text and the CSV are never altered.
    """
    blocks: list[str] = []
    for para in re.split(r"\n\s*\n", text.strip()):
        current: list[str] = []
        for line in (ln.strip() for ln in para.splitlines()):
            if not line:
                continue
            if current and _BULLET.match(line):
                blocks.append(" ".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            blocks.append(" ".join(current))
    return blocks or [text.strip()]


def query_terms(query: str) -> list[str]:
    """Distinct content words from the query (used only to mark matches)."""
    seen: dict[str, None] = {}
    for token in re.findall(r"\w+", query.lower()):
        if len(token) >= 3 and token not in _STOPWORDS:
            seen.setdefault(token, None)
    return list(seen)


def _highlight_pattern(query: str) -> re.Pattern[str] | None:
    terms = sorted(query_terms(query), key=len, reverse=True)
    if not terms:
        return None
    alternatives = "|".join(re.escape(t) for t in terms)
    return re.compile(rf"(?<!\w)(?:{alternatives})(?:e?s)?(?!\w)", re.IGNORECASE)


def mark_matches(text: str, pattern: re.Pattern[str] | None) -> str:
    """HTML-escape ``text`` and wrap literal query-word matches in <mark>."""
    if pattern is None:
        return esc(text)
    out: list[str] = []
    last = 0
    for match in pattern.finditer(text):
        out.append(esc(text[last : match.start()]))
        out.append(f"<mark>{esc(match.group(0))}</mark>")
        last = match.end()
    out.append(esc(text[last:]))
    return "".join(out)


def results_html(results: list[dict], query: str) -> str:
    """Ranked, semantic ordered list of passages."""
    pattern = _highlight_pattern(query)
    items: list[str] = []
    for rank, result in enumerate(results, start=1):
        paragraphs = "".join(f"<p>{mark_matches(block, pattern)}</p>" for block in reflow_passage(result["text"]))
        name = esc(result["filename"])
        items.append(
            '<li class="fm-result">'
            f'<span class="fm-rank"><span class="fm-sr">Rank </span>{rank}</span>'
            '<div class="fm-result-main"><div class="fm-result-head">'
            f'<span class="fm-doc">{_DOC_ICON}<span class="fm-doc-name" title="{name}">{name}</span></span>'
            f'<span class="fm-page">Page {esc(result["page"])}</span>'
            f'<span class="fm-score">Similarity <b>{result["similarity"]:.4f}</b></span></div>'
            f'<div class="fm-passage">{paragraphs}</div></div></li>'
        )
    return f'<ol class="fm-results" aria-label="Search results, ranked by similarity">{"".join(items)}</ol>'
