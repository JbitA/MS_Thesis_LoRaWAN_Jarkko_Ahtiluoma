"""
FILE STORY — ``figure_titles.py``
==================================

**Role.** Frozen thesis titles → PNG basenames and companion artifact names.

**Connects.** ``thesis_figure_keys.py``, ``build.py``, ``generate_figure_companion_tables.py``.

**Does not** change trainer hyperparameters.

**Developed with Cursor AI.**
"""

from __future__ import annotations  # Enable postponed evaluation of type hints (PEP 563).



import re  # Regular expressions for sanitizing title strings into safe filenames.

import unicodedata  # Unicode normalization (NFKD) before ASCII-only basename conversion.



# ---------------------------------------------------------------------------

# Canonical display titles (11 primary thesis figures)

# ---------------------------------------------------------------------------

# Thesis selection — 11 figures (ERS325 / active320 mirror set).

# Keys match primary PNG stems (see thesis_figure_keys.py).

FIGURE_MAIN_TITLE: dict[str, str] = {

    "battery_missingness": "Battery missingness",

    "threshold_f2_impute_sensitivity_comparison": "Threshold F2 impute sensitivity comparison",

    "imputed_panel": "Imputed panel",

    "imputed_vs_non_imputed_halt": "Imputed vs non-imputed halt",

    "confusion_grid": "Confusion grid",

    "pr_f2_only": "PR F2 only",

    "policy_map": "Policy map",

    "method_model_ranking": "Method model ranking",

    "regression_4_models": "Regression (4 models)",

    "stl_vs_mtl_delta": "STL vs MTL delta",

    "deep_learning_time_series_integration": "Deep learning time-series integration",

}



# ---------------------------------------------------------------------------

# Optional extra figure PNGs per folder (variant index 1, …)

# ---------------------------------------------------------------------------

# Optional second PNG per folder (variant index 1, …).

FIGURE_EXTRA_TITLE: dict[str, dict[int, str]] = {

    "regression_4_models": {1: "Regression LSTM only"},

}



# ---------------------------------------------------------------------------

# Subplot / lower-panel labels (not used for PNG basenames)

# ---------------------------------------------------------------------------

# Lower-panel / subplot labels (not used for PNG basenames).

FIGURE_SUBPLOT_TITLE: dict[str, tuple[str, ...]] = {

    "threshold_f2_impute_sensitivity_comparison": ("With feature imputation", "Without feature imputation"),

    "imputed_panel": ("Imputed preprocessing", "Non-imputed preprocessing"),

    "imputed_vs_non_imputed_halt": ("Imputed preprocessing", "Non-imputed preprocessing"),

}



# ---------------------------------------------------------------------------

# Companion metric-sheet PNG titles (when not default “{main} metrics”)

# ---------------------------------------------------------------------------

# Companion metric-sheet PNG titles (when different from ``{FIGURE_MAIN_TITLE} metrics``).

FIGURE_TABLE_SHEET_TITLE: dict[str, str] = {

    "deep_learning_time_series_integration": "Deep learning time-series contingency matrix",

}



# ---------------------------------------------------------------------------

# Extra companion table PNG/CSV titles per figure key

# ---------------------------------------------------------------------------

# Extra table PNG/CSV per figure key (e.g. Method B tier-2 ranking beside main figure).

FIGURE_EXTRA_TABLE_TITLE: dict[str, dict[int, str]] = {

    "method_model_ranking": {1: "Method model ranking utility table"},

}





def title_to_png_basename(title: str) -> str:

    """Lowercase filename from the canonical figure title (Unicode F₂ → F2, spaces → underscores)."""

    t = title.replace("\u2082", "2").replace("₂", "2")  # Normalize subscript two to ASCII “2”.

    t = unicodedata.normalize("NFKD", t)  # Decompose Unicode for ASCII transliteration.

    t = t.encode("ascii", "ignore").decode("ascii")  # Drop non-ASCII characters.

    t = t.lower().strip()  # Lowercase and trim outer whitespace.

    t = re.sub(r"[^\w\s-]+", "", t)  # Remove punctuation except word chars, space, hyphen.

    t = re.sub(r"[-\s]+", "_", t)  # Collapse spaces and hyphens to underscores.

    t = re.sub(r"_+", "_", t).strip("_")  # Collapse repeated underscores; trim leading/trailing.

    return f"{t}.png"  # Append standard PNG extension (thesis-frozen basename).





def figure_png_names_for_key(figure_key: str) -> tuple[str, ...]:

    """All figure PNG filenames for a figure key (primary + extras)."""

    if figure_key not in FIGURE_MAIN_TITLE:  # Guard unknown keys early.

        raise KeyError(f"No title mapping for figure key {figure_key!r}")

    names = [title_to_png_basename(FIGURE_MAIN_TITLE[figure_key])]  # Primary figure PNG.

    for idx in sorted(FIGURE_EXTRA_TITLE.get(figure_key, {})):  # Stable order for variant indices.

        names.append(title_to_png_basename(FIGURE_EXTRA_TITLE[figure_key][idx]))  # Extra variant PNG.

    return tuple(names)  # Immutable tuple for callers.





def figure_png_names_for_slug(slug: str) -> tuple[str, ...]:

    """Deprecated alias for :func:`figure_png_names_for_key`."""

    return figure_png_names_for_key(slug)  # Delegate to canonical key-based resolver.





def _stem(png_name: str) -> str:

    return png_name[:-4] if png_name.endswith(".png") else png_name  # Strip “.png” for CSV/MD stems.





def companion_metrics_sheet_title(figure_key: str) -> str:

    return FIGURE_TABLE_SHEET_TITLE.get(figure_key, f"{FIGURE_MAIN_TITLE[figure_key]} metrics")  # Sheet title.





def companion_metrics_sheet_png(figure_key: str) -> str:

    return title_to_png_basename(companion_metrics_sheet_title(figure_key))  # Sheet PNG basename.





def companion_absolute_metrics_sheet_title(figure_key: str) -> str:

    return f"{FIGURE_MAIN_TITLE[figure_key]} absolute metrics"  # STL vs MTL absolute companion title.





def companion_absolute_metrics_sheet_png(figure_key: str) -> str:

    return title_to_png_basename(companion_absolute_metrics_sheet_title(figure_key))  # Absolute sheet PNG.





def companion_extra_table_png(figure_key: str, variant: int) -> str:

    return title_to_png_basename(FIGURE_EXTRA_TABLE_TITLE[figure_key][variant])  # Extra table PNG basename.





def companion_metrics_csv(figure_key: str) -> str:

    return f"{_stem(companion_metrics_sheet_png(figure_key))}.csv"  # Primary metrics CSV filename.





def companion_metrics_md(figure_key: str) -> str:

    return f"{_stem(companion_metrics_sheet_png(figure_key))}.md"  # Primary metrics Markdown filename.





def companion_absolute_metrics_csv(figure_key: str) -> str:

    return f"{_stem(companion_absolute_metrics_sheet_png(figure_key))}.csv"  # Absolute metrics CSV.





def companion_extra_table_csv(figure_key: str, variant: int) -> str:

    return f"{_stem(companion_extra_table_png(figure_key, variant))}.csv"  # Extra table CSV basename.





def companion_artifact_names(figure_key: str) -> tuple[str, ...]:

    """All companion CSV/MD/PNG basenames for a thesis figure key."""

    names = [  # Default companions present for every figure key.

        companion_metrics_csv(figure_key),

        companion_metrics_md(figure_key),

        companion_metrics_sheet_png(figure_key),

    ]

    if figure_key == "stl_vs_mtl_delta":  # This figure also ships absolute MTL/STL companions.

        names.extend(

            [companion_absolute_metrics_csv(figure_key), companion_absolute_metrics_sheet_png(figure_key)]

        )

    for idx in sorted(FIGURE_EXTRA_TABLE_TITLE.get(figure_key, {})):  # Optional extra table variants.

        names.append(companion_extra_table_png(figure_key, idx))

        names.append(companion_extra_table_csv(figure_key, idx))

    return tuple(names)  # Full allowlist of companion artifact basenames.





def all_companion_table_png_basenames() -> frozenset[str]:

    out: set[str] = set()  # Accumulator for every companion table PNG across all figures.

    for figure_key in FIGURE_MAIN_TITLE:  # Walk all primary thesis figure keys.

        out.add(companion_metrics_sheet_png(figure_key))  # Standard metrics sheet PNG.

        if figure_key == "stl_vs_mtl_delta":  # Include absolute metrics sheet when applicable.

            out.add(companion_absolute_metrics_sheet_png(figure_key))

        for idx in FIGURE_EXTRA_TABLE_TITLE.get(figure_key, {}):  # Extra table PNGs (e.g. tier-2 ranking).

            out.add(companion_extra_table_png(figure_key, idx))

    return frozenset(out)  # Immutable set for membership tests during PNG migration.


