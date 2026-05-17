"""
FILE STORY — ``thesis_figure_slugs.py``
======================================

**Role.** Legacy alias module: re-exports ``THESIS_FIGURE_KEYS`` / slug names for old imports.

**Connects.** ``thesis_figure_keys.py``.

**Developed with Cursor AI.**
"""

from __future__ import annotations



from thesis_figure_keys import (  # noqa: F401 — re-export canonical figure-key API.

    EXCLUDED_GRAPH_SLUGS,

    LEGACY_FIGURE_DIR_TO_KEY,

    THESIS_FIGURE_KEYS,

    THESIS_FIGURE_PNG_NAMES,

    THESIS_FIGURE_SLUGS,

    figure_png_path,

    is_thesis_figure_slug,

    iter_thesis_figure_dirs,

    main_title,

    migrate_graph_outputs,

    normalize_figure_key,

    remove_excluded_graph_dirs,

    remove_legacy_companion_artifacts,

    remove_legacy_figure_pngs,

)



# Historical export name: slugs are now figure keys.

normalize_figure_slug = normalize_figure_key  # Alias for legacy CLI/scripts using “slug” terminology.


