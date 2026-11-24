"""Architecture documentation surface.

The /architecture page in the UI is rendered from static, hand-authored
content under ``prebuilt/<section_id>.json`` — one file per entry in
``sections.SECTIONS``. The API (`api/routes/architecture.py`) only reads
those files via ``prebuilt.get_prebuilt`` / ``prebuilt.list_prebuilt``;
nothing here calls an LLM. ``architecture-facts.yaml`` is the curated
source-of-truth reference to read when re-authoring a section.
"""
from openexecutive.architecture.prebuilt import get_prebuilt, list_prebuilt
from openexecutive.architecture.sections import SECTIONS, SectionSpec, get_section

__all__ = [
    "SectionSpec",
    "SECTIONS",
    "get_prebuilt",
    "list_prebuilt",
    "get_section",
]
