"""Safe mermaid rendering for caller- and trace-supplied identifiers.

Node ids and edge labels reach graphforge from ingested third-party traces,
so both the ``GraphSpec`` renderer and the MCP read path share one escaping
implementation rather than each inventing its own.

Mermaid has two very different positions:

- the **identifier** position has no quoting at all, so anything that is not
  ``[A-Za-z0-9_]`` is slugged away (and reserved words are prefixed —
  a node literally named ``end`` aborts the whole flowchart);
- the **quoted** positions (``["..."]`` and ``|"..."|``) have no backslash
  escape, so the characters that would close them become entity codes.

The emitter never uses the bare ``-- text -->`` link form: ``---`` is itself
a complete link terminator, so a label of ``a --- b`` would close the link
early and declare a node ``b`` that never ran. The piped form ``-->|"..."|``
has no such escape hatch.
"""

from __future__ import annotations

import re

MAX_ID_CHARS = 64
MAX_LABEL_CHARS = 80

_UNSAFE = re.compile(r"[^A-Za-z0-9_]")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")

#: Mermaid keywords that cannot serve as node ids.
RESERVED = frozenset(
    {"end", "graph", "subgraph", "class", "classDef", "click", "style", "linkStyle", "default"}
)


def mermaid_id(text: str) -> str:
    """Slugify one node id for mermaid's unquoted identifier position."""
    slug = _UNSAFE.sub("_", str(text))[:MAX_ID_CHARS]
    if not slug or slug[0].isdigit() or slug in RESERVED:
        slug = f"n_{slug}"
    return slug


def mermaid_ids(nodes: list[str]) -> dict[str, str]:
    """Map node text -> unique identifier.

    Slugs collide (``a b`` and ``a-b`` both slug to ``a_b``), and two
    distinct nodes sharing an id would silently merge into one box.
    """
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for node in nodes:
        if node in mapping:
            continue
        base = mermaid_id(node)
        ident, suffix = base, 2
        while ident in used:
            ident = f"{base}_{suffix}"
            suffix += 1
        used.add(ident)
        mapping[node] = ident
    return mapping


def mermaid_label(text: str) -> str:
    """Text for a quoted position: a node ``["..."]`` or a link ``|"..."|``.

    ``#`` is escaped first, or the entities emitted here would themselves be
    re-escapable. Whitespace is collapsed so a truncated label cannot end in
    a dangling run of dashes next to the link syntax.
    """
    label = _WHITESPACE.sub(" ", _CONTROL_CHARS.sub(" ", str(text))).strip()
    label = label[:MAX_LABEL_CHARS]
    return label.replace("#", "#35;").replace('"', "#quot;").replace("|", "#124;")


def edge_line(source: str, target: str, label: str = "") -> str:
    """One flowchart edge, using the injection-proof piped label form."""
    if label:
        return f"    {source} -->|\"{mermaid_label(label)}\"| {target}"
    return f"    {source} --> {target}"
