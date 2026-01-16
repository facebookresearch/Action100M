from typing import Any, Dict, List, Deque, Optional, Callable
from collections import deque


def nodes_list_to_tree(
    nodes: List[Dict[str, Any]],
    *,
    root_id: Optional[str] = None,
    caption_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
    sort_children: bool = True,
    make_synthetic_root_if_needed: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Convert Action100M-style flat `nodes` (each with node_id/parent_id/start/end/...) into
    a nested dict tree that works with your `print_tree`, `plot_tree`, `format_tree`.

    Returns:
      - a single root dict with optional "children"
      - or None if input is empty / malformed

    Notes:
      - `print_tree`/`plot_tree` expect `start`, `end`, `caption`, and `children`.
      - `format_tree` can also use `captions.video_clip_caption`; we populate both
        `caption` and `captions.video_clip_caption` for convenience.
    """

    if not isinstance(nodes, list) or len(nodes) == 0:
        return None

    # Default caption extraction: prefer PLM caption, then GPT summary (brief), then anything reasonable.
    def _default_caption_fn(n: Dict[str, Any]) -> str:
        # Action100M preview fields seen in your sample
        s = n.get("plm_caption")
        if isinstance(s, str) and s.strip():
            return s.strip()

        gpt = n.get("gpt") or {}
        summ = (gpt.get("summary") or {})
        s = summ.get("brief")
        if isinstance(s, str) and s.strip():
            return s.strip()

        s = n.get("llama3_caption")
        if isinstance(s, str) and s.strip():
            return s.strip()

        # last resort: stringify something stable
        return str(n.get("node_id", "")).strip()

    caption_fn = caption_fn or _default_caption_fn

    # 1) Copy nodes into an index and add required fields.
    by_id: Dict[str, Dict[str, Any]] = {}
    for n in nodes:
        if not isinstance(n, dict):
            continue
        nid = n.get("node_id")
        if not isinstance(nid, str) or not nid:
            continue

        start = float(n.get("start", 0.0))
        end = float(n.get("end", 0.0))

        nn = dict(n)  # shallow copy
        nn["start"] = start
        nn["end"] = end

        cap = caption_fn(nn) or ""
        nn["caption"] = cap
        # also provide the structure expected by `format_tree` (it falls back to `caption` anyway)
        nn["captions"] = {
            "video_clip_caption": cap,
            "mid_frame_caption": None,
        }

        # We'll (re)build children.
        nn.pop("children", None)

        by_id[nid] = nn

    if not by_id:
        return None

    # 2) Attach children.
    roots: List[Dict[str, Any]] = []
    for nid, n in by_id.items():
        pid = n.get("parent_id", None)
        if isinstance(pid, str) and pid in by_id:
            parent = by_id[pid]
            parent.setdefault("children", []).append(n)
        else:
            roots.append(n)

    # Optional: sort children by time (helps plots + readability).
    if sort_children:
        for n in by_id.values():
            if "children" in n:
                n["children"].sort(key=lambda c: (float(c.get("start", 0.0)), float(c.get("end", 0.0))))

    # 3) Select root.
    if root_id is not None and root_id in by_id:
        return by_id[root_id]

    if len(roots) == 1:
        return roots[0]

    # 4) Multiple roots: either pick the earliest, or wrap in a synthetic root.
    roots.sort(key=lambda r: (float(r.get("start", 0.0)), float(r.get("end", 0.0))))

    if not make_synthetic_root_if_needed:
        return roots[0]

    # Make a synthetic root spanning everything.
    min_start = min(float(r.get("start", 0.0)) for r in roots)
    max_end = max(float(r.get("end", 0.0)) for r in roots)
    synth: Dict[str, Any] = {
        "node_id": "__root__",
        "parent_id": None,
        "level": 0,
        "start": min_start,
        "end": max_end,
        "caption": "ROOT",
        "captions": {"video_clip_caption": "ROOT", "mid_frame_caption": None},
        "children": roots,
    }
    return synth

    

def _count_nodes(node: Dict[str, Any], *, min_duration: float) -> int:
    """Return the number of nodes in *node* (inclusive) whose duration ≥ *min_duration*."""
    start, end = node["start"], node["end"]
    if end - start < min_duration:
        return 0
    return 1 + sum(_count_nodes(c, min_duration=min_duration) for c in node.get("children", []))


def prune_tree(node: dict) -> dict | None:
    pruned_children = []
    for child in node.get("children", []):
        kept = prune_tree(child)
        if kept is not None:
            pruned_children.append(kept)

    if pruned_children:
        node["children"] = pruned_children
    else:
        node.pop("children", None)

    if "caption" in node or pruned_children:
        return node
    else:
        return None


def extract_subtrees_bfs(
    tree: Dict[str, Any] | None, *, min_duration: float=0
) -> List[Dict[str, Any]]:
    """
    Breadth-first collection of every sub-tree whose
      • duration ≥ min_duration, and
      • has at least one child (i.e. is not a leaf).
    Gracefully skips null / malformed nodes.
    """
    if not isinstance(tree, dict):                     # root itself might be null
        return []

    subtrees: List[Dict[str, Any]] = []
    q: Deque[Dict[str, Any]] = deque([tree])

    while q:
        node = q.popleft()

        # Skip anything that isn't a proper dict node
        if not isinstance(node, dict):
            continue

        children_raw = node.get("children", [])
        # keep only proper dict children
        children = [c for c in children_raw if isinstance(c, dict)]

        duration = node.get("end", 0) - node.get("start", 0)

        if children and duration >= min_duration:
            subtrees.append(node)

        q.extend(children)

    return subtrees


def format_tree(
    node: Dict[str, Any],
    depth: int = 0,
    max_layers: int = 10,
    idx_path: Optional[List[int]] = None,
) -> str:
    """
    Format a hierarchical temporal-segmentation tree into Markdown.
    Expect keys:
      - start (float), end (float)
      - captions.video_clip_caption (str)  [fallback: node['caption']]
      - captions.mid_frame_caption (Optional[str])
      - children (list[dict], optional)
    """

    # Stop when deeper than requested
    if depth > max_layers:
        return ""

    # Root initializes path; children enumerate from 1
    if idx_path is None:
        idx_path = [1]

    start = float(node.get("start", 0.0))
    end   = float(node.get("end", 0.0))
    dur   = end - start

    # Robust caption extraction
    caps = node.get("captions", {}) or {}
    video_caption = caps.get("video_clip_caption") or node.get("caption") or ""
    mid_caption   = caps.get("mid_frame_caption")
    mid_caption_time = round((start + end) / 2, 2)

    # Normalize captions
    def _clean(s: Optional[str]) -> str:
        return s.strip() if isinstance(s, str) else ""
    video_caption = _clean(video_caption)
    mid_caption   = _clean(mid_caption)

    # Build the heading line
    time_str = f"{start:.2f}s → {end:.2f}s (duration: {dur:.1f}s)"
    if depth == 0:
        title = f"## {time_str}"
    elif 1 <= depth <= 2:
        seg_label = ".".join(map(str, idx_path[1:]))  # 1-based labels (skip root index)
        hashes = "#" * (depth + 2)
        title = f"{hashes} Segment {seg_label} — {time_str}"
    else:
        title = f"**{time_str}**"

    # Compose the caption block (include mid-frame only if present)
    body_lines = [video_caption] if video_caption else []
    if mid_caption:
        mid_caption = '>' + '\n> '.join(mid_caption.splitlines())  # Normalize mid-caption
        body_lines.append(f"*Middle-frame caption at {mid_caption_time}*:\n{mid_caption}\n")
    body = ("\n\n".join(body_lines) + "\n\n") if body_lines else "\n"

    # Insert a horizontal rule before non-first siblings for top 3 levels
    prefix = ""
    if idx_path[-1] != 1 and depth < 3:
        prefix = "---\n\n"

    # Recurse over children (depth-first)
    children_md = ""
    for i, child in enumerate(node.get("children", []) or [], start=1):
        children_md += format_tree(
            child,
            depth=depth + 1,
            max_layers=max_layers,
            idx_path=idx_path + [i],
        )

    return f"{prefix}{title}\n\n{body}{children_md}"