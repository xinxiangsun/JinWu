"""Simple parser for XSELECT `xselect.mdb` mission database.

This module provides:
- `load_mdb(path, use_cache=True, cache_path=None)` - parse file into an internal mapping
- `get_keyword(tree, mission, instrument=None, mode=None, keyword=None, default=None)` - lookup with inheritance
- `get_defaults(tree, mission, instrument=None, mode=None)` - merged dict of defaults for a context

Caching: by default an in-memory cache is used keyed by file path and mtime. Optionally a
persistent cache file (pickle) can be used via `cache_path`.

This is a lightweight, dependency-free implementation intended for use by `jinwu.core.xselect`.
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, Tuple, Optional, Any

# In-memory cache: path -> (mtime, parsed_tree)
_MEM_CACHE: Dict[str, Tuple[float, Dict[Tuple[str, ...], Dict[str, str]]]] = {}


def _parse_value(s: str) -> Any:
    s = s.strip()
    if not s or s.upper() == 'NONE':
        return None
    low = s.lower()
    if low in ('yes', 'true', 'on'):
        return True
    if low in ('no', 'false', 'off'):
        return False
    # try int
    try:
        if '.' not in s:
            return int(s)
    except Exception:
        pass
    # try float
    try:
        return float(s)
    except Exception:
        pass
    # strip surrounding quotes
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]
    return s


def _parse_lines(lines):
    """Parse lines of an xselect.mdb file into a mapping.

    Returns: dict mapping context tuple -> dict(keyword -> value)
    Context tuple is like (mission,), (mission,instrument), (mission,instrument,mode), etc.
    """
    tree: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        # comments start with '!'
        if line.startswith('!'):
            continue
        # Some lines may contain leading spaces then a key; use split(None,1)
        parts = line.split(None, 1)
        if not parts:
            continue
        keypart = parts[0]
        valpart = parts[1].strip() if len(parts) > 1 else ''

        # Key may contain colons: Mission:Submission:Detector:Datamode:keyword
        key_elems = keypart.split(':')
        if len(key_elems) < 2:
            # ignore malformed
            continue
        keyword = key_elems[-1]
        context = tuple([k for k in key_elems[:-1] if k])

        # normalize multiple spaces in valpart
        value = _parse_value(valpart)

        tree.setdefault(context, {})[keyword] = value

    return tree


def load_mdb(path: str, use_cache: bool = True, cache_path: Optional[str] = None) -> Dict[Tuple[str, ...], Dict[str, Any]]:
    """Load and parse an xselect.mdb file.

    :param path: path to xselect.mdb
    :param use_cache: if True, use in-memory cache keyed by file mtime
    :param cache_path: optional path to a persistent cache (pickle). If provided and newer than
                       the MDB file, the pickled tree will be loaded instead of reparsing.
    :return: parsed tree mapping context tuples to keyword dicts
    """
    path = os.path.abspath(path)
    mtime = os.path.getmtime(path)

    # try persistent cache first
    if cache_path:
        try:
            if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= mtime:
                with open(cache_path, 'rb') as f:
                    tree = pickle.load(f)
                    return tree
        except Exception:
            # fall through to reparse
            pass

    if use_cache:
        entry = _MEM_CACHE.get(path)
        if entry and entry[0] >= mtime:
            return entry[1]

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    tree = _parse_lines(lines)

    # store in memory cache
    _MEM_CACHE[path] = (mtime, tree)

    # store persistent cache if requested
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(tree, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    return tree


def _contexts_for_lookup(mission: str, instrument: Optional[str], mode: Optional[str]):
    """Return list of context tuples from most specific to least."""
    ctxs = []
    # most specific: mission:instrument:mode
    if instrument and mode:
        ctxs.append((mission, instrument, mode))
    if instrument:
        ctxs.append((mission, instrument))
    ctxs.append((mission,))
    return ctxs


def get_keyword(tree: Dict[Tuple[str, ...], Dict[str, Any]], mission: str, instrument: Optional[str], mode: Optional[str], keyword: str, default: Any = None) -> Any:
    """Lookup a single `keyword` with inheritance: mission:instrument:mode -> mission:instrument -> mission.

    Returns default if not found.
    """
    for ctx in _contexts_for_lookup(mission, instrument, mode):
        d = tree.get(ctx)
        if d and keyword in d:
            return d[keyword]
    return default


def get_defaults(tree: Dict[Tuple[str, ...], Dict[str, Any]], mission: str, instrument: Optional[str] = None, mode: Optional[str] = None) -> Dict[str, Any]:
    """Return a merged dict of keywords for the given context.

    Values from more specific contexts override more general ones.
    """
    out: Dict[str, Any] = {}
    # start with mission-level, then instrument-level, then mode-level
    for ctx in reversed(_contexts_for_lookup(mission, instrument, mode)):
        # reversed ensures mission-level applied first, then instrument, then mode
        d = tree.get(ctx)
        if d:
            out.update(d)
    return out


def infer_adjustgti_timepixr_and_frame(tree: Dict[Tuple[str, ...], Dict[str, Any]], header: Optional[Dict[str, Any]], meta: Optional[Any]) -> Tuple[bool, Optional[float], Optional[float]]:
    """Given a parsed MDB `tree`, and file `header`/`meta`, infer (adjustgti, timepixr, frame_dt).

    - header: FITS header dict (may be None)
    - meta: OgipMeta-like object with attributes `telescop`, `instrume`, `binsize` (optional)
    Returns: (adjustgti_flag, timepixr_or_None, frame_dt_or_None)
    """
    mission = None
    instr = None
    try:
        if meta is not None:
            mission = getattr(meta, 'telescop', None)
            instr = getattr(meta, 'instrume', None)
        if mission is None and header is not None:
            mission = header.get('TELESCOP')
        if instr is None and header is not None:
            instr = header.get('INSTRUME')
    except Exception:
        mission = mission or None
        instr = instr or None

    if mission is None:
        return False, None, None
    mission_u = str(mission).upper()
    instr_u = str(instr).upper() if instr is not None else None

    adj = get_keyword(tree, mission_u, instr_u, None, 'adjustgti', default=False)
    tp = get_keyword(tree, mission_u, instr_u, None, 'timepixr', default=None)

    frame_dt = None
    try:
        if meta is not None and getattr(meta, 'binsize', None) is not None:
            frame_dt = float(getattr(meta, 'binsize'))
        if header is not None:
            for k in ('TIMEDEL', 'TBIN', 'DELTAT', 'BIN_SIZE', 'BINSIZE', 'FRAME_TIME', 'FRAME'):
                if k in header and header[k] not in (None, ''):
                    try:
                        frame_dt = float(header[k])
                        break
                    except Exception:
                        continue
    except Exception:
        frame_dt = None

    return bool(adj), (float(tp) if tp is not None else None), (float(frame_dt) if frame_dt is not None else None)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python xselect_mdb.py /path/to/xselect.mdb')
        raise SystemExit(1)
    p = sys.argv[1]
    tree = load_mdb(p)
    print('Parsed contexts:', len(tree))
    # show some sample keys for a common mission if present
    if ('ASCA',) in tree:
        print('ASCA keys sample:', list(tree[('ASCA',)].items())[:10])
    else:
        some = next(iter(tree.items()))
        print('Example context:', some[0], list(some[1].items())[:10])
