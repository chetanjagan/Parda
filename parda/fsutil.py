"""Folder search that never reads a huge folder to the end.

Kaggle's newer input mounts are slow to list: one folder with 200k photos (CelebA) can take many minutes.
bounded_walk reads at most max_entries names per folder; a bigger folder is reported as truncated and not
descended into. Our datasets never need anything inside such a folder to be *found*.
"""
import os


def scan_dir(path, max_entries=2000):
    """(files, subdirs, truncated). Stops reading after max_entries names."""
    files, dirs, n = [], [], 0
    try:
        with os.scandir(path) as it:
            for e in it:
                n += 1
                if n > max_entries:
                    return files, dirs, True
                try:
                    is_dir = e.is_dir()
                except OSError:
                    is_dir = False
                (dirs if is_dir else files).append(e.name)
    except OSError:
        return [], [], False
    return files, dirs, False


def bounded_walk(root, max_entries=2000, max_depth=8):
    """Like os.walk (top-down, callers may prune `subdirs` in place) but yields (dir, files, subdirs, truncated)
    and never descends into a truncated folder."""
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        files, subdirs, truncated = scan_dir(d, max_entries)
        subdirs.sort()
        yield d, files, subdirs, truncated
        if not truncated and depth < max_depth:
            stack.extend((os.path.join(d, s), depth + 1) for s in reversed(subdirs))


def find_path(root, rel, max_entries=2000, max_depth=8):
    """All paths root/**/rel (rel may contain '/'), without ever listing a huge folder in full."""
    hits = []
    for d, files, subdirs, _ in bounded_walk(root, max_entries, max_depth):
        if os.path.exists(os.path.join(d, rel)):
            hits.append(os.path.join(d, rel))
    return sorted(hits)
