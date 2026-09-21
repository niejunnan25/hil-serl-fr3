#!/usr/bin/env python3
"""Export, verify, or restore the pinned source patches in vendor/upstream.lock.json.

Uses only the Python standard library and Git. It does not install packages,
start services, or discard existing worktree changes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {".py", ".sh", ".md", ".txt", ".json", ".yaml", ".yml", ".toml",
                   ".cfg", ".ini", ".xml", ".html", ".css", ".js", ".cpp", ".h",
                   ".hpp", ".c", ".cmake", ".env"}
SOURCE_NAMES = {"LICENSE", "COPYING", "Dockerfile", "Makefile", "CMakeLists.txt",
                ".gitignore", ".gitattributes"}

def run(args, *, cwd, env=None, capture=True):
    return subprocess.run(args, cwd=cwd, env=env, check=True,
                          stdout=subprocess.PIPE if capture else None).stdout

def git(repo, *args, env=None):
    return run(["git", *args], cwd=repo, env=env)

def inside(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a repository-relative path: {relative}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes repository: {relative}")
    return resolved

def is_source(name):
    path = Path(name)
    if any(part in {"__pycache__", "build", "dist", ".pytest_cache", ".git"}
           or part.endswith(".egg-info") for part in path.parts):
        return False
    if any(mark in path.name for mark in (".bak", ".orig", ".swp")):
        return False
    return path.suffix in SOURCE_SUFFIXES or path.name in SOURCE_NAMES

def source_patch(root, repo, commit):
    actual = git(repo, "rev-parse", "HEAD").decode().strip()
    if actual != commit:
        raise ValueError(f"{repo}: HEAD {actual} differs from pinned {commit}")
    baseline = set(git(repo, "ls-tree", "-r", "--name-only", "-z", commit).decode().split("\0"))
    tracked = git(repo, "ls-files", "-z").decode().split("\0")
    untracked = git(repo, "ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
    # Staged deletions disappear from ls-files but must still be removed from
    # our baseline index. A staged addition already removed from disk, on the
    # other hand, has no entry in the final source tree.
    paths = sorted(p for p in baseline | set(tracked) | set(untracked)
                   if p and is_source(p)
                   and (p in baseline or (repo / p).exists() or (repo / p).is_symlink()))
    # A separate index captures additions/deletions, including empty files,
    # without changing the index or working files of the live dependency.
    with tempfile.TemporaryDirectory(prefix="upstream-index-", dir=root / ".git") as tmp:
        env = os.environ.copy()
        env["GIT_INDEX_FILE"] = str(Path(tmp) / "index")
        git(repo, "read-tree", commit, env=env)
        for offset in range(0, len(paths), 100):
            git(repo, "add", "-A", "--", *paths[offset:offset+100], env=env)
        return git(repo, "-c", "core.quotePath=false", "diff", "--cached",
                   "--binary", "--full-index", "--no-ext-diff", "--no-renames",
                   commit, "--", env=env)

def read_lock(root):
    path = root / "vendor/upstream.lock.json"
    lock = json.loads(path.read_text())
    if lock.get("version") != 1:
        raise ValueError("Unsupported upstream lock version")
    seen = set()
    for dep in lock["dependencies"]:
        if dep["name"] in seen:
            raise ValueError("Duplicate dependency")
        seen.add(dep["name"])
        inside(root, dep["path"])
        inside(root, dep["patch"])
        if len(dep["commit"]) != 40 or any(c not in "0123456789abcdef" for c in dep["commit"]):
            raise ValueError("Dependency commit must be a full Git SHA")
    return path, lock

def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "export", "restore"])
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="Repository to operate on (defaults to this script's repository)")
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / ".git").is_dir():
        raise ValueError("Run against a normal Git clone with a .git directory")
    lock_path, lock = read_lock(root)
    # Validate every existing patch before starting any restoration.
    expected = {}
    for dep in lock["dependencies"]:
        if args.command != "export":
            patch = inside(root, dep["patch"]).read_bytes()
            if hashlib.sha256(patch).hexdigest() != dep["sha256"]:
                raise ValueError(f"{dep['name']}: patch checksum mismatch")
            expected[dep["name"]] = patch

    if args.command == "export":
        # Prepare all patches first; a missing or mismatched dependency does not
        # replace any checked-in patch.
        prepared = []
        for dep in lock["dependencies"]:
            patch = source_patch(root, inside(root, dep["path"]), dep["commit"])
            prepared.append((dep, patch))
        for dep, patch in prepared:
            atomic_write(inside(root, dep["patch"]), patch)
            dep["sha256"] = hashlib.sha256(patch).hexdigest()
            print(f"{dep['name']}: exported {len(patch)} bytes")
        atomic_write(lock_path, (json.dumps(lock, indent=2) + "\n").encode())
        return

    for dep in lock["dependencies"]:
        repo = inside(root, dep["path"])
        patch = expected[dep["name"]]
        if not repo.exists():
            if args.command != "restore":
                raise ValueError(f"{dep['name']}: dependency missing; run restore")
            repo.parent.mkdir(parents=True, exist_ok=True)
            run(["git", "clone", "--no-checkout", "--", dep["url"], str(repo)],
                cwd=root, capture=False)
            # The exact commit is pinned, so a newer upstream default is never
            # used implicitly.
            if subprocess.run(["git", "cat-file", "-e", dep["commit"] + "^{commit}"],
                              cwd=repo, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode:
                git(repo, "fetch", "origin", dep["commit"])
            git(repo, "checkout", "--detach", dep["commit"])
        actual = source_patch(root, repo, dep["commit"])
        if actual == patch:
            print(f"{dep['name']}: verified")
            continue
        if args.command != "restore" or actual:
            raise ValueError(f"{dep['name']}: source changes differ from saved patch; "
                             "review and export them, or use a separate fresh clone")
        # Refuse even an unrelated untracked source collision instead of
        # overwriting files. git apply --check verifies the entire patch first.
        if patch:
            patch_path = str(inside(root, dep["patch"]))
            git(repo, "apply", "--check", patch_path)
            git(repo, "apply", patch_path)
        if source_patch(root, repo, dep["commit"]) != patch:
            raise ValueError(f"{dep['name']}: restored patch verification failed")
        print(f"{dep['name']}: restored and verified")

if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"upstream management failed: {exc}") from exc
