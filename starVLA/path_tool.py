#!/usr/bin/env python3
"""Single source of truth for StarVLA repo/storage paths.

Only `.starvla.env` contains hardcoded machine-specific paths:
    REPO_ROOT=/abs/path/to/starVLA
    DATA_ROOT=/abs/path/to/storage/root

All repo code should keep path literals relative and delegate resolution here.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict


ENV_FILE_NAME = ".starvla.env"
REPO_PREFIXES = (
    "./",
    "../",
    "playground/",
    "results/",
    "examples/",
    "starVLA/",
    "deployment/",
    "docs/",
    "assets/",
    "cache/",
)
LOCAL_SUFFIXES = {
    ".pt",
    ".safetensors",
    ".yaml",
    ".yml",
    ".json",
    ".jsonl",
    ".parquet",
    ".mp4",
    ".png",
    ".jpg",
    ".jpeg",
    ".bin",
    ".py",
    ".sh",
}


def _search_env(start: Path) -> Path | None:
    start = start.resolve()
    for base in [start, *start.parents]:
        candidate = base / ENV_FILE_NAME
        if candidate.exists():
            return candidate
    return None


def env_file() -> Path:
    explicit = os.environ.get("STARVLA_ENV_FILE")
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"STARVLA_ENV_FILE points to missing file: {candidate}")

    here = _search_env(Path(__file__).resolve())
    if here is not None:
        return here

    cwd = _search_env(Path.cwd())
    if cwd is not None:
        return cwd

    raise FileNotFoundError(f"Could not locate {ENV_FILE_NAME} from {__file__} or {Path.cwd()}")


def load_env() -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for raw_line in env_file().read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError(f"Invalid line in {ENV_FILE_NAME}: {raw_line!r}")
        parsed[key.strip()] = value.strip()

    for required in ("REPO_ROOT", "DATA_ROOT"):
        if required not in parsed or not parsed[required]:
            raise ValueError(f"Missing {required} in {ENV_FILE_NAME}")
    return parsed


def repo_root() -> Path:
    return Path(load_env()["REPO_ROOT"]).expanduser().resolve()


def data_root() -> Path:
    return Path(load_env()["DATA_ROOT"]).expanduser().resolve()


def derived_paths() -> Dict[str, Path]:
    repo = repo_root()
    storage = data_root()
    playground = repo / "playground"
    return {
        "repo_root": repo,
        "data_root": storage,
        "datasets_dir": storage / "data",
        "models_root_dir": storage / "models",
        "models_dir": storage / "models" / "models",
        "checkpoints_dir": storage / "models" / "checkpoints",
        "playground_dir": playground,
        "playground_datasets": playground / "Datasets",
        "playground_pretrained_models": playground / "Pretrained_models",
        "playground_checkpoints": playground / "Checkpoints",
        "results_storage_dir": storage / "starVLA" / "results",
        "results_dir": repo / "results",
    }


def repo_path(relative_path: str | os.PathLike[str]) -> Path:
    rel = Path(relative_path)
    if rel.is_absolute():
        return rel
    return (repo_root() / rel).resolve()


def maybe_resolve_path(path_like: str | os.PathLike[str] | None) -> str | None:
    if path_like is None:
        return None
    value = str(path_like)
    if not value:
        return value

    path = Path(value)
    if path.is_absolute():
        return str(path)

    if value.startswith(REPO_PREFIXES) or path.suffix in LOCAL_SUFFIXES:
        return str(repo_path(value))

    candidate = repo_root() / path
    if candidate.exists():
        return str(candidate.resolve())

    return value


def _replace_path_node(cfg, key: str) -> None:
    try:
        from omegaconf import OmegaConf
    except Exception:
        return

    value = OmegaConf.select(cfg, key, default=None)
    resolved = maybe_resolve_path(value)
    if resolved is not None and resolved != value:
        OmegaConf.update(cfg, key, resolved, force_add=True)


def normalize_cfg_paths(cfg):
    """Resolve repo-relative config fields against REPO_ROOT."""
    try:
        targets = (
            "run_root_dir",
            "config_yaml",
            "framework.qwenvl.base_vlm",
            "framework.wm.base_wm",
            "framework.wm.base_vlm",
            "datasets.vla_data.data_root_dir",
            "datasets.vlm_data.data_root_dir",
            "trainer.pretrained_checkpoint",
            "trainer.resume_from_checkpoint",
        )
        for key in targets:
            _replace_path_node(cfg, key)
    except Exception:
        return cfg
    return cfg


def _safe_remove_existing(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    entries = list(path.iterdir())
    if all(entry.is_symlink() for entry in entries):
        shutil.rmtree(path)
        return
    if not entries:
        path.rmdir()
        return
    raise RuntimeError(f"Refusing to replace non-empty directory: {path}")


def _relative_target(link_parent: Path, target: Path) -> str:
    return os.path.relpath(target, start=link_parent)


def _migrate_directory_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    if not src.exists() or src.is_symlink() or not src.is_dir():
        return
    for entry in src.iterdir():
        target = dst / entry.name
        if target.exists():
            continue
        shutil.move(str(entry), str(target))
    try:
        src.rmdir()
    except OSError:
        pass


def _migrate_model_dirs(models_root: Path, models_dir: Path, checkpoints_dir: Path) -> None:
    models_root.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for entry in list(models_root.iterdir()):
        if entry.name in {"models", "checkpoints"}:
            continue
        target = models_dir / entry.name
        if target.exists():
            continue
        shutil.move(str(entry), str(target))


def setup_links(*, dry_run: bool = False) -> None:
    paths = derived_paths()
    repo = paths["repo_root"]
    playground = paths["playground_dir"]
    playground.mkdir(parents=True, exist_ok=True)
    paths["datasets_dir"].mkdir(parents=True, exist_ok=True)
    paths["models_root_dir"].mkdir(parents=True, exist_ok=True)
    _migrate_model_dirs(paths["models_root_dir"], paths["models_dir"], paths["checkpoints_dir"])
    paths["checkpoints_dir"].mkdir(parents=True, exist_ok=True)
    paths["results_storage_dir"].mkdir(parents=True, exist_ok=True)

    link_pairs = (
        (paths["playground_datasets"], paths["datasets_dir"]),
        (paths["playground_pretrained_models"], paths["models_dir"]),
        (paths["playground_checkpoints"], paths["checkpoints_dir"]),
        (paths["results_dir"], paths["results_storage_dir"]),
    )
    for link_path, target_path in link_pairs:
        rel_target = _relative_target(link_path.parent, target_path)
        repo_rel = link_path.relative_to(repo)
        cmd = f"ln -sfn {rel_target} {repo_rel}"
        if dry_run:
            print(cmd)
            continue
        if repo_rel == Path("results"):
            _migrate_directory_contents(link_path, target_path)
        _safe_remove_existing(link_path)
        link_path.symlink_to(rel_target)
        print(cmd)


def print_value(name: str) -> None:
    paths = derived_paths()
    key = name.lower()
    if key not in paths:
        available = ", ".join(sorted(paths))
        raise KeyError(f"Unknown key {name!r}. Available: {available}")
    print(paths[key])


def print_shell() -> None:
    env = load_env()
    print(f'export REPO_ROOT="{env["REPO_ROOT"]}"')
    print(f'export DATA_ROOT="{env["DATA_ROOT"]}"')


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StarVLA path helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("shell", help="Print shell exports for REPO_ROOT and DATA_ROOT")

    print_parser = subparsers.add_parser("print", help="Print one derived path")
    print_parser.add_argument("name")

    setup_parser = subparsers.add_parser("setup-links", help="Create playground symlinks using relative targets")
    setup_parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.command == "shell":
        print_shell()
        return 0
    if args.command == "print":
        print_value(args.name)
        return 0
    if args.command == "setup-links":
        setup_links(dry_run=args.dry_run)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
