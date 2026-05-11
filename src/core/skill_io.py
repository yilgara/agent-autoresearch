"""How the library reads and writes skill prompt files.

Most teams keep skills as files on disk in some predictable layout
(e.g. `skills/<name>/SKILL.md`). For those, the default
`FilesystemSkillIO` works out of the box — pass a path template to
`__init__` if your layout differs from `skills/<name>/SKILL.md`.

If your skills live somewhere unusual (S3, a DB, a git API),
implement your own `SkillIO` subclass and pass it via the CLI's
`--skill-io` flag.
"""

from __future__ import annotations

import difflib
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path


# ── Sentinel for the orchestrator ───────────────────────────────────────────

UNATTRIBUTED = "(unattributed)"  # commonly emitted by adapters when a session
                                  # couldn't be tied to a specific skill —
                                  # consumers should skip targets with this name


# ── Base class ───────────────────────────────────────────────────────────────

class SkillIO(ABC):
    """Abstract interface for loading and writing skill prompts.

    The library only ever calls `load(name)` to get a skill's current
    content and `write_version(name, new_content, run_id)` to write
    the proposed version — implement those two methods and you're done.
    """

    @abstractmethod
    def load(self, name: str) -> str:
        """Return the current content of the named skill as a string.

        Raise `FileNotFoundError` (or a subclass) if the skill doesn't
        exist. The orchestrator catches that and treats the target as
        unloadable.
        """

    @abstractmethod
    def write_version(
        self,
        name: str,
        new_content: str,
        *,
        run_id: str,
        outputs_root: Path,
    ) -> Path:
        """Write a proposed new version under
        `outputs_root/<run_id>/<name>/`, with `v_old.md`, `v_new.md`,
        and `diff.txt`. Return the absolute path to `v_new.md`.

        Subclasses generally don't need to override this — the default
        helper `_write_version_to_filesystem()` below handles the
        layout and is what `FilesystemSkillIO` calls. Override only if
        you want a different on-disk shape.
        """


def _write_version_to_filesystem(
    name: str,
    *,
    old_content: str,
    new_content: str,
    run_id: str,
    outputs_root: Path,
) -> Path:
    """Shared implementation for the standard outputs/<run>/<skill>/ layout.

    Writes `v_old.md` (snapshot of the original), `v_new.md` (the
    proposed content), and `diff.txt` (unified diff). Returns the
    `v_new.md` path. Existing files are overwritten.
    """
    skill_dir = outputs_root / run_id / name
    skill_dir.mkdir(parents=True, exist_ok=True)

    old_path = skill_dir / "v_old.md"
    new_path = skill_dir / "v_new.md"
    diff_path = skill_dir / "diff.txt"

    old_path.write_text(old_content, encoding="utf-8")
    new_path.write_text(new_content, encoding="utf-8")

    diff_lines = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"{name} (current)",
        tofile=f"{name} (proposed)",
        lineterm="",
    )
    diff_path.write_text("".join(diff_lines), encoding="utf-8")

    return new_path


# ── Filesystem default ──────────────────────────────────────────────────────

class FilesystemSkillIO(SkillIO):
    """Default `SkillIO` for skills stored as files on disk.

    The path template uses `{name}` and optionally `{category}` as
    placeholders. Common layouts:

    - `skills/<name>/SKILL.md` (default — one folder per skill, like
      Anthropic's skill convention)
    - `skills/<category>/<name>.md` (flat with categories)

    If neither template matches your layout, subclass this and
    override `_resolve_path(name)`.
    """

    DEFAULT_TEMPLATE: str = "{root}/{name}/SKILL.md"

    def __init__(
        self,
        root: str | Path = "skills",
        path_template: str | None = None,
    ):
        """
        Args:
            root: base directory for the skill files
            path_template: `str.format()` template with `{root}` +
                `{name}` placeholders. Default: `{root}/{name}/SKILL.md`.
                Templates may also use `{category}` as a recursive
                glob if your skills are nested under category folders.

        Naming: the skill name is used verbatim as `{name}` in the
        template. If your eval pipeline emits a different convention
        than your disk uses (e.g. kebab vs snake), normalize in the
        adapter's `load_targets()` — don't rely on the I/O layer to
        guess.
        """
        self.root = Path(root)
        self.path_template = path_template or self.DEFAULT_TEMPLATE

    def _resolve_path(self, name: str) -> Path:
        """Find the file for `name`. Substitutes `{name}` (and
        `{category}` as a glob, if used) in the template and returns
        the first existing match.
        """
        if "{category}" in self.path_template:
            # Glob-style resolution — search anywhere under root
            glob_target = (
                self.path_template
                .replace("{root}", str(self.root))
                .replace("{category}", "*")
                .replace("{name}", name)
            )
            matches = list(Path(".").glob(glob_target))
            if matches:
                return matches[0]
        else:
            p = Path(self.path_template.format(root=self.root, name=name))
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Skill {name!r} not found under {self.root} "
            f"with template {self.path_template!r}"
        )

    def load(self, name: str) -> str:
        return self._resolve_path(name).read_text(encoding="utf-8")

    def write_version(
        self,
        name: str,
        new_content: str,
        *,
        run_id: str,
        outputs_root: Path,
    ) -> Path:
        old_content = self.load(name)
        return _write_version_to_filesystem(
            name,
            old_content=old_content,
            new_content=new_content,
            run_id=run_id,
            outputs_root=outputs_root,
        )


# ── Convenience helpers used by the orchestrator ────────────────────────────

def make_run_id(prefix: str = "run") -> str:
    """Stable timestamp-based run id — `run_2026-05-04_19-30-12`."""
    return f"{prefix}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


def write_artifact(
    name: str,
    filename: str,
    content: str,
    *,
    run_id: str,
    outputs_root: Path,
) -> Path:
    """Write a free-form artifact (program.md, critic.md, replay.md,
    verdict.md, …) into the same per-target output folder. Caller
    picks the filename — this just handles the path.
    """
    skill_dir = outputs_root / run_id / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    p = skill_dir / filename
    p.write_text(content, encoding="utf-8")
    return p
