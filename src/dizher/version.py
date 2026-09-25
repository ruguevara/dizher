"""The version as uZX shows it: a release build is `0.2.0-alpha`, every other build is dev, `0.2.0-dev+g1a2b3c4`,
`-dirty` after either when the tree had uncommitted changes, with the time it was built. `python -m dizher.version`
freezes this into _build.py for packaging/build.sh (DIZHER_RELEASE=1 for a release); from source, git is asked."""
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from . import CHANNEL, __version__

ROOT = Path(__file__).resolve().parents[2]   # the repository, in a source checkout


@dataclass(frozen=True)
class Build:
    core: str
    channel: str
    commit: str = ''     # short hash, empty outside git
    dirty: bool = False
    built: str = ''      # UTC time of a packaged build, empty when run from source

    @property
    def display(self) -> str:
        suffix = f'+g{self.commit}' if self.channel == 'dev' and self.commit else ''
        return f"{self.core}-{self.channel}{suffix}{'-dirty' if self.dirty else ''}"


def _git(*args) -> str:
    try:
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ''


def from_git(release: bool = False, built: str = '') -> Build:
    commit = _git('rev-parse', '--short', 'HEAD')
    return Build(__version__, CHANNEL if release else 'dev', commit, bool(commit and _git('status', '--porcelain')), built)


@lru_cache(1)
def current() -> Build:
    try:
        from ._build import BUILD
        return BUILD
    except ImportError:
        return from_git()


if __name__ == '__main__':
    build = from_git(os.environ.get('DIZHER_RELEASE') == '1', datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
    Path(__file__).with_name('_build.py').write_text(f'from .version import Build\n\nBUILD = {build!r}\n')
    print(build.display, build.built)
