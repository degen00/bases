"""Download trained policies from the project's GitHub releases.

Policies are not committed to git (the 3x3 table alone is 6 MB); each release
carries them as assets named ``policy_<w>x<h>.json.gz``.  ``fetch_policies``
downloads them into the configured policy directory so the GUI and CLI find
them at ``Config.path("policy", size)``.

    python play.py fetch-policies              # latest release, every size
    python play.py fetch-policies --tag v0.4 --size 3
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from .config import Config, load_config

API = "https://api.github.com/repos/{repo}/releases/{which}"
ASSET_RE = re.compile(r"^policy_(\d+)x(\d+)\.json(?:\.gz)?$")
Opener = Callable[[urllib.request.Request], object]


def release_url(repo: str, tag: Optional[str] = None) -> str:
    which = "latest" if tag in (None, "", "latest") else f"tags/{tag}"
    return API.format(repo=repo, which=which)


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": "bases-fetch-policies",
                                                "Accept": "application/vnd.github+json"})


def list_policy_assets(repo: str, tag: Optional[str] = None,
                       opener: Opener = urllib.request.urlopen) -> list[dict]:
    """Policy assets of a release: ``[{name, url, size, width, height, tag}]``."""
    with opener(_request(release_url(repo, tag))) as resp:
        release = json.load(resp)
    assets = []
    for asset in release.get("assets", []):
        m = ASSET_RE.match(asset.get("name", ""))
        if not m:
            continue
        assets.append({"name": asset["name"], "url": asset["browser_download_url"],
                       "size": int(asset.get("size", 0)), "width": int(m.group(1)),
                       "height": int(m.group(2)), "tag": release.get("tag_name", tag or "latest")})
    return sorted(assets, key=lambda a: (a["width"], a["height"]))


def fetch_policies(cfg: Optional[Config] = None, tag: Optional[str] = None,
                   sizes: Optional[list[int]] = None,
                   opener: Opener = urllib.request.urlopen,
                   progress: Optional[Callable[[str], None]] = print) -> list[Path]:
    """Download policy assets into the configured policy paths.

    ``sizes`` restricts to square boards of those sizes.  Returns the paths
    written.  Raises ``urllib.error.URLError``/``HTTPError`` on network or
    API failures and ``FileNotFoundError`` if the release has no policies.
    """
    cfg = cfg or load_config()
    repo = cfg.get("release_repo") or cfg.data.get("release_repo")
    assets = list_policy_assets(repo, tag, opener)
    if sizes:
        assets = [a for a in assets if a["width"] == a["height"] and a["width"] in sizes]
    if not assets:
        raise FileNotFoundError(f"release {tag or 'latest'} of {repo} has no policy assets"
                                + (f" for sizes {sizes}" if sizes else ""))
    written = []
    for asset in assets:
        dest = cfg.path("policy", asset["width"]) if asset["width"] == asset["height"] \
            else cfg.path("policy", asset["width"]).parent / asset["name"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        with opener(_request(asset["url"])) as resp, \
                tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
            shutil.copyfileobj(resp, tmp)
            tmp_path = Path(tmp.name)
        tmp_path.replace(dest)
        dest.chmod(0o644)
        written.append(dest)
        if progress:
            progress(f"{asset['name']} ({asset['size'] / 1e6:.1f} MB, {asset['tag']}) -> {dest}")
    return written
