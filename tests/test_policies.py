"""fetch_policies without a network: the GitHub API and downloads are faked."""
import io
import json
import tempfile
import unittest
from pathlib import Path

from bases.config import load_config
from bases.policies import fetch_policies, list_policy_assets, release_url


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def make_opener(release: dict, files: dict[str, bytes]):
    calls = []

    def opener(request):
        url = request.full_url
        calls.append(url)
        if "api.github.com" in url:
            return FakeResponse(json.dumps(release).encode())
        name = url.rsplit("/", 1)[-1]
        return FakeResponse(files[name])
    return opener, calls


RELEASE = {
    "tag_name": "v0.4",
    "assets": [
        {"name": "policy_3x3.json.gz", "size": 6_162_069,
         "browser_download_url": "https://example.test/dl/policy_3x3.json.gz"},
        {"name": "README.md", "size": 10,
         "browser_download_url": "https://example.test/dl/README.md"},
        {"name": "policy_2x2.json.gz", "size": 9_747,
         "browser_download_url": "https://example.test/dl/policy_2x2.json.gz"},
    ],
}


class PolicyDownloadTests(unittest.TestCase):
    def test_release_url(self):
        self.assertEqual(release_url("o/r"), "https://api.github.com/repos/o/r/releases/latest")
        self.assertEqual(release_url("o/r", "latest"),
                         "https://api.github.com/repos/o/r/releases/latest")
        self.assertEqual(release_url("o/r", "v0.4"),
                         "https://api.github.com/repos/o/r/releases/tags/v0.4")

    def test_lists_only_policy_assets_sorted_by_size(self):
        opener, _ = make_opener(RELEASE, {})
        assets = list_policy_assets("o/r", opener=opener)
        self.assertEqual([a["name"] for a in assets], ["policy_2x2.json.gz", "policy_3x3.json.gz"])
        self.assertEqual(assets[0]["width"], 2)
        self.assertEqual(assets[0]["tag"], "v0.4")

    def test_downloads_into_configured_paths(self):
        cfg = load_config()
        with tempfile.TemporaryDirectory() as tmp:
            cfg.base_dir = Path(tmp)
            files = {"policy_2x2.json.gz": b"two", "policy_3x3.json.gz": b"three"}
            opener, calls = make_opener(RELEASE, files)
            written = fetch_policies(cfg, "v0.4", opener=opener, progress=None)
            self.assertEqual(written, [cfg.path("policy", 2), cfg.path("policy", 3)])
            self.assertEqual(cfg.path("policy", 3).read_bytes(), b"three")
            self.assertEqual(calls[0], release_url("degen00/bases", "v0.4"))
            self.assertEqual(len(calls), 3)
            self.assertEqual(list(Path(tmp, "data", "policy").glob("tmp*")), [],
                             "no temp files left behind")

    def test_size_filter_and_missing(self):
        cfg = load_config()
        with tempfile.TemporaryDirectory() as tmp:
            cfg.base_dir = Path(tmp)
            opener, _ = make_opener(RELEASE, {"policy_3x3.json.gz": b"x"})
            written = fetch_policies(cfg, sizes=[3], opener=opener, progress=None)
            self.assertEqual([p.name for p in written], ["policy_3x3.json.gz"])
            with self.assertRaises(FileNotFoundError):
                fetch_policies(cfg, sizes=[7], opener=opener, progress=None)


if __name__ == "__main__":
    unittest.main()
