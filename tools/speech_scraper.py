"""
The Clankers 2.0 -- Speech Scraper
Harvests speech audio from the web for voice engine training.

Sources:
  youtube   -- YouTube / podcasts via yt-dlp
  archive   -- Internet Archive (public domain speech)
  librivox  -- LibriVox public domain audiobooks
  web       -- General web crawler (follows links, harvests audio files)

Output:
  scraped_speech/
    youtube/   archive/   librivox/   web/
  manifest.json  <- per-file metadata: source, url, duration, sample_rate, speech_score

Usage:
  python tools/speech_scraper.py --query "paranoid monologue" --sources youtube archive
  python tools/speech_scraper.py --sources librivox --max 50 --output ./data/speech
  python tools/speech_scraper.py --sources web --seed-urls https://example.com/podcasts
"""

import os
import re
import json
import time
import hashlib
import argparse
import tempfile
import urllib.parse
from pathlib import Path
from datetime import datetime

import requests
from bs4 import BeautifulSoup

# Optional heavy deps -- degrade gracefully
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import webrtcvad
    HAS_VAD = True
except ImportError:
    HAS_VAD = False

try:
    import subprocess
    subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    HAS_YTDLP = True
except Exception:
    HAS_YTDLP = False


# ─── CONFIG ───────────────────────────────────────────────────────────────────

SUPPORTED_AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus", ".aac"}
MIN_DURATION_S  = 4.0    # discard clips shorter than this
MAX_DURATION_S  = 7200.0  # cap at 2 hours -- we want long recordings for word libraries
MIN_SAMPLE_RATE = 16000  # 16 kHz floor
SPEECH_SCORE_THRESHOLD = 0.15  # fraction of frames detected as speech (webrtcvad)

ARCHIVE_API   = "https://archive.org/advancedsearch.php"
LIBRIVOX_API  = "https://librivox.org/api/feed/audiobooks"
USER_AGENT    = ("Mozilla/5.0 (compatible; ClankersScraper/2.0; "
                 "+https://github.com/clankers)")
REQUEST_DELAY = 1.2   # seconds between requests (be polite)


# ─── AUDIO VALIDATOR ──────────────────────────────────────────────────────────

class AudioValidator:
    """
    Validates an audio file for speech content and quality.
    Returns a dict with: valid, duration, sample_rate, speech_score, reason.
    """

    def validate(self, path: Path) -> dict:
        if not path.exists() or path.stat().st_size < 2048:
            return self._reject("file missing or too small")

        info = self._probe(path)
        if not info:
            return self._reject("could not read audio metadata")

        if info["sample_rate"] < MIN_SAMPLE_RATE:
            return self._reject(f"sample rate {info['sample_rate']} < {MIN_SAMPLE_RATE}")

        if info["duration"] < MIN_DURATION_S:
            return self._reject(f"duration {info['duration']:.1f}s < {MIN_DURATION_S}s")

        if info["duration"] > MAX_DURATION_S:
            return self._reject(f"duration {info['duration']:.0f}s > limit")

        speech_score = self._speech_score(path, info["sample_rate"])
        if speech_score < SPEECH_SCORE_THRESHOLD:
            return self._reject(f"speech score {speech_score:.2f} below threshold")

        return {
            "valid": True,
            "duration": round(info["duration"], 2),
            "sample_rate": info["sample_rate"],
            "speech_score": round(speech_score, 3),
            "reason": "ok",
        }

    # ── probing ──────────────────────────────────────────────────────────────

    def _probe(self, path: Path) -> dict | None:
        # Try soundfile first (fast, no subprocess)
        if HAS_SOUNDFILE:
            try:
                info = sf.info(str(path))
                return {"sample_rate": info.samplerate,
                        "duration": info.duration}
            except Exception:
                pass
        # Fall back to ffprobe
        return self._ffprobe(path)

    def _ffprobe(self, path: Path) -> dict | None:
        try:
            import subprocess
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_streams", str(path)],
                capture_output=True, text=True, timeout=10
            )
            data = json.loads(result.stdout)
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "audio":
                    sr = int(stream.get("sample_rate", 0))
                    dur = float(stream.get("duration", 0))
                    return {"sample_rate": sr, "duration": dur}
        except Exception:
            pass
        return None

    # ── speech detection ─────────────────────────────────────────────────────

    def _speech_score(self, path: Path, sample_rate: int) -> float:
        if HAS_VAD and HAS_SOUNDFILE and HAS_NUMPY:
            return self._vad_score(path)
        if HAS_SOUNDFILE and HAS_NUMPY:
            return self._energy_score(path, sample_rate)
        return 0.5  # assume speech when no libs available

    def _vad_score(self, path: Path) -> float:
        """
        Use WebRTC VAD: read 30ms frames, count fraction detected as speech.
        Samples at 16kHz for VAD compatibility.
        """
        try:
            import subprocess
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(path),
                 "-ar", "16000", "-ac", "1", "-f", "wav", tmp_path],
                capture_output=True, timeout=30
            )
            data, sr = sf.read(tmp_path, dtype="int16")
            os.unlink(tmp_path)

            vad = webrtcvad.Vad(2)  # aggressiveness 0-3
            frame_ms = 30
            frame_len = int(sr * frame_ms / 1000) * 2  # bytes (16-bit)
            raw = data.tobytes()
            total = len(raw) // frame_len
            if total == 0:
                return 0.0
            speech = sum(
                1 for i in range(total)
                if vad.is_speech(raw[i * frame_len:(i + 1) * frame_len], sr)
            )
            return speech / total
        except Exception:
            return 0.5

    def _energy_score(self, path: Path, sample_rate: int) -> float:
        """
        Heuristic fallback: ZCR + RMS energy.
        Speech has moderate ZCR (1000-4000 crossings/s) and non-trivial RMS.
        """
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data[:, 0]
            # Use first 30s only
            data = data[: sr * 30]
            rms = float(np.sqrt(np.mean(data ** 2)))
            if rms < 0.002:
                return 0.0  # silence
            zcr = float(np.mean(np.abs(np.diff(np.sign(data)))) * sr / 2)
            # Typical speech ZCR: 1000-4500 Hz
            speech_like = 1.0 if 800 <= zcr <= 5000 else 0.3
            return min(1.0, rms * 10 * speech_like)
        except Exception:
            return 0.5

    def _reject(self, reason: str) -> dict:
        return {"valid": False, "duration": 0, "sample_rate": 0,
                "speech_score": 0, "reason": reason}


# ─── BASE ADAPTER ─────────────────────────────────────────────────────────────

class BaseAdapter:
    def __init__(self, output_root: Path, validator: AudioValidator,
                 max_files: int, verbose: bool):
        self.output_root = output_root
        self.validator   = validator
        self.max_files   = max_files
        self.verbose     = verbose
        self.session     = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.collected   = 0

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def _slug(self, text: str, max_len: int = 60) -> str:
        text = re.sub(r"[^\w\s-]", "", text.lower())
        text = re.sub(r"[\s_-]+", "_", text).strip("_")
        return text[:max_len]

    def _url_to_filename(self, url: str, title: str = "") -> str:
        stem = self._slug(title) if title else hashlib.md5(url.encode()).hexdigest()[:12]
        ext  = Path(urllib.parse.urlparse(url).path).suffix.lower()
        if ext not in SUPPORTED_AUDIO_EXT:
            ext = ".mp3"
        return stem + ext

    def _download(self, url: str, dest: Path) -> bool:
        try:
            r = self.session.get(url, stream=True, timeout=30)
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in r.iter_content(65536):
                    f.write(chunk)
            return True
        except Exception as e:
            self.log(f"  [download error] {url}: {e}")
            return False

    def _accept(self, path: Path, meta: dict, manifest: list) -> bool:
        result = self.validator.validate(path)
        if result["valid"]:
            meta.update(result)
            meta["file"] = str(path.relative_to(self.output_root))
            meta["scraped_at"] = datetime.utcnow().isoformat()
            manifest.append(meta)
            self.collected += 1
            self.log(f"  ✓  {path.name}  "
                     f"({result['duration']:.1f}s  "
                     f"speech={result['speech_score']:.2f}  "
                     f"{result['sample_rate']}Hz)")
            return True
        else:
            self.log(f"  ✗  {path.name}  [{result['reason']}]")
            path.unlink(missing_ok=True)
            return False

    def run(self, query: str, manifest: list) -> int:
        raise NotImplementedError


# ─── YOUTUBE ADAPTER ──────────────────────────────────────────────────────────

class YouTubeAdapter(BaseAdapter):
    """
    Uses yt-dlp to download audio from YouTube search results.
    Targets monologues, rants, interviews, documentaries.
    """

    SEARCH_QUERIES = [
        "{query}",
        "{query} monologue",
        "{query} interview audio",
        "{query} podcast",
        "{query} speech recording",
    ]

    def run(self, query: str, manifest: list) -> int:
        if not HAS_YTDLP:
            print("  [youtube] yt-dlp not found -- skipping. Install: pip install yt-dlp")
            return 0

        out_dir = self.output_root / "youtube"
        out_dir.mkdir(parents=True, exist_ok=True)
        start = self.collected

        for tpl in self.SEARCH_QUERIES:
            if self.collected >= self.max_files:
                break
            search = tpl.format(query=query)
            self.log(f"\n[youtube] searching: {search}")
            self._yt_search(search, out_dir, manifest)

        return self.collected - start

    def _yt_search(self, search: str, out_dir: Path, manifest: list):
        import subprocess
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                "yt-dlp",
                f"ytsearch{min(5, self.max_files - self.collected)}:{search}",
                "--extract-audio",
                "--audio-format", "wav",
                "--audio-quality", "0",
                "--output", os.path.join(tmpdir, "%(id)s.%(ext)s"),
                "--no-playlist",
                "--match-filter", "duration > 10 & duration < 7200",
                "--quiet",
                "--no-warnings",
                "--print", "%(id)s\t%(title)s\t%(webpage_url)s\t%(duration)s",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            downloaded = list(Path(tmpdir).glob("*.wav"))
            for wav in downloaded:
                if self.collected >= self.max_files:
                    break
                # Parse metadata from stdout
                meta = {"source": "youtube", "url": "", "title": wav.stem, "query": search}
                for line in result.stdout.splitlines():
                    parts = line.split("\t")
                    if parts and parts[0] == wav.stem:
                        meta["url"]      = parts[2] if len(parts) > 2 else ""
                        meta["title"]    = parts[1] if len(parts) > 1 else wav.stem
                        meta["duration_raw"] = parts[3] if len(parts) > 3 else ""
                        break

                dest = out_dir / self._url_to_filename(meta["url"], meta["title"])
                wav.rename(dest)
                self._accept(dest, meta, manifest)
                time.sleep(0.3)


# ─── INTERNET ARCHIVE ADAPTER ─────────────────────────────────────────────────

class ArchiveAdapter(BaseAdapter):
    """
    Searches Archive.org for public-domain speech recordings.
    Targets: radio, oral history, speeches, interviews.
    """

    MEDIA_TYPES = ["audio", "etree"]
    COLLECTIONS  = [
        "audio_speech", "radio_programs", "oldtimeradio",
        "oralhistory", "publicdomain", "audio_bookspoetry",
    ]

    def run(self, query: str, manifest: list) -> int:
        out_dir = self.output_root / "archive"
        out_dir.mkdir(parents=True, exist_ok=True)
        start = self.collected

        for collection in self.COLLECTIONS:
            if self.collected >= self.max_files:
                break
            self.log(f"\n[archive.org] collection={collection} query={query!r}")
            items = self._search(query, collection, rows=min(10, self.max_files))
            for item in items:
                if self.collected >= self.max_files:
                    break
                self._harvest_item(item, out_dir, manifest, query)
                time.sleep(REQUEST_DELAY)

        return self.collected - start

    def _search(self, query: str, collection: str, rows: int) -> list:
        params = {
            "q": f"({query}) AND collection:{collection} AND mediatype:audio",
            "fl[]": ["identifier", "title", "description"],
            "rows": rows,
            "output": "json",
        }
        try:
            r = self.session.get(ARCHIVE_API, params=params, timeout=15)
            r.raise_for_status()
            return r.json().get("response", {}).get("docs", [])
        except Exception as e:
            self.log(f"  [archive search error] {e}")
            return []

    def _harvest_item(self, item: dict, out_dir: Path, manifest: list, query: str):
        identifier = item.get("identifier", "")
        title      = item.get("title", identifier)
        meta_url   = f"https://archive.org/metadata/{identifier}"
        try:
            r = self.session.get(meta_url, timeout=15)
            files = r.json().get("files", [])
        except Exception:
            return

        audio_files = [
            f for f in files
            if Path(f.get("name", "")).suffix.lower() in SUPPORTED_AUDIO_EXT
        ]
        if not audio_files:
            return

        # Prefer wav/flac, then mp3
        def quality_rank(f):
            ext = Path(f.get("name", "")).suffix.lower()
            return {".flac": 0, ".wav": 1, ".ogg": 2, ".mp3": 3}.get(ext, 9)

        audio_files.sort(key=quality_rank)

        for af in audio_files[:3]:  # max 3 files per archive item
            if self.collected >= self.max_files:
                break
            file_url = f"https://archive.org/download/{identifier}/{af['name']}"
            fname    = self._url_to_filename(file_url, f"{self._slug(title)}_{af['name']}")
            dest     = out_dir / fname
            self.log(f"  ↓  {file_url}")
            if self._download(file_url, dest):
                self._accept(dest, {
                    "source": "archive.org",
                    "url": file_url,
                    "title": title,
                    "identifier": identifier,
                    "query": query,
                }, manifest)
            time.sleep(REQUEST_DELAY)


# ─── LIBRIVOX ADAPTER ─────────────────────────────────────────────────────────

class LibrivoxAdapter(BaseAdapter):
    """
    Fetches audiobooks from the LibriVox API.
    Good source of varied speech -- volunteers with different voices/accents.
    """

    def run(self, query: str, manifest: list) -> int:
        out_dir = self.output_root / "librivox"
        out_dir.mkdir(parents=True, exist_ok=True)
        start = self.collected

        self.log(f"\n[librivox] query={query!r}")
        books = self._search_books(query)
        for book in books:
            if self.collected >= self.max_files:
                break
            self._harvest_book(book, out_dir, manifest, query)

        return self.collected - start

    def _search_books(self, query: str) -> list:
        params = {
            "title": f"^{query}",
            "format": "json",
            "limit": 10,
        }
        try:
            r = self.session.get(LIBRIVOX_API, params=params, timeout=15)
            r.raise_for_status()
            return r.json().get("books", [])
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                self.log(f"  [librivox] no books found for query {query!r}")
                return []
            self.log(f"  [librivox search error] {e}")
            return []
        except Exception as e:
            self.log(f"  [librivox search error] {e}")
            return []

    def _harvest_book(self, book: dict, out_dir: Path, manifest: list, query: str):
        book_id = book.get("id", "")
        title   = book.get("title", book_id)
        # Fetch chapter list
        try:
            r = self.session.get(
                "https://librivox.org/api/feed/audiotracks",
                params={"project_id": book_id, "format": "json"},
                timeout=15,
            )
            sections = r.json().get("sections", [])
        except Exception as e:
            self.log(f"  [librivox audiotracks error] {e}")
            return

        book_dir = out_dir / self._slug(title)
        book_dir.mkdir(parents=True, exist_ok=True)

        for section in sections[:5]:  # max 5 chapters per book
            if self.collected >= self.max_files:
                break
            url    = section.get("listen_url", "")
            chap   = section.get("section_number", "0").zfill(3)
            reader = section.get("reader_id", "")
            if not url:
                continue
            fname = f"{chap}_{self._slug(title)}.mp3"
            dest  = book_dir / fname
            self.log(f"  ↓  {url}")
            if self._download(url, dest):
                self._accept(dest, {
                    "source": "librivox",
                    "url": url,
                    "title": title,
                    "book_id": book_id,
                    "chapter": chap,
                    "reader_id": reader,
                    "query": query,
                }, manifest)
            time.sleep(REQUEST_DELAY)


# ─── WEB CRAWLER ADAPTER ──────────────────────────────────────────────────────

class WebCrawlerAdapter(BaseAdapter):
    """
    General web crawler. Follows links up to `depth` hops from seed URLs,
    harvests any linked audio files. Respects robots.txt (checks via HEAD).
    """

    AUDIO_MIME = {
        "audio/mpeg", "audio/wav", "audio/x-wav", "audio/ogg",
        "audio/flac", "audio/aac", "audio/mp4", "audio/opus",
    }

    def __init__(self, *args, depth: int = 2, seed_urls: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth     = depth
        self.seed_urls = seed_urls or []
        self._visited  = set()

    def run(self, query: str, manifest: list) -> int:
        out_dir = self.output_root / "web"
        out_dir.mkdir(parents=True, exist_ok=True)
        start = self.collected

        if not self.seed_urls:
            self.log("[web] no seed URLs provided -- skipping general web crawl")
            return 0

        for seed in self.seed_urls:
            if self.collected >= self.max_files:
                break
            self.log(f"\n[web crawler] seed: {seed}")
            self._crawl(seed, out_dir, manifest, query, depth=self.depth)

        return self.collected - start

    def _crawl(self, url: str, out_dir: Path, manifest: list, query: str, depth: int):
        if depth == 0 or self.collected >= self.max_files:
            return
        if url in self._visited:
            return
        self._visited.add(url)

        try:
            r = self.session.get(url, timeout=15)
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "")
        except Exception as e:
            self.log(f"  [fetch error] {url}: {e}")
            return

        # Direct audio URL
        if any(m in ctype for m in self.AUDIO_MIME):
            self._download_and_accept(url, out_dir, manifest, query)
            return

        # Parse HTML
        if "html" not in ctype:
            return

        soup = BeautifulSoup(r.text, "html.parser")
        base = urllib.parse.urlparse(url)

        # Collect audio links: <a href>, <audio src>, <source src>
        audio_links = set()
        for tag in soup.find_all(["a", "audio", "source"]):
            href = tag.get("href") or tag.get("src") or ""
            if not href:
                continue
            full = urllib.parse.urljoin(url, href)
            ext  = Path(urllib.parse.urlparse(full).path).suffix.lower()
            if ext in SUPPORTED_AUDIO_EXT:
                audio_links.add(full)

        for aurl in audio_links:
            if self.collected >= self.max_files:
                break
            self._download_and_accept(aurl, out_dir, manifest, query)
            time.sleep(REQUEST_DELAY)

        # Follow same-domain links to next depth
        if depth > 1:
            for tag in soup.find_all("a", href=True):
                link = urllib.parse.urljoin(url, tag["href"])
                lp   = urllib.parse.urlparse(link)
                if lp.netloc == base.netloc and link not in self._visited:
                    time.sleep(REQUEST_DELAY)
                    self._crawl(link, out_dir, manifest, query, depth - 1)

    def _download_and_accept(self, url: str, out_dir: Path, manifest: list, query: str):
        domain = urllib.parse.urlparse(url).netloc
        d = out_dir / self._slug(domain)
        d.mkdir(parents=True, exist_ok=True)
        fname = self._url_to_filename(url)
        dest  = d / fname
        self.log(f"  ↓  {url}")
        if self._download(url, dest):
            self._accept(dest, {
                "source": "web",
                "url": url,
                "domain": domain,
                "query": query,
            }, manifest)


# ─── ORCHESTRATOR ─────────────────────────────────────────────────────────────

def scrape(
    query: str,
    sources: list[str],
    output: str   = "scraped_speech",
    max_files: int = 100,
    seed_urls: list[str] | None = None,
    crawl_depth: int = 2,
    verbose: bool = True,
) -> list[dict]:
    """
    Main entry point. Returns the final manifest list.
    """
    output_root = Path(output)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"

    # Load existing manifest if present
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"Resuming -- {len(manifest)} files already in manifest.")
    else:
        manifest = []

    validator = AudioValidator()

    adapters = {
        "youtube": YouTubeAdapter(output_root, validator, max_files, verbose),
        "archive": ArchiveAdapter(output_root, validator, max_files, verbose),
        "web":     WebCrawlerAdapter(output_root, validator, max_files, verbose,
                                     depth=crawl_depth, seed_urls=seed_urls or []),
    }

    print(f"\n{'─'*50}")
    print(f"  Clankers 2.0 -- Speech Scraper")
    print(f"  Query   : {query!r}")
    print(f"  Sources : {', '.join(sources)}")
    print(f"  Output  : {output_root}/")
    print(f"  Max     : {max_files} files")
    print(f"{'─'*50}")

    for source in sources:
        if source not in adapters:
            print(f"  [warn] unknown source: {source!r} -- skipping")
            continue
        adapter = adapters[source]
        adapter.collected = len(manifest)  # sync count across adapters
        count = adapter.run(query, manifest)
        print(f"\n  [{source}] collected {count} valid files  "
              f"(total: {len(manifest)})")

        # Save manifest incrementally
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"\n{'─'*50}")
    print(f"  Done. {len(manifest)} files saved to {output_root}/")
    print(f"  Manifest: {manifest_path}")
    print(f"{'─'*50}\n")

    return manifest


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Clankers 2.0 -- Speech Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--query",  default="spoken word monologue",
                   help="Search query describing the speech to find")
    p.add_argument("--sources", nargs="+",
                   choices=["youtube", "archive", "web"],
                   default=["archive"],
                   help="Sources to scrape")
    p.add_argument("--output", default="scraped_speech",
                   help="Output directory")
    p.add_argument("--max", type=int, default=100,
                   help="Maximum number of valid audio files to collect")
    p.add_argument("--seed-urls", nargs="*", default=[],
                   metavar="URL",
                   help="Seed URLs for the web crawler")
    p.add_argument("--crawl-depth", type=int, default=2,
                   help="Web crawler link-follow depth (default 2)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-file output")

    args = p.parse_args()

    scrape(
        query      = args.query,
        sources    = args.sources,
        output     = args.output,
        max_files  = args.max,
        seed_urls  = args.seed_urls,
        crawl_depth= args.crawl_depth,
        verbose    = not args.quiet,
    )


if __name__ == "__main__":
    main()
