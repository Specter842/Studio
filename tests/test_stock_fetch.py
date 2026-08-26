"""Pexels and Pixabay — the licensed, sanctioned way to source clips.

Run against mock servers returning each service's real response shape, so
authentication, filtering, quality selection, caching and rate-limit handling
are all covered without touching the network or burning a quota.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingest import stock_fetch
from ingest.stock_fetch import (
    CreditLedger,
    MissingApiKey,
    PexelsSource,
    PixabaySource,
    StockClip,
    StockFetcher,
    queries_from_brief,
)
from mock_http import MockServer, Request, Response, routed


def pexels_payload(base_url: str) -> dict:
    return {
        "videos": [
            {
                "id": 101, "width": 3840, "height": 2160, "duration": 12,
                "url": "https://www.pexels.com/video/101/",
                "user": {"name": "Alice", "url": "https://www.pexels.com/@alice"},
                "video_files": [
                    {"file_type": "video/mp4", "width": 640, "height": 360,
                     "link": f"{base_url}/dl/p101_sd.mp4"},
                    {"file_type": "video/mp4", "width": 1920, "height": 1080,
                     "link": f"{base_url}/dl/p101_hd.mp4"},
                    {"file_type": "video/mp4", "width": 3840, "height": 2160,
                     "link": f"{base_url}/dl/p101_4k.mp4"},
                ],
            },
            {   # Too short: filtered out by min_duration.
                "id": 102, "width": 1920, "height": 1080, "duration": 1,
                "url": "https://www.pexels.com/video/102/",
                "user": {"name": "Bob", "url": ""},
                "video_files": [
                    {"file_type": "video/mp4", "width": 1920, "height": 1080,
                     "link": f"{base_url}/dl/p102.mp4"},
                ],
            },
        ]
    }


def pixabay_payload(base_url: str) -> dict:
    return {
        "hits": [
            {
                "id": 201, "pageURL": "https://pixabay.com/videos/201/",
                "duration": 9, "user": "Carol", "user_id": 42,
                "videos": {
                    "large": {"url": f"{base_url}/dl/x201_large.mp4",
                              "width": 1920, "height": 1080},
                    "small": {"url": f"{base_url}/dl/x201_small.mp4",
                              "width": 640, "height": 360},
                },
            }
        ]
    }


@pytest.fixture
def stock_server(tiny_video_bytes: bytes, monkeypatch):
    """A server that answers both APIs and serves the downloads."""
    holder: dict = {}

    def search_pexels(request: Request) -> Response:
        if request.headers.get("authorization") != "pexels-test-key":
            return Response.json({"error": "unauthorised"}, status=401)
        return Response.json(pexels_payload(holder["url"]))

    def search_pixabay(request: Request) -> Response:
        if request.param("key") != "pixabay-test-key":
            return Response.json({"error": "unauthorised"}, status=401)
        return Response.json(pixabay_payload(holder["url"]))

    def download(request: Request) -> Response:
        return Response.binary(tiny_video_bytes)

    handler = routed(
        {"/pexels/search": search_pexels, "/pixabay/search": search_pixabay},
        fallback=None,
    )

    def dispatch(request: Request) -> Response:
        if request.path.startswith("/dl/"):
            return download(request)
        return handler(request)

    with MockServer(dispatch) as server:
        holder["url"] = server.url
        monkeypatch.setattr(
            stock_fetch, "PEXELS_SEARCH_URL", f"{server.url}/pexels/search"
        )
        monkeypatch.setattr(
            stock_fetch, "PIXABAY_SEARCH_URL", f"{server.url}/pixabay/search"
        )
        monkeypatch.setenv("PEXELS_API_KEY", "pexels-test-key")
        monkeypatch.setenv("PIXABAY_API_KEY", "pixabay-test-key")
        yield server


# --- searching ------------------------------------------------------------

def test_pexels_results_are_normalised(stock_server) -> None:
    clips = PexelsSource("pexels-test-key").search(
        "city", count=5, min_duration=2.0, min_width=1280, orientation="landscape"
    )

    assert len(clips) == 1  # the 1-second clip was filtered out
    clip = clips[0]
    assert clip.source == "pexels"
    assert clip.id == "101"
    assert clip.duration == 12
    assert clip.credit_name == "Alice"
    assert clip.page_url.endswith("/101/")


def test_pixabay_results_are_normalised(stock_server) -> None:
    clips = PixabaySource("pixabay-test-key").search(
        "city", count=5, min_duration=2.0, min_width=1280, orientation="landscape"
    )

    assert len(clips) == 1
    clip = clips[0]
    assert clip.source == "pixabay"
    assert clip.id == "201"
    assert clip.credit_name == "Carol"
    assert "42" in clip.credit_url


def test_it_takes_the_smallest_file_that_still_meets_the_width(stock_server) -> None:
    """A 4K download that will be scaled to 1080p is wasted bandwidth."""
    clips = PexelsSource("pexels-test-key").search(
        "city", count=5, min_duration=2.0, min_width=1280, orientation="landscape"
    )

    assert clips[0].width == 1920
    assert "hd" in clips[0].download_url


def test_a_lower_width_requirement_takes_an_even_smaller_file(stock_server) -> None:
    clips = PexelsSource("pexels-test-key").search(
        "city", count=5, min_duration=2.0, min_width=640, orientation="landscape"
    )

    assert clips[0].width == 640


def test_search_sends_the_query_through(stock_server) -> None:
    PexelsSource("pexels-test-key").search(
        "rainy street", count=3, min_duration=0, min_width=0, orientation="portrait"
    )

    request = next(r for r in stock_server.requests if r.path == "/pexels/search")
    assert request.param("query") == "rainy street"
    assert request.param("orientation") == "portrait"


def test_a_rejected_key_says_which_variable_is_wrong(stock_server) -> None:
    with pytest.raises(MissingApiKey, match="PEXELS_API_KEY"):
        PexelsSource("wrong-key").search(
            "city", count=1, min_duration=0, min_width=0, orientation="landscape"
        )


def test_a_missing_key_is_reported_without_a_request(monkeypatch) -> None:
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)

    with pytest.raises(MissingApiKey, match="free at https://www.pexels.com/api/"):
        PexelsSource.from_env()


def test_placeholder_keys_from_the_template_count_as_missing(monkeypatch) -> None:
    """`.env` copied from `.env.example` must not look like a configured key."""
    monkeypatch.setenv("PEXELS_API_KEY", "your_pexels_key_here")

    with pytest.raises(MissingApiKey):
        PexelsSource.from_env()


def test_rate_limits_are_retried_after_a_pause(tiny_video_bytes: bytes, monkeypatch):
    state = {"calls": 0}

    def search(request: Request) -> Response:
        state["calls"] += 1
        if state["calls"] == 1:
            return Response.json({"error": "too many requests"}, status=429)
        return Response.json({"videos": []})

    with MockServer(routed({"/s": search})) as server:
        monkeypatch.setattr(stock_fetch, "PEXELS_SEARCH_URL", f"{server.url}/s")
        monkeypatch.setattr(stock_fetch, "BACKOFF_SECONDS", 0.01)

        result = PexelsSource("k").search(
            "city", count=1, min_duration=0, min_width=0, orientation="landscape"
        )

    assert result == []
    assert state["calls"] == 2


# --- fetching -------------------------------------------------------------

def test_fetch_downloads_from_both_sources(stock_server, tmp_path: Path) -> None:
    fetcher = StockFetcher(tmp_path / "stock")

    paths = fetcher.fetch(["city"], per_query=2, min_duration=2.0, min_width=1280)

    assert len(paths) == 2
    assert {path.exists() for path in paths} == {True}
    assert {path.name.split("_")[0] for path in paths} == {"pexels", "pixabay"}


def test_results_interleave_so_one_source_cannot_dominate(
    stock_server, tmp_path: Path
) -> None:
    fetcher = StockFetcher(tmp_path / "stock")

    clips = fetcher.search("city", count=2, min_duration=2.0, min_width=1280)

    assert [clip.source for clip in clips] == ["pexels", "pixabay"]


def test_a_cached_clip_is_not_downloaded_twice(stock_server, tmp_path: Path) -> None:
    fetcher = StockFetcher(tmp_path / "stock", sources=["pexels"])
    fetcher.fetch(["city"], per_query=1, min_duration=2.0, min_width=1280)
    downloads_after_first = sum(
        1 for r in stock_server.requests if r.path.startswith("/dl/")
    )

    fetcher.fetch(["city"], per_query=1, min_duration=2.0, min_width=1280)

    total = sum(1 for r in stock_server.requests if r.path.startswith("/dl/"))
    assert downloads_after_first == 1
    assert total == 1


def test_sources_without_keys_are_skipped_not_fatal(
    stock_server, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    fetcher = StockFetcher(tmp_path / "stock")

    paths = fetcher.fetch(["city"], per_query=2, min_duration=2.0, min_width=1280)

    assert len(paths) == 1
    assert paths[0].name.startswith("pexels_")


def test_no_keys_at_all_yields_no_sources(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)

    assert StockFetcher(tmp_path / "stock").available_sources() == []


def test_credits_are_recorded_for_everything_downloaded(
    stock_server, tmp_path: Path
) -> None:
    cache = tmp_path / "stock"
    StockFetcher(cache).fetch(["city"], per_query=2, min_duration=2.0, min_width=1280)

    credits = json.loads((cache / "CREDITS.json").read_text(encoding="utf-8"))
    markdown = (cache / "CREDITS.md").read_text(encoding="utf-8")

    assert len(credits) == 2
    assert any(entry["credit_name"] == "Alice" for entry in credits.values())
    assert "Alice" in markdown and "Carol" in markdown


def test_the_credit_ledger_survives_a_corrupt_file(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "CREDITS.json").write_text("{not json", encoding="utf-8")

    ledger = CreditLedger(tmp_path)
    ledger.record(StockClip("pexels", "1", "http://x", 1920, 1080, 5.0))
    ledger.save()

    assert json.loads((tmp_path / "CREDITS.json").read_text(encoding="utf-8"))


# --- brief parsing --------------------------------------------------------

def test_a_plain_brief_is_one_query() -> None:
    assert queries_from_brief("a rainy city at night") == ["a rainy city at night"]


def test_semicolons_and_newlines_split_a_brief() -> None:
    assert queries_from_brief("city at night; neon signs\nrain on glass") == [
        "city at night", "neon signs", "rain on glass",
    ]


def test_brief_parsing_is_capped(monkeypatch) -> None:
    brief = "; ".join(f"query {index}" for index in range(20))

    assert len(queries_from_brief(brief, limit=4)) == 4
