#!/usr/bin/env python3
"""Acquire the study's 25 voraus-AD columns via validated HTTP byte ranges.

This is a portable replacement for the exploratory acquisition scripts. It does
not reproduce their weaker HTTP validation. --self-test uses synthetic data and
an in-memory HTTP mock; it does not download the research data.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
from datetime import datetime, timezone
import urllib.error
import urllib.request

URL = "https://media.vorausrobotik.com/voraus-ad-dataset-100hz.parquet"
COLUMNS = ["time", "sample", "anomaly", "category", "setting", "action", "active"] + [
    f"{name}_{i}" for name in ("motor_torque", "joint_position", "joint_velocity")
    for i in range(1, 7)
]
LICENSE_SOURCE = "https://github.com/vorausrobotik/voraus-ad-dataset"


class IntegrityError(RuntimeError):
    """The server response cannot safely be used as part of the pinned object."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def response_range(response, start, end):
    if response.status != 206:
        raise IntegrityError(f"Expected HTTP 206, received {response.status}; no full-file fallback")
    encoding = response.headers.get("Content-Encoding", "identity").lower()
    if encoding != "identity":
        raise IntegrityError(f"Unexpected content encoding: {encoding}")
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
    if match is None:
        raise IntegrityError("Missing or malformed Content-Range")
    a, b, total = map(int, match.groups())
    if (a, b) != (start, end) or total <= end:
        raise IntegrityError(f"Unexpected Content-Range: {a}-{b}/{total}; requested {start}-{end}")
    size = end - start + 1
    content_length = response.headers.get("Content-Length")
    if content_length is not None and int(content_length) != size:
        raise IntegrityError("Range Content-Length does not match requested interval")
    return total, size


class RangeFile(io.RawIOBase):
    """Seekable read-only file, pinned by strong ETag and total object length."""

    def __init__(self, source, timeout=45, retries=3, opener=None, event=None):
        super().__init__()
        self.source = source
        self.pos = 0
        self.size = source["content_length"]
        self.transferred = 0
        self.requests = 0
        self.timeout = timeout
        self.retries = retries
        self.opener = opener or urllib.request.urlopen
        self.event = event or (lambda event: None)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence not in (io.SEEK_SET, io.SEEK_CUR, io.SEEK_END):
            raise ValueError("Invalid whence")
        pos = offset + (0 if whence == io.SEEK_SET else self.pos if whence == io.SEEK_CUR else self.size)
        if pos < 0:
            raise ValueError("Negative seek")
        self.pos = pos
        return pos

    def _fetch(self, start, end):
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
                   "If-Match": self.source["etag"], "User-Agent": "voraus-research-acquisition/1.0"}
        request = urllib.request.Request(self.source["url"], headers=headers)
        for attempt in range(1, self.retries + 1):
            self.requests += 1
            try:
                with self.opener(request, timeout=self.timeout) as response:
                    total, expected = response_range(response, start, end)
                    if total != self.size or response.headers.get("ETag") != self.source["etag"]:
                        raise IntegrityError("Source ETag or total length changed during acquisition")
                    if response.geturl() != self.source["resolved_url"]:
                        raise IntegrityError("Source redirect destination changed")
                    data = response.read(expected + 1)
                    self.transferred += len(data)
                    if len(data) != expected:
                        raise IntegrityError(f"Incomplete or oversized range: expected {expected}, got {len(data)}")
                    self.event({"kind": "range_ok", "start": start, "end": end,
                                "attempt": attempt, "bytes": len(data)})
                    return data
            except urllib.error.HTTPError as exc:
                retryable = exc.code in (408, 429, 500, 502, 503, 504)
                self.event({"kind": "range_error", "start": start, "end": end,
                            "attempt": attempt, "error": repr(exc), "retryable": retryable})
                if not retryable or attempt == self.retries:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                self.event({"kind": "range_error", "start": start, "end": end,
                            "attempt": attempt, "error": repr(exc), "retryable": True})
                if attempt == self.retries:
                    raise
            # Integrity errors intentionally are not retried or silently accepted.
            time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError("Unreachable retry state")

    def read(self, n=-1):
        if self.closed:
            raise ValueError("Read on closed file")
        if self.pos >= self.size or n == 0:
            return b""
        if n is None or n < 0:
            n = self.size - self.pos
        end = min(self.size, self.pos + n) - 1
        data = self._fetch(self.pos, end)
        self.pos += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def probe_source(url, timeout=45, retries=3, opener=None):
    opener = opener or urllib.request.urlopen
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"})
    for attempt in range(1, retries + 1):
        try:
            with opener(request, timeout=timeout) as response:
                total, expected = response_range(response, 0, 0)
                etag = response.headers.get("ETag")
                if not etag or etag.startswith("W/") or not (etag.startswith('"') and etag.endswith('"')):
                    raise IntegrityError("A strong quoted ETag is required to pin ranged reads")
                if len(response.read(expected + 1)) != expected:
                    raise IntegrityError("Probe body length mismatch")
                return {"url": url, "resolved_url": response.geturl(), "etag": etag,
                        "content_length": total, "last_modified": response.headers.get("Last-Modified"),
                        "probe_time_utc": utc_now(), "probe_attempt": attempt,
                        "original_file_sha256": None,
                        "original_file_sha256_reason": "Selected ranges only; complete object was not downloaded"}
        except urllib.error.HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries:
                raise
        time.sleep(min(2 ** (attempt - 1), 8))


def acquire(args):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = args.output_columns.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = (args.manifest or out / "acquisition_manifest.json").resolve()
    event_path = manifest_path.with_suffix(".events.jsonl")
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    existing = [name for name in COLUMNS if (out / f"{name}.npy").exists()]
    if existing and (not args.resume or previous is None):
        raise IntegrityError("Existing column files require --resume and their acquisition manifest; use a new directory")
    source = probe_source(args.url, args.timeout, args.retries)
    if args.expected_etag and args.expected_etag.strip('"') != source["etag"].strip('"'):
        raise IntegrityError("Source does not match --expected-etag")
    if args.expected_content_length and args.expected_content_length != source["content_length"]:
        raise IntegrityError("Source does not match --expected-content-length")
    if previous:
        if previous.get("schema_version") != 1 or previous.get("selected_columns") != COLUMNS:
            raise IntegrityError("Cannot resume an incompatible or exploratory manifest")
        for field in ("url", "resolved_url", "etag", "content_length"):
            if previous["source"][field] != source[field]:
                raise IntegrityError(f"Cannot resume: source {field} changed")
    run = {"schema_version": 1, "status": "running", "started_utc": utc_now(),
           "source": source, "selected_columns": COLUMNS, "columns": [],
           "dataset_license": "CC BY-NC-SA-4.0", "license_evidence_url": LICENSE_SOURCE,
           "repository_code_license": "MIT (does not apply to the dataset)",
           "versions": {"python": sys.version, "numpy": np.__version__, "pyarrow": pa.__version__},
           "retry_policy": {"max_attempts_per_range": args.retries,
                            "backoff_seconds": "1, 2, 4, then capped at 8",
                            "integrity_errors": "abort without retry"}}
    previous_items = {item["column"]: item for item in (previous or {}).get("columns", [])}
    atomic_json(manifest_path, run)
    with event_path.open("a", encoding="utf-8") as events:
        def emit(event):
            events.write(json.dumps({"time_utc": utc_now(), **event}) + "\n")
            events.flush()
        try:
            for column in COLUMNS:
                target = out / f"{column}.npy"
                start = time.monotonic()
                if target.exists():
                    old = previous_items.get(column)
                    if not old or sha256(target) != old["sha256"] or target.stat().st_size != old["bytes"]:
                        raise IntegrityError(f"Cached column has no matching verified hash: {column}")
                    item = {**old, "file": target.name, "reused": True}
                    run["columns"].append(item)
                    emit({"kind": "cache_verified", "column": column})
                    atomic_json(manifest_path, run)
                    continue
                with RangeFile(source, args.timeout, args.retries,
                               event=lambda event: emit({"column": column, **event})) as remote:
                    parquet = pq.ParquetFile(remote)
                    if column not in parquet.schema_arrow.names:
                        raise IntegrityError(f"Column absent from source schema: {column}")
                    num_rows = parquet.metadata.num_rows
                    if "num_rows" in run and run["num_rows"] != num_rows:
                        raise IntegrityError("Parquet row count changed between columns")
                    run["num_rows"] = num_rows
                    run["row_groups"] = parquet.metadata.num_row_groups
                    table = parquet.read(columns=[column], use_threads=False)
                    values = table.column(0)
                    if values.null_count:
                        raise IntegrityError(f"Unexpected null values in {column}; no imputation is authorized")
                    array = values.to_numpy(zero_copy_only=False)
                    if array.ndim != 1 or len(array) != num_rows or array.dtype.hasobject:
                        raise IntegrityError(f"Unexpected shape/type for {column}")
                    part = target.with_name(target.name + ".part")
                    with part.open("wb") as stream:
                        np.save(stream, array, allow_pickle=False)
                        stream.flush()
                        os.fsync(stream.fileno())
                    item = {"column": column, "file": target.name, "sha256": sha256(part),
                            "bytes": part.stat().st_size, "rows": len(array), "dtype": str(array.dtype),
                            "transfer_bytes": remote.transferred, "range_requests": remote.requests,
                            "seconds": time.monotonic() - start, "reused": False}
                    part.replace(target)
                run["columns"].append(item)
                atomic_json(manifest_path, run)
                print(json.dumps(item), flush=True)
            lengths = {item["rows"] for item in run["columns"]}
            if len(lengths) != 1:
                raise IntegrityError("Column row counts disagree")
            run["num_rows"] = next(iter(lengths))
            run["status"] = "complete"
            run["finished_utc"] = utc_now()
            atomic_json(manifest_path, run)
        except BaseException as exc:
            run["status"] = "failed"
            run["finished_utc"] = utc_now()
            run["error"] = repr(exc)
            emit({"kind": "acquisition_failed", "error": repr(exc)})
            atomic_json(manifest_path, run)
            raise
    print(f"Complete: {len(COLUMNS)} columns. Manifest: {manifest_path}")


def self_test():
    """Synthetic Parquet + mocked HTTP only; never opens a network connection."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    buffer = io.BytesIO()
    pq.write_table(pa.table({name: np.arange(9, dtype=np.int64) for name in COLUMNS}), buffer)
    content = buffer.getvalue()
    class Response(io.BytesIO):
        def __init__(self, data, headers, status=206):
            super().__init__(data)
            self.headers, self.status = headers, status
        def geturl(self):
            return URL
    def factory(mode="good"):
        def open_mock(request, timeout=45):
            a, b = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range")).groups())
            data = content[a:b + 1]
            headers = {"Content-Range": f"bytes {a}-{b}/{len(content)}",
                       "Content-Length": str(len(data)), "ETag": '"synthetic-v1"'}
            if mode == "changed": headers["ETag"] = '"synthetic-v2"'
            if mode == "wrong_range": headers["Content-Range"] = f"bytes 1-{b}/{len(content)}"
            if mode == "short": data = data[:-1]
            return Response(data, headers, 200 if mode == "status200" else 206)
        return open_mock
    source = probe_source(URL, opener=factory())
    remote = RangeFile(source, opener=factory())
    table = pq.ParquetFile(remote).read(columns=COLUMNS, use_threads=False)
    assert table.num_rows == 9 and table.num_columns == 25
    for name in COLUMNS:
        assert np.array_equal(table[name].to_numpy(), np.arange(9))
    remote.seek(remote.size)
    assert remote.read(1) == b""
    for mode in ("changed", "wrong_range", "short", "status200"):
        try:
            RangeFile(source, opener=factory(mode)).read(2)
        except IntegrityError:
            pass
        else:
            raise AssertionError(f"Failed to reject {mode}")
    print("PASS: synthetic 25-column Parquet read; EOF; changed ETag; invalid Content-Range; short body; ignored Range")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-columns", type=Path, default=Path("data/columns"))
    parser.add_argument("--manifest", type=Path, help="Default: OUTPUT_COLUMNS/acquisition_manifest.json")
    parser.add_argument("--url", default=URL)
    parser.add_argument("--expected-etag", help="Optional pinned historical ETag; with or without surrounding quotes")
    parser.add_argument("--expected-content-length", type=int)
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="Reuse only files hash-verified against this script's manifest")
    parser.add_argument("--self-test", action="store_true", help="Synthetic mocked HTTP test; no network or research-data access")
    args = parser.parse_args()
    if args.retries < 1 or args.timeout <= 0:
        parser.error("retries and timeout must be positive")
    if args.self_test:
        self_test()
    else:
        acquire(args)


if __name__ == "__main__":
    main()
