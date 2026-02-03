import json

from functions.core import search as search_module


class _FakeBlob:
    def __init__(self, name: str, content: str) -> None:
        self.name = name
        self._content = content

    def download_as_text(self) -> str:
        return self._content


class _FakeBucket:
    def __init__(self, blobs: list[_FakeBlob]) -> None:
        self._blobs = blobs

    def list_blobs(self, prefix: str):
        return self._blobs


class _FakeStorageClient:
    def __init__(self, bucket: _FakeBucket) -> None:
        self._bucket = bucket

    def bucket(self, name: str) -> _FakeBucket:
        return self._bucket


def test_load_metadata_from_gcs_filters_to_requested_ids(monkeypatch):
    record = {
        "id": "item-1",
        "embedding_metadata": {"id": "item-1", "title": "hello"},
    }
    other = {
        "id": "item-2",
        "embedding_metadata": {"id": "item-2", "title": "skip"},
    }
    content = "\n".join([json.dumps(record), json.dumps(other)]) + "\n"
    blobs = [_FakeBlob("part-00000.json", content)]
    bucket = _FakeBucket(blobs)

    monkeypatch.setattr(
        search_module.storage,
        "Client",
        lambda: _FakeStorageClient(bucket),
    )

    found = search_module._load_metadata_from_gcs("gs://bucket/prefix", {"item-1"})
    assert found == {"item-1": {"id": "item-1", "title": "hello"}}


def test_load_metadata_from_gcs_skips_non_dict_records(monkeypatch):
    record = {
        "id": "item-1",
        "embedding_metadata": {"id": "item-1", "title": "hello"},
    }
    content = "\n".join([json.dumps([record]), json.dumps(record)]) + "\n"
    blobs = [_FakeBlob("part-00000.json", content)]
    bucket = _FakeBucket(blobs)

    monkeypatch.setattr(
        search_module.storage,
        "Client",
        lambda: _FakeStorageClient(bucket),
    )

    found = search_module._load_metadata_from_gcs("gs://bucket/prefix", {"item-1"})
    assert found == {"item-1": {"id": "item-1", "title": "hello"}}
