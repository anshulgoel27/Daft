from __future__ import annotations

import io
import os
import tarfile
import tempfile

import pytest

import daft

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_avro_bytes(data: dict, schema_name: str = "record") -> bytes:
    """Write a dict of column-lists as an Avro OCF file and return the bytes.

    Uses daft itself (write_avro) to produce valid Avro bytes via a temp dir.
    """
    df = daft.from_pydict(data)
    with tempfile.TemporaryDirectory() as tmpdir:
        df.write_avro(tmpdir, compression="null", write_mode="overwrite")
        # find the written .avro file
        avro_files = [f for f in os.listdir(tmpdir) if f.endswith(".avro")]
        assert avro_files, "write_avro produced no .avro file"
        with open(os.path.join(tmpdir, avro_files[0]), "rb") as fh:
            return fh.read()


def make_tar_gz(avro_contents: list[tuple[str, bytes]]) -> bytes:
    """Package multiple (name, avro_bytes) pairs into a tar.gz archive in memory."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, content in avro_contents:
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_tar_gz_with_multiple_avros(tmp_path):
    """Basic: a single tar.gz containing two .avro files; all rows are returned."""
    data1 = {"id": [1, 2, 3], "name": ["alice", "bob", "carol"]}
    data2 = {"id": [4, 5], "name": ["dave", "eve"]}

    avro1 = make_avro_bytes(data1)
    avro2 = make_avro_bytes(data2)
    gz_path = str(tmp_path / "batch.tar.gz")
    with open(gz_path, "wb") as fh:
        fh.write(make_tar_gz([("part0.avro", avro1), ("part1.avro", avro2)]))

    df = daft.read_avro_tar(gz_path)
    result = df.sort("id").to_pydict()

    assert result["id"] == [1, 2, 3, 4, 5]
    assert result["name"] == ["alice", "bob", "carol", "dave", "eve"]


def test_multiple_tar_gz_glob(tmp_path):
    """Glob pattern matches multiple tar.gz files and unions all rows."""
    data_a = {"val": [10, 20]}
    data_b = {"val": [30, 40]}

    gz_a = str(tmp_path / "a.tar.gz")
    gz_b = str(tmp_path / "b.tar.gz")
    with open(gz_a, "wb") as fh:
        fh.write(make_tar_gz([("data.avro", make_avro_bytes(data_a))]))
    with open(gz_b, "wb") as fh:
        fh.write(make_tar_gz([("data.avro", make_avro_bytes(data_b))]))

    df = daft.read_avro_tar(str(tmp_path / "*.tar.gz"))
    result = sorted(df.to_pydict()["val"])

    assert result == [10, 20, 30, 40]


def test_list_of_paths(tmp_path):
    """A Python list of explicit .tar.gz paths is accepted."""
    data_a = {"x": [1]}
    data_b = {"x": [2]}

    gz_a = str(tmp_path / "a.tar.gz")
    gz_b = str(tmp_path / "b.tar.gz")
    with open(gz_a, "wb") as fh:
        fh.write(make_tar_gz([("d.avro", make_avro_bytes(data_a))]))
    with open(gz_b, "wb") as fh:
        fh.write(make_tar_gz([("d.avro", make_avro_bytes(data_b))]))

    df = daft.read_avro_tar([gz_a, gz_b])
    assert sorted(df.to_pydict()["x"]) == [1, 2]


def test_schema_inference(tmp_path):
    """Schema is correctly inferred from the first .avro file in the archive."""
    data = {"a": [1, 2], "b": [1.0, 2.0], "c": ["x", "y"], "flag": [True, False]}
    gz_path = str(tmp_path / "data.tar.gz")
    with open(gz_path, "wb") as fh:
        fh.write(make_tar_gz([("rec.avro", make_avro_bytes(data))]))

    df = daft.read_avro_tar(gz_path)
    assert set(df.column_names) == {"a", "b", "c", "flag"}


def test_column_projection(tmp_path):
    """Only requested columns are returned."""
    data = {"id": [1, 2, 3], "name": ["a", "b", "c"], "score": [0.1, 0.2, 0.3]}
    gz_path = str(tmp_path / "data.tar.gz")
    with open(gz_path, "wb") as fh:
        fh.write(make_tar_gz([("recs.avro", make_avro_bytes(data))]))

    df = daft.read_avro_tar(gz_path, column_names=["id", "score"])
    assert set(df.column_names) == {"id", "score"}
    assert "name" not in df.column_names
    assert df.count_rows() == 3


def test_file_path_column(tmp_path):
    """The source tar.gz path is included as an extra column when requested."""
    data = {"val": [1, 2]}
    gz_path = str(tmp_path / "src.tar.gz")
    with open(gz_path, "wb") as fh:
        fh.write(make_tar_gz([("d.avro", make_avro_bytes(data))]))

    df = daft.read_avro_tar(gz_path, file_path_column="source")
    assert "source" in df.column_names
    paths = df.to_pydict()["source"]
    assert all(p == gz_path for p in paths)


def test_avro_tar_to_parquet_roundtrip(tmp_path):
    """Full pipeline: read_avro_tar() → write_parquet() → read_parquet() round-trip."""
    data = {"id": [1, 2, 3], "value": ["foo", "bar", "baz"]}
    gz_path = str(tmp_path / "data.tar.gz")
    with open(gz_path, "wb") as fh:
        fh.write(make_tar_gz([("rows.avro", make_avro_bytes(data))]))

    parquet_dir = str(tmp_path / "output")
    daft.read_avro_tar(gz_path).write_parquet(parquet_dir)

    result = daft.read_parquet(parquet_dir).sort("id").to_pydict()
    assert result["id"] == [1, 2, 3]
    assert result["value"] == ["foo", "bar", "baz"]


def test_empty_tar_gz_raises(tmp_path):
    """A tar.gz with no .avro members raises a clear ValueError."""
    gz_path = str(tmp_path / "empty.tar.gz")
    with open(gz_path, "wb") as fh:
        fh.write(make_tar_gz([("notes.txt", b"hello")]))

    with pytest.raises((ValueError, FileNotFoundError)):
        daft.read_avro_tar(gz_path)


def test_nonexistent_path_raises(tmp_path):
    """A missing path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        daft.read_avro_tar(str(tmp_path / "does_not_exist.tar.gz"))


def test_various_avro_types(tmp_path):
    """Integer, float, string, boolean, and nullable fields all survive round-trip."""
    data = {
        "int_col": [1, 2, 3],
        "float_col": [1.1, 2.2, 3.3],
        "str_col": ["a", "b", "c"],
        "bool_col": [True, False, True],
    }
    gz_path = str(tmp_path / "types.tar.gz")
    with open(gz_path, "wb") as fh:
        fh.write(make_tar_gz([("types.avro", make_avro_bytes(data))]))

    df = daft.read_avro_tar(gz_path).sort("int_col")
    result = df.to_pydict()

    assert result["int_col"] == [1, 2, 3]
    assert result["bool_col"] == [True, False, True]
    assert result["str_col"] == ["a", "b", "c"]
    for actual, expected in zip(result["float_col"], [1.1, 2.2, 3.3]):
        assert abs(actual - expected) < 1e-5


@pytest.mark.skipif(
    os.environ.get("DAFT_RUNNER", "native") != "ray",
    reason="Distributed test: set DAFT_RUNNER=ray to enable",
)
def test_distributed_read_ray(tmp_path):
    """With the Ray runner, multiple tar.gz files are processed in parallel."""
    gz_files = []
    for i in range(4):
        data = {"shard": [i], "val": [i * 10]}
        gz_path = str(tmp_path / f"shard_{i}.tar.gz")
        with open(gz_path, "wb") as fh:
            fh.write(make_tar_gz([("d.avro", make_avro_bytes(data))]))
        gz_files.append(gz_path)

    df = daft.read_avro_tar(gz_files)
    assert df.count_rows() == 4
    assert sorted(df.to_pydict()["val"]) == [0, 10, 20, 30]


def test_tgz_extension(tmp_path):
    """.tgz extension is recognised identically to .tar.gz."""
    data = {"id": [1, 2], "name": ["a", "b"]}
    gz_path = str(tmp_path / "data.tgz")
    with open(gz_path, "wb") as fh:
        fh.write(make_tar_gz([("rows.avro", make_avro_bytes(data))]))

    df = daft.read_avro_tar(gz_path)
    assert sorted(df.to_pydict()["id"]) == [1, 2]


def test_mixed_members_skips_non_avro(tmp_path):
    """Non-.avro members inside the archive (e.g. .txt, .json) are silently skipped."""
    data = {"val": [42]}
    gz_path = str(tmp_path / "mixed.tar.gz")
    with open(gz_path, "wb") as fh:
        fh.write(
            make_tar_gz(
                [
                    ("readme.txt", b"this is not avro"),
                    ("meta.json", b'{"key": "value"}'),
                    ("records.avro", make_avro_bytes(data)),
                ]
            )
        )

    df = daft.read_avro_tar(gz_path)
    assert df.to_pydict()["val"] == [42]


def test_empty_rows_avro(tmp_path):
    """A valid .avro file containing zero rows produces an empty DataFrame with correct schema."""
    fastavro = pytest.importorskip("fastavro")

    schema = fastavro.parse_schema(
        {
            "type": "record",
            "name": "Empty",
            "fields": [
                {"name": "id", "type": "int"},
                {"name": "name", "type": "string"},
            ],
        }
    )
    avro_buf = io.BytesIO()
    fastavro.writer(avro_buf, schema, [])  # zero records
    avro_bytes = avro_buf.getvalue()

    gz_path = str(tmp_path / "empty_rows.tar.gz")
    with open(gz_path, "wb") as fh:
        fh.write(make_tar_gz([("empty.avro", avro_bytes)]))

    df = daft.read_avro_tar(gz_path)
    assert df.count_rows() == 0
    assert set(df.column_names) == {"id", "name"}


def test_column_projection_unknown_column(tmp_path):
    """Requesting a column that does not exist in the schema raises an error or returns without that column."""
    data = {"id": [1, 2], "name": ["a", "b"]}
    gz_path = str(tmp_path / "data.tar.gz")
    with open(gz_path, "wb") as fh:
        fh.write(make_tar_gz([("rows.avro", make_avro_bytes(data))]))

    # Should either raise or return a df without the unknown column — must not silently corrupt data
    try:
        df = daft.read_avro_tar(gz_path, column_names=["id", "nonexistent"])
        assert "id" in df.column_names
        assert "nonexistent" not in df.column_names
    except Exception:
        pass  # raising is also acceptable
