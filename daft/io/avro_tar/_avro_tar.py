from __future__ import annotations

import io
import os
import tarfile
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

import daft
from daft.api_annotations import PublicAPI
from daft.daft import io_glob
from daft.daft import read_avro as _rust_read_avro
from daft.daft import read_avro_schema as _rust_read_avro_schema
from daft.filesystem import _resolve_paths_and_filesystem
from daft.io.source import DataSource, DataSourceTask
from daft.logical.schema import Schema
from daft.recordbatch import RecordBatch

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from daft import DataFrame
    from daft.io import IOConfig
    from daft.io.pushdowns import Pushdowns


def _list_tar_gz_files(path: str, io_config: IOConfig | None) -> list[str]:
    """List all .tar.gz / .tgz files at the given path or matching the given glob pattern."""
    from daft.dependencies import pafs

    if "*" in path or "?" in path:
        files = io_glob(path, io_config=io_config)
        return [f["path"] for f in files if f["type"] == "File"]

    [resolved], fs = _resolve_paths_and_filesystem(path, io_config=io_config)
    try:
        file_info = fs.get_file_info(resolved)
    except FileNotFoundError:
        return []

    if file_info.type == pafs.FileType.File:
        return [resolved]
    if file_info.type != pafs.FileType.Directory:
        # Covers FileType.NotFound and FileType.Unknown — path does not exist
        return []

    selector = pafs.FileSelector(resolved, recursive=True)
    try:
        infos = fs.get_file_info(selector)
    except (NotADirectoryError, FileNotFoundError):
        return []

    return [
        fi.path
        for fi in infos
        if fi.type == pafs.FileType.File and (fi.path.endswith(".tar.gz") or fi.path.endswith(".tgz"))
    ]


def _read_first_avro_schema(tar_gz_path: str, io_config: IOConfig | None) -> Schema:
    """Download a tar.gz, extract the first .avro member, and read its Avro schema."""
    with daft.open_file(tar_gz_path, "rb", io_config=io_config) as f:
        gz_bytes = f.read()

    with tarfile.open(fileobj=io.BytesIO(gz_bytes), mode="r:gz") as tf:
        for member in tf.getmembers():
            if not member.name.endswith(".avro") or not member.isfile():
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            avro_bytes = extracted.read()
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".avro", delete=False) as tmp:
                    tmp.write(avro_bytes)
                    tmp_path = tmp.name
                py_schema = _rust_read_avro_schema(tmp_path)
                return Schema._from_pyschema(py_schema)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    raise ValueError(f"No .avro files found inside tar.gz archive: {tar_gz_path}")


@PublicAPI
def read_avro_tar(
    path: str | list[str],
    io_config: IOConfig | None = None,
    column_names: list[str] | None = None,
    file_path_column: str | None = None,
) -> DataFrame:
    """Creates a DataFrame from Avro files packaged inside tar.gz archives.

    Each .tar.gz archive may contain multiple .avro files. All .avro files within
    each archive are read and concatenated. Archives are distributed across workers
    when using the Ray runner, with one task per tar.gz file.

    All .avro files across all archives are assumed to share the same schema.
    Schema is inferred from the first .avro file in the first archive.

    Args:
        path: Path, glob pattern, or list of paths to ``.tar.gz`` / ``.tgz`` files.
            Supports S3 (``s3://``), GCS (``gs://``), Azure Blob (``abfs://``), and local paths.
            Glob wildcards (``*``, ``?``) are supported.
        io_config: :class:`~daft.io.IOConfig` for authentication and storage configuration
            when reading from object stores.
        column_names: Optional list of column names to project. Only these columns will be read.
            Defaults to ``None`` (read all columns).
        file_path_column: If provided, include the source ``.tar.gz`` file path as an
            extra column with this name. Defaults to ``None``.

    Returns:
        DataFrame: DataFrame with the schema inferred from the Avro files.

    Examples:
        Read Avro files from tar.gz archives in an S3 bucket:

        >>> from daft.io import IOConfig, S3Config
        >>> io_config = IOConfig(s3=S3Config(region="us-east-1"))
        >>> df = daft.read_avro_tar("s3://my-bucket/data/*.tar.gz", io_config=io_config)
        >>> df.show()

        Convert to Parquet using distributed execution:

        >>> df = daft.read_avro_tar("s3://my-bucket/data/*.tar.gz", io_config=io_config)
        >>> df.write_parquet("s3://my-bucket/output/")

        Read only specific columns:

        >>> df = daft.read_avro_tar(
        ...     "s3://my-bucket/data/*.tar.gz",
        ...     io_config=io_config,
        ...     column_names=["id", "name", "timestamp"],
        ... )
    """
    paths = [path] if isinstance(path, str) else list(path)
    return AvroTarSource(
        paths=paths,
        io_config=io_config,
        column_names=column_names,
        file_path_column=file_path_column,
    ).read()


class AvroTarSource(DataSource):
    def __init__(
        self,
        paths: list[str],
        io_config: IOConfig | None = None,
        column_names: list[str] | None = None,
        file_path_column: str | None = None,
    ):
        self._io_config = io_config
        self._column_names = column_names
        self._file_path_column = file_path_column

        # Expand all paths/globs into a flat list of .tar.gz URIs
        self._tar_gz_uris: list[str] = []
        for p in paths:
            self._tar_gz_uris.extend(_list_tar_gz_files(p, io_config))

        if not self._tar_gz_uris:
            raise FileNotFoundError(f"No .tar.gz files found at: {paths}")

        # Infer schema from the first archive
        import pyarrow as pa

        base_schema = _read_first_avro_schema(self._tar_gz_uris[0], io_config)

        # Apply column projection to the schema if requested
        if column_names is not None:
            col_set = set(column_names)
            full_pa = base_schema.to_pyarrow_schema()
            projected_fields = [f for f in full_pa if f.name in col_set]
            base_schema = Schema.from_pyarrow_schema(pa.schema(projected_fields))

        # Optionally append the file_path_column to the schema
        if file_path_column is not None:
            path_col_schema = Schema.from_pyarrow_schema(pa.schema([pa.field(file_path_column, pa.large_string())]))
            self._schema = base_schema.union(path_col_schema)
        else:
            self._schema = base_schema

    @property
    def name(self) -> str:
        return "AvroTarSource"

    @property
    def schema(self) -> Schema:
        return self._schema

    def display_name(self) -> str:
        return f"AvroTarSource({self._tar_gz_uris})"

    def multiline_display(self) -> list[str]:
        return [
            self.display_name(),
            f"Schema = {self._schema}",
        ]

    async def get_tasks(self, pushdowns: Pushdowns) -> AsyncIterator[AvroTarSourceTask]:
        for uri in self._tar_gz_uris:
            yield AvroTarSourceTask(
                _uri=uri,
                _schema=self._schema,
                _io_config=self._io_config,
                _column_names=self._column_names,
                _file_path_column=self._file_path_column,
            )


@dataclass
class AvroTarSourceTask(DataSourceTask):
    _uri: str
    _schema: Schema
    _io_config: IOConfig | None = None
    _column_names: list[str] | None = None
    _file_path_column: str | None = None

    @property
    def schema(self) -> Schema:
        return self._schema

    async def read(self) -> AsyncIterator[RecordBatch]:
        with daft.open_file(self._uri, "rb", io_config=self._io_config) as f:
            gz_bytes = f.read()

        with tarfile.open(fileobj=io.BytesIO(gz_bytes), mode="r:gz") as tf:
            for member in tf.getmembers():
                if not member.name.endswith(".avro") or not member.isfile():
                    continue
                extracted = tf.extractfile(member)
                if extracted is None:
                    continue

                avro_bytes = extracted.read()
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".avro", delete=False) as tmp:
                        tmp.write(avro_bytes)
                        tmp_path = tmp.name

                    # Use the Rust Avro reader via a local tempfile path
                    py_batch = _rust_read_avro(
                        tmp_path,
                        io_config=None,  # tempfile is local, no io_config needed
                        column_projection=self._column_names,
                    )
                    rb = RecordBatch._from_pyrecordbatch(py_batch)

                    # Apply Python-level column projection as a safety net
                    if self._column_names is not None:
                        arrow_table = rb.to_arrow_table()
                        arrow_table = arrow_table.select(
                            [c for c in self._column_names if c in arrow_table.schema.names]
                        )
                        rb = RecordBatch.from_arrow_table(arrow_table)

                    if self._file_path_column is not None:
                        import pyarrow as pa

                        n_rows = len(rb)
                        arrow_table = rb.to_arrow_table()
                        path_array = pa.array([self._uri] * n_rows, type=pa.large_string())
                        combined = arrow_table.append_column(
                            pa.field(self._file_path_column, pa.large_string()),
                            path_array,
                        )
                        rb = RecordBatch.from_arrow_table(combined)

                    yield rb
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)
