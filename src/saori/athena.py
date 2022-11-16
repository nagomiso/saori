from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

import awswrangler as _wr
import boto3
import jinja2
from pandas import DataFrame
from pathlibfs import Path

from .path import mktempdir

if sys.version_info < (3, 9):
    from typing import Iterator
else:
    from collections.abc import Iterator

LOGGER = logging.getLogger(__name__)


def _move_file(src, dst) -> None:
    src_path = Path(src)
    dst_dir = Path(dst)
    src_path.move(dst_dir / src_path.name)


def read_sql_query(
    sql: str,
    database: str,
    ctas_approach: bool = True,
    unload_approach: bool = False,
    unload_parameters: dict[str, Any] | None = None,
    categories: list[str] | None = None,
    chunksize: int | bool | None = None,
    s3_output: str | None = None,
    workgroup: str | None = None,
    encryption: str | None = None,
    kms_key: str | None = None,
    keep_files: bool = True,
    ctas_database_name: str | None = None,
    ctas_temp_table_name: str | None = None,
    ctas_bucketing_info: tuple[list[str], int] | None = None,
    use_threads: bool | int = True,
    boto3_session: boto3.Session | None = None,
    max_cache_seconds: int = 0,
    max_cache_query_inspections: int = 50,
    max_remote_cache_entries: int = 50,
    max_local_cache_entries: int = 100,
    data_source: str | None = None,
    params: dict[str, Any] | None = None,
    jinja_params: dict[str, Any] | None = None,
    jinja_options: dict[str, Any] | None = None,
    s3_additional_kwargs: dict[str, Any] | None = None,
    pyarrow_additional_kwargs: dict[str, Any] | None = None,
) -> DataFrame | Iterator[DataFrame]:
    """Same of wr.athena.read_sql_query().

    In addition, this function can use Jinja for template rendering.

    Args:
        sql: SQL query.
        database: \
            AWS Glue/Athena database name - It is only the origin database \
            from where the query will be launched. \
            You can still using and mixing several databases writing \
            the full table name within the sql (e.g. `database.table`).
        ctas_approach: \
            Wraps the query using a CTAS, and read the resulted parquet data on S3. \
            If false, read the regular CSV on S3.
        unload_approach: \
            Wraps the query using UNLOAD, and read the results from S3. \
            Only PARQUET format is supported.
        unload_parameters: \
            Params of the UNLOAD such as format, compression, field_delimiter, \
            and partitioned_by.
        categories: \
            List of columns names that should be returned as pandas.Categorical. \
            Recommended for memory restricted environments.
        chunksize: \
            If passed will split the data in a Iterable of \
            DataFrames (Memory friendly). \
            If `True` awswrangler iterates on the data by files \
            in the most efficient way without guarantee of chunksize. \
            If an `INTEGER` is passed awswrangler will iterate on \
            the data by number of rows igual the received INTEGER.
        s3_output: Amazon S3 path.
        workgroup: Athena workgroup.
        encryption: \
            Valid values: [None, 'SSE_S3', 'SSE_KMS']. \
            Notice: 'CSE_KMS' is not supported.
        kms_key: For SSE-KMS, this is the KMS key ARN or ID.
        keep_files: \
            Whether staging files produced by Athena are retained. 'True' by default.
        ctas_database_name: \
            The name of the alternative database \
            where the CTAS temporary table is stored. \
            If None, the default `database` is used.
        ctas_temp_table_name: \
            The name of the temporary table and also the directory name on S3 \
            where the CTAS result is stored. \
            If None, it will use the follow \
            random pattern: `f"temp_table_{uuid.uuid4().hex()}"`. \
            On S3 this directory will be under under \
            the pattern: `f"{s3_output}/{ctas_temp_table_name}/"`.
        ctas_bucketing_info: \
            Tuple consisting of the column names used for bucketing \
            as the first element and the number of buckets as the second element. \
            Only `str`, `int` and `bool` are supported \
            as column data types for bucketing.
        use_threads: \
            True to enable concurrent requests, False to disable multiple threads. \
            If enabled os.cpu_count() will be used as the max number of threads. \
            If integer is provided, specified number is used.
        boto3_session:
            Boto3 Session. \
            The default boto3 session will be used if boto3_session receive None.
        max_cache_seconds: \
            awswrangler can look up in Athena's history \
            if this query has been run before.
            If so, and its completion time is less than `max_cache_seconds` \
            before now, awswrangler skips query execution \
            and just returns the same results as last time. \
            If cached results are valid, awswrangler ignores \
            the `ctas_approach`, `s3_output`, `encryption`, `kms_key`, \
            `keep_files` and `ctas_temp_table_name` params. \
            If reading cached data fails for any reason, \
            execution falls back to the usual query run path.
        max_cache_query_inspections: \
            Max number of queries that will be inspected from \
            the history to try to find some result to reuse. \
            The bigger the number of inspection, \
            the bigger will be the latency for not cached queries. \
            Only takes effect if max_cache_seconds > 0.
        max_remote_cache_entries: \
            Max number of queries that will be retrieved \
            from AWS for cache inspection. \
            The bigger the number of inspection, \
            the bigger will be the latency for not cached queries. \
            Only takes effect if max_cache_seconds > 0 and default value is 50.
        max_local_cache_entries:
            Max number of queries for which metadata will be cached locally. \
            This will reduce the latency and also enables keeping more than \
            `max_remote_cache_entries` available for the cache. \
            This value should not be smaller than max_remote_cache_entries. \
            Only takes effect if max_cache_seconds > 0 and default value is 100.
        data_source: \
            Data Source / Catalog name. \
            If None, 'AwsDataCatalog' will be used by default.
        params: \
            Dict of parameters that will be used for constructing the SQL query. \
            Only named parameters are supported. \
            The dict needs to contain the information in the form {'name': 'value'} \
            and the SQL query needs to contain \
            `:name;`. Note that for varchar columns and similar, \
            you must surround the value in single quotes.
        jinja_params: \
            Dict of Jinja rendering parameters.
        jinja_options: \
            Dict of Jinja template options.
        s3_additional_kwargs: \
            Forwarded to botocore requests. \
            e.g. s3_additional_kwargs={'RequestPayer': 'requester'}
        pyarrow_additional_kwargs: \
            Forward to the ParquetFile class or converting an Arrow table to Pandas, \
            currently only an "coerce_int96_timestamp_unit" or "timestamp_as_object" \
            argument will be considered. If reading parquet \
            files where you cannot convert a timestamp to pandas Timestamp[ns] \
            consider setting timestamp_as_object=True, \
            to allow for timestamp units larger than "ns". \
            If reading parquet data that still uses INT96 (like Athena outputs) \
            you can use coerce_int96_timestamp_unit to specify \
            what timestamp unit to encode INT96 to (by default this is "ns", \
            if you know the output parquet came \
            from a system that encodes timestamp to a particular unit \
            then set this to that same unit e.g. coerce_int96_timestamp_unit="ms").

    Returns:
        DataFrame | Iterator[DataFrame]: \
            Pandas DataFrame or Generator of Pandas DataFrames if chunksize is passed.
    """
    if jinja_params:
        if not isinstance(jinja_options, dict):
            jinja_options = {}
        template: jinja2.Template = jinja2.Template(source=sql, **jinja_options)
        sql_query = template.render(**jinja_params)
    else:
        sql_query = sql
    LOGGER.debug("Rendered SQL:\n%s", sql_query)
    try:
        return _wr.athena.read_sql_query(
            sql=sql_query,
            database=database,
            ctas_approach=ctas_approach,
            unload_approach=unload_approach,
            unload_parameters=unload_parameters,
            categories=categories,
            chunksize=chunksize,
            s3_output=s3_output,
            workgroup=workgroup,
            encryption=encryption,
            kms_key=kms_key,
            keep_files=keep_files,
            ctas_database_name=ctas_database_name,
            ctas_temp_table_name=ctas_temp_table_name,
            ctas_bucketing_info=ctas_bucketing_info,
            use_threads=use_threads,
            boto3_session=boto3_session,
            max_cache_seconds=max_cache_seconds,
            max_cache_query_inspections=max_cache_query_inspections,
            max_remote_cache_entries=max_remote_cache_entries,
            max_local_cache_entries=max_local_cache_entries,
            data_source=data_source,
            params=params,
            s3_additional_kwargs=s3_additional_kwargs,
            pyarrow_additional_kwargs=pyarrow_additional_kwargs,
        )
    except _wr.exceptions.QueryFailed:
        LOGGER.error("Failed Query:\n%s", sql_query)
        raise


def save_sql_query_results(
    sql: str,
    save_dir: str | Path,
    database: str,
    categories: list[str] | None = None,
    s3_output: str | None = None,
    workgroup: str | None = None,
    encryption: str | None = None,
    kms_key: str | None = None,
    ctas_database_name: str | None = None,
    ctas_bucketing_info: tuple[list[str], int] | None = None,
    use_threads: bool | int = True,
    boto3_session: boto3.Session | None = None,
    max_cache_seconds: int = 0,
    max_cache_query_inspections: int = 50,
    max_remote_cache_entries: int = 50,
    max_local_cache_entries: int = 100,
    data_source: str | None = None,
    params: dict[str, Any] | None = None,
    jinja_params: dict[str, Any] | None = None,
    jinja_options: dict[str, Any] | None = None,
    s3_additional_kwargs: dict[str, Any] | None = None,
    pyarrow_additional_kwargs: dict[str, Any] | None = None,
    max_workers: int | None = None,
) -> None:
    """Save query results in Parquet format.

    Args:
        sql: SQL query.
        save_dir: \
            Directory path to save the results. \
            If this value is local filesystem directory path, \
            you must specify `s3_output`.
        database: \
            AWS Glue/Athena database name - It is only the origin database \
            from where the query will be launched. \
            You can still using and mixing several databases writing \
            the full table name within the sql (e.g. `database.table`).
        categories: \
            List of columns names that should be returned as pandas.Categorical. \
            Recommended for memory restricted environments.
        s3_output: Amazon S3 path.
        workgroup: Athena workgroup.
        encryption: \
            Valid values: [None, 'SSE_S3', 'SSE_KMS']. \
            Notice: 'CSE_KMS' is not supported.
        kms_key: For SSE-KMS, this is the KMS key ARN or ID.
        keep_files: \
            Whether staging files produced by Athena are retained. 'True' by default.
        ctas_database_name: \
            The name of the alternative database \
            where the CTAS temporary table is stored. \
            If None, the default `database` is used.
        ctas_bucketing_info: \
            Tuple consisting of the column names used for bucketing \
            as the first element and the number of buckets as the second element. \
            Only `str`, `int` and `bool` are supported \
            as column data types for bucketing.
        use_threads: \
            True to enable concurrent requests, False to disable multiple threads. \
            If enabled os.cpu_count() will be used as the max number of threads. \
            If integer is provided, specified number is used.
        boto3_session:
            Boto3 Session. \
            The default boto3 session will be used if boto3_session receive None.
        max_cache_seconds: \
            awswrangler can look up in Athena's history \
            if this query has been run before.
            If so, and its completion time is less than `max_cache_seconds` \
            before now, awswrangler skips query execution \
            and just returns the same results as last time. \
            If cached results are valid, awswrangler ignores \
            the `ctas_approach`, `s3_output`, `encryption`, `kms_key`, \
            `keep_files` and `ctas_temp_table_name` params. \
            If reading cached data fails for any reason, \
            execution falls back to the usual query run path.
        max_cache_query_inspections: \
            Max number of queries that will be inspected from \
            the history to try to find some result to reuse. \
            The bigger the number of inspection, \
            the bigger will be the latency for not cached queries. \
            Only takes effect if max_cache_seconds > 0.
        max_remote_cache_entries: \
            Max number of queries that will be retrieved \
            from AWS for cache inspection. \
            The bigger the number of inspection, \
            the bigger will be the latency for not cached queries. \
            Only takes effect if max_cache_seconds > 0 and default value is 50.
        max_local_cache_entries:
            Max number of queries for which metadata will be cached locally. \
            This will reduce the latency and also enables keeping more than \
            `max_remote_cache_entries` available for the cache. \
            This value should not be smaller than max_remote_cache_entries. \
            Only takes effect if max_cache_seconds > 0 and default value is 100.
        data_source: \
            Data Source / Catalog name. \
            If None, 'AwsDataCatalog' will be used by default.
        params: \
            Dict of parameters that will be used for constructing the SQL query. \
            Only named parameters are supported. \
            The dict needs to contain the information in the form {'name': 'value'} \
            and the SQL query needs to contain \
            `:name;`. Note that for varchar columns and similar, \
            you must surround the value in single quotes.
        jinja_params: \
            Dict of Jinja rendering parameters.
        jinja_options: \
            Dict of Jinja template options.
        s3_additional_kwargs: \
            Forwarded to botocore requests. \
            e.g. s3_additional_kwargs={'RequestPayer': 'requester'}
        pyarrow_additional_kwargs: \
            Forward to the ParquetFile class or converting an Arrow table to Pandas, \
            currently only an "coerce_int96_timestamp_unit" or "timestamp_as_object" \
            argument will be considered. If reading parquet \
            files where you cannot convert a timestamp to pandas Timestamp[ns] \
            consider setting timestamp_as_object=True, \
            to allow for timestamp units larger than "ns". \
            If reading parquet data that still uses INT96 (like Athena outputs) \
            you can use coerce_int96_timestamp_unit to specify \
            what timestamp unit to encode INT96 to (by default this is "ns", \
            if you know the output parquet came \
            from a system that encodes timestamp to a particular unit \
            then set this to that same unit e.g. coerce_int96_timestamp_unit="ms").
        max_workers: \
            Number of parallelism executing CTAS result movement.

    Returns:
        DataFrame | Iterator[DataFrame]: \
            Pandas DataFrame or Generator of Pandas DataFrames if chunksize is passed.
    """
    if Path(save_dir).protocol == "file":
        if not s3_output:
            raise ValueError(
                "If `save_dir` is local filesystem directory path, "
                "you must specify `s3_output`."
            )
        tempdir_prefix = s3_output
    else:
        tempdir_prefix = save_dir
    with mktempdir(prefix=tempdir_prefix) as tempdir:
        query_resuts = read_sql_query(
            sql=sql,
            database=database,
            ctas_approach=True,
            categories=categories,
            chunksize=True,
            s3_output=str(tempdir),
            workgroup=workgroup,
            encryption=encryption,
            kms_key=kms_key,
            keep_files=True,
            ctas_database_name=ctas_database_name,
            ctas_bucketing_info=ctas_bucketing_info,
            use_threads=use_threads,
            boto3_session=boto3_session,
            max_cache_seconds=max_cache_seconds,
            max_cache_query_inspections=max_cache_query_inspections,
            max_remote_cache_entries=max_remote_cache_entries,
            max_local_cache_entries=max_local_cache_entries,
            data_source=data_source,
            jinja_params=jinja_params,
            jinja_options=jinja_options,
            params=params,
            s3_additional_kwargs=s3_additional_kwargs,
            pyarrow_additional_kwargs=pyarrow_additional_kwargs,
        )
        next(query_resuts)  # type: ignore
        move_file = partial(_move_file, dst=save_dir)
        sources = tempdir.glob("temp_table*/*")
        default_max_workers = min(32, mp.cpu_count() + 4)
        if not max_workers:
            max_workers_ = default_max_workers
        elif isinstance(max_workers, int):
            if max_workers < 0:
                max_workers_ = max(default_max_workers + max_workers, 1)
            else:
                max_workers_ = max_workers
        else:
            raise ValueError(
                f"`max_workers` must be integer. {repr(max_workers)} is not integer."
            )
        if 1 < max_workers_:
            with ThreadPoolExecutor(max_workers=max_workers_) as executor:
                for _ in executor.map(move_file, sources):
                    pass
        else:
            for src in sources:
                move_file(src)
