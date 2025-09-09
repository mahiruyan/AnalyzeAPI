import os
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

_engine: Optional[Engine] = None


def get_engine() -> Optional[Engine]:
    global _engine
    if _engine is not None:
        return _engine
    url = os.getenv("DATABASE_URL")
    if not url:
        return None
    _engine = create_engine(url, pool_pre_ping=True)
    return _engine


def init_schema() -> None:
    engine = get_engine()
    if engine is None:
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                create table if not exists analyses (
                  id varchar(64) primary key,
                  platform varchar(64),
                  status varchar(32),
                  created_at timestamptz default now(),
                  updated_at timestamptz default now(),
                  result jsonb,
                  error text
                );
                """
            )
        )
        conn.execute(
            text(
                """
                create table if not exists platform_metrics (
                  id bigserial primary key,
                  platform varchar(64) not null,
                  post_id varchar(128) not null,
                  metrics jsonb not null,
                  created_at timestamptz default now()
                );
                """
            )
        )


def save_analysis(id: str, platform: str, status: str, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
    engine = get_engine()
    if engine is None:
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                insert into analyses (id, platform, status, result, error)
                values (:id, :platform, :status, cast(:result as jsonb), :error)
                on conflict (id) do update set
                  status = excluded.status,
                  updated_at = now(),
                  result = excluded.result,
                  error = excluded.error
                """
            ),
            {
                "id": id,
                "platform": platform,
                "status": status,
                "result": (None if result is None else json_dumps(result)),
                "error": error,
            },
        )


def save_platform_metrics(platform: str, post_id: str, metrics: Dict[str, Any]) -> None:
    engine = get_engine()
    if engine is None:
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                insert into platform_metrics (platform, post_id, metrics)
                values (:platform, :post_id, cast(:metrics as jsonb))
                """
            ),
            {"platform": platform, "post_id": post_id, "metrics": json_dumps(metrics)},
        )


def json_dumps(data: Dict[str, Any]) -> str:
    import json
    return json.dumps(data, ensure_ascii=False)


