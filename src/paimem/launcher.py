"""Launch PaiMem from a unified TOML configuration."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from .configuration import boolean, default_path, prepare
from .database_check import check_database


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_path())
    parser.add_argument("--check", action="store_true", help="Validate configuration and database compatibility without loading models or calling endpoints")
    args = parser.parse_args(argv)
    try:
        service, env = prepare(args.config)
        database_status = check_database(env)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.check:
        print(json.dumps({"valid": True, "config": str(args.config), "embedding_backend": env["EMBEDDING_BACKEND"], "reranker": service.get("reranker", True), "database_status": database_status}, indent=2))
        return
    os.environ.update(env)
    if "EMBEDDING_QUERY_PREFIX" not in env:
        os.environ.pop("EMBEDDING_QUERY_PREFIX", None)
    if not service.get("reranker", True):
        os.environ.pop("MEMORY_RERANKER_MODEL", None)
    import uvicorn
    uvicorn.run("paimem.api:app", host=service.get("host", "127.0.0.1"), port=service.get("port", 8000), workers=1)

if __name__ == "__main__":
    main()
