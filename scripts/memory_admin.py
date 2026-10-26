from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def stats(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        "requests": int(connection.execute("SELECT COUNT(*) FROM add_requests").fetchone()[0]),
        "memories": int(connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]),
        "users": int(connection.execute("SELECT COUNT(DISTINCT user_id) FROM memories").fetchone()[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or purge the SQLite memory database")
    parser.add_argument("--db", default="artifacts/memory.db")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("stats")
    purge = commands.add_parser("purge-prefix", help="Delete all data whose user_id starts with a run prefix")
    purge.add_argument("--user-prefix", required=True)
    purge.add_argument("--confirm", action="store_true")
    args = parser.parse_args()

    path = Path(args.db)
    if not path.exists():
        raise SystemExit(f"Database does not exist: {path}")
    connection = sqlite3.connect(path, timeout=60)
    try:
        if args.command == "stats":
            print(json.dumps(stats(connection), indent=2))
            return
        if not args.confirm:
            raise SystemExit("Refusing to delete data without --confirm")
        request_ids = [
            row[0]
            for row in connection.execute(
                "SELECT request_id FROM add_requests WHERE substr(user_id, 1, ?) = ?",
                (len(args.user_prefix), args.user_prefix),
            )
        ]
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "DELETE FROM memories WHERE substr(user_id, 1, ?) = ?",
            (len(args.user_prefix), args.user_prefix),
        )
        connection.executemany("DELETE FROM add_requests WHERE request_id = ?", [(value,) for value in request_ids])
        connection.commit()
        print(json.dumps({"deleted_requests": len(request_ids), **stats(connection)}, indent=2))
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()
