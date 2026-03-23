#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


# Unicode-kompatibilis szófelismerés
# A kötőjeles és aposztrófos alakokat egy szónak veszi.
WORD_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def count_value_words(value: Any) -> int:
    total = 0

    if value is None:
        return 0

    if isinstance(value, str):
        return count_words(value)

    if isinstance(value, list):
        for item in value:
            total += count_value_words(item)
        return total

    if isinstance(value, dict):
        for v in value.values():
            total += count_value_words(v)
        return total

    return 0


def count_json_words(data: Any) -> int:
    total = 0

    if isinstance(data, dict):
        for key, value in data.items():
            key_lower = str(key).lower()

            if key_lower in {"data", "title", "comments"}:
                total += count_value_words(value)
            else:
                total += count_json_words(value)

    elif isinstance(data, list):
        for item in data:
            total += count_json_words(item)

    return total


def count_file_words(file_path: Path) -> int:
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        raise RuntimeError(f"Hiba olvasás közben: {file_path} -> {e}") from e

    try:
        parsed = json.loads(content)
    except Exception as e:
        raise RuntimeError(f"Hibás JSON: {file_path} -> {e}") from e

    return count_json_words(parsed)


def process_directory(input_dir: Path, output_file: Path) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"A megadott input mappa nem létezik: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"A megadott input útvonal nem mappa: {input_dir}")

    json_files = sorted(input_dir.rglob("*.json"))

    # Ha az output is a bejárt mappában van, ne számoljuk bele
    output_resolved = output_file.resolve()
    json_files = [p for p in json_files if p.resolve() != output_resolved]

    results = []
    grand_total = 0

    for file_path in json_files:
        word_count = count_file_words(file_path)
        rel_name = file_path.relative_to(input_dir).as_posix()
        results.append((rel_name, word_count))
        grand_total += word_count

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="\n") as f:
        for filename, word_count in results:
            f.write(f"{filename}\t{word_count}\n")

        # Két üres sor, majd az összegzés
        f.write("\n\n")
        f.write(f"szum:\t{grand_total}\n")

    print(f"Feldolgozott fájlok száma: {len(results)}")
    print(f"Összes szó: {grand_total}")
    print(f"Eredményfájl: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="JSON fájlok szószámlálása a data, comments és title mezők alapján."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="A bemeneti mappa, ahol a .json fájlok vannak."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Az eredményfájl neve/útvonala. Alapértelmezés: <input>/counted.txt"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_file = Path(args.output) if args.output else input_dir / "counted.txt"

    process_directory(input_dir, output_file)


if __name__ == "__main__":
    main()