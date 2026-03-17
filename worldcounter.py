#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


# Unicode-kompatibilis szófelismerés
# A kötőjeles és aposztrófos alakokat egy szónak veszi.
WORD_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)

TITLE_RE = re.compile(r"^title:\s*(.*)$", re.IGNORECASE)
BY_RE = re.compile(r"^by\s+[^:]+:\s*(.*)$", re.IGNORECASE)
ANY_PREFIX_RE = re.compile(r"^[^:]+:\s*(.*)$")


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def count_file_words(file_path: Path) -> int:
    try:
        lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as e:
        raise RuntimeError(f"Hiba olvasás közben: {file_path} -> {e}") from e

    if not lines:
        return 0

    total = 0
    after_comment_marker = False

    # Az első sort mindig kihagyjuk
    for raw_line in lines[1:]:
        stripped = raw_line.strip()

        if not stripped:
            continue

        lowered = stripped.lower()

        # Csak szerkezeti elemek, ezeket nem számoljuk
        if lowered == "post:":
            after_comment_marker = False
            continue

        if lowered == "comment:":
            after_comment_marker = True
            continue

        if lowered.startswith("subreddit:"):
            after_comment_marker = False
            continue

        if lowered == "body:":
            after_comment_marker = False
            continue

        # title: -> csak a kettőspont utáni rész számít
        m = TITLE_RE.match(stripped)
        if m:
            total += count_words(m.group(1))
            after_comment_marker = False
            continue

        # by username: -> csak a kettőspont utáni rész számít
        m = BY_RE.match(stripped)
        if m:
            total += count_words(m.group(1))
            after_comment_marker = False
            continue

        # comment: után a következő sorban username: szöveg
        # Itt csak a kettőspont utáni részt számoljuk
        if after_comment_marker:
            m = ANY_PREFIX_RE.match(stripped)
            if m:
                total += count_words(m.group(1))
            else:
                total += count_words(stripped)
            after_comment_marker = False
            continue

        # Minden más normál sor teljesen számít
        total += count_words(stripped)

    return total


def process_directory(input_dir: Path, output_file: Path) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"A megadott input mappa nem létezik: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"A megadott input útvonal nem mappa: {input_dir}")

    txt_files = sorted(input_dir.rglob("*.txt"))

    # Ha az output is a bejárt mappában van, ne számoljuk bele
    output_resolved = output_file.resolve()
    txt_files = [p for p in txt_files if p.resolve() != output_resolved]

    results = []
    grand_total = 0

    for file_path in txt_files:
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
        description="TXT fájlok szószámlálása speciális Post/Comment struktúrához."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="A bemeneti mappa, ahol a .txt fájlok vannak."
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