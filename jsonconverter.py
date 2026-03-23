#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from datetime import datetime
from pathlib import Path


def current_iso_datetime() -> str:
    return datetime.now().isoformat(timespec="seconds")


def detect_language(text: str) -> str:
    # Egyszerű alapértelmezés: magyar
    # Később bővíthető, ha kell
    return "hu"


def make_author(username: str) -> list:
    return [
        {
            "family": username,
            "given": ""
        }
    ]


def normalize_subreddit(value: str) -> str:
    value = (value or "").strip()

    # Speciális eset: r/u_Kaiokensanpaida -> u/Kaiokensanpaida
    m = re.fullmatch(r"r/u_(.+)", value, flags=re.IGNORECASE)
    if m:
        return f"u/{m.group(1)}"

    return value


def base_post_object() -> dict:
    return {
        "type": "forum_post",
        "title": "",
        "authors": [],
        "data": "",
        "comments": [],
        "likes": None,
        "dislikes": None,
        "score": None,
        "date": None,
        "url": None,
        "language": "hu",
        "tags": [],
        "rights": None,
        "date_modified": current_iso_datetime(),
        "extra": {},
        "origin": "reddit"
    }


def base_comment_object() -> dict:
    return {
        "data": "",
        "likes": None,
        "dislikes": None,
        "score": None,
        "date": None,
        "url": None,
        "language": "hu",
        "extra": {}
    }


def extract_username_from_filename(filename: str) -> str:
    name = Path(filename).stem

    # izzystraveldiaries_chats -> izzystraveldiaries
    if name.endswith("_chats"):
        return name[:-6]

    # kaiokensanpaida_posts -> kaiokensanpaida
    if name.endswith("_posts"):
        return name[:-6]

    return name


def prettify_username(username: str) -> str:
    # Speciális mapping az általad kért formákhoz
    special = {
        "izzystraveldiaries": "izzystraveldiaries",
        "kaiokensanpaida": "Kaiokensanpaida"
    }
    return special.get(username.lower(), username)


def parse_comment_only_file(text: str, filename: str) -> list:
    username = prettify_username(extract_username_from_filename(filename))

    pattern = re.compile(
        r"Comment:\s*\n\s*subreddit:\s*(?P<subreddit>.+?)\s*\n\s*body:\s*\n(?P<body>(?:\s+.*(?:\n|$))+)",
        re.MULTILINE
    )

    results = []

    for match in pattern.finditer(text):
        subreddit = match.group("subreddit").strip()
        body = dedent_block(match.group("body")).strip()

        obj = base_post_object()
        obj["title"] = ""
        obj["authors"] = make_author(username)
        obj["data"] = body
        obj["comments"] = []
        obj["language"] = detect_language(body)
        obj["extra"] = {
            "source_type": "comment",
            "subreddit": normalize_subreddit(subreddit)
        }

        results.append(obj)

    return results


def parse_post_only_file(text: str, filename: str) -> list:
    username = prettify_username(extract_username_from_filename(filename))

    # Post-only formátum:
    # Post:
    #   subreddit: ...
    #   title: ...
    #   body:
    #     ...
    pattern = re.compile(
        r"Post:\s*\n"
        r"\s*subreddit:\s*(?P<subreddit>.+?)\s*\n"
        r"\s*title:\s*(?P<title>.+?)\s*\n"
        r"\s*body:\s*\n"
        r"(?P<body>(?:\s+.*(?:\n|$))+?)(?=(?:\nPost:|\Z))",
        re.MULTILINE
    )

    results = []

    for match in pattern.finditer(text):
        subreddit = match.group("subreddit").strip()
        title = match.group("title").strip()
        body = dedent_block(match.group("body")).strip()

        obj = base_post_object()
        obj["title"] = title
        obj["authors"] = make_author(username)
        obj["data"] = body
        obj["comments"] = []
        obj["language"] = detect_language(f"{title}\n{body}")
        obj["extra"] = {
            "source_type": "post",
            "subreddit": normalize_subreddit(subreddit)
        }

        results.append(obj)

    return results


def parse_post_with_comments_file(text: str, filename: str) -> list:
    # Itt a tényleges author a blokkból jön: "by USER: Title"
    # Egy post blokk végét a következő "Post:" vagy EOF jelzi
    post_blocks = re.findall(
        r"Post:\s*\n(?P<content>.*?)(?=\nPost:\s*\n|\Z)",
        text,
        flags=re.DOTALL
    )

    results = []

    for block in post_blocks:
        parsed = parse_single_post_with_comments_block(block, filename)
        if parsed is not None:
            results.append(parsed)

    return results


def parse_single_post_with_comments_block(block: str, filename: str) -> dict | None:
    lines = block.splitlines()

    author = ""
    title = ""
    body_lines = []
    comments = []

    current_mode = None
    current_comment_author = None
    current_comment_lines = []

    visited_value = None

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        if not line.strip():
            if current_mode == "body":
                body_lines.append("")
            elif current_mode == "comment" and current_comment_author is not None:
                current_comment_lines.append("")
            continue

        stripped = line.strip()

        # Példák:
        # by Rude_Ant_9007: Kávé mellett munka?
        # by ahh_szellem: Where to buy good Christmas stocking?
        if stripped.startswith("by "):
            m = re.match(r"by\s+([^:]+):\s*(.*)", stripped)
            if m:
                author = m.group(1).strip()
                title = m.group(2).strip()
                current_mode = None
                continue

        # opcionális visited sor, ha lenne
        if stripped.lower().startswith("visited:"):
            visited_value = stripped.split(":", 1)[1].strip()
            continue

        if stripped.startswith("body:"):
            # ha épp kommentet gyűjtöttünk, zárjuk le
            if current_mode == "comment" and current_comment_author is not None:
                comments.append(build_comment(current_comment_author, "\n".join(current_comment_lines).strip()))
                current_comment_author = None
                current_comment_lines = []
            current_mode = "body"
            continue

        if stripped.startswith("comment:"):
            # ha előző komment volt nyitva, zárjuk le
            if current_mode == "comment" and current_comment_author is not None:
                comments.append(build_comment(current_comment_author, "\n".join(current_comment_lines).strip()))
                current_comment_author = None
                current_comment_lines = []

            current_mode = "comment"
            continue

        if current_mode == "comment":
            # sor mintája: username: szöveg
            if current_comment_author is None:
                m = re.match(r"\s*([^:]+):\s*(.*)", line)
                if m:
                    current_comment_author = m.group(1).strip()
                    first_text = m.group(2).strip()
                    current_comment_lines = [first_text] if first_text else []
                else:
                    # ha nincs author-forma, akkor unknown
                    current_comment_author = "unknown"
                    current_comment_lines = [stripped]
            else:
                current_comment_lines.append(stripped)

        elif current_mode == "body":
            body_lines.append(stripped)

    # utolsó nyitott komment lezárása
    if current_mode == "comment" and current_comment_author is not None:
        comments.append(build_comment(current_comment_author, "\n".join(current_comment_lines).strip()))

    # Ha se author, se title, se body nincs, akkor ezt a blokkot eldobjuk
    body = "\n".join(body_lines).strip()
    if not author and not title and not body and not comments:
        return None

    obj = base_post_object()
    obj["title"] = title
    obj["authors"] = make_author(author if author else prettify_username(extract_username_from_filename(filename)))
    obj["data"] = body
    obj["comments"] = comments
    obj["language"] = detect_language(f"{title}\n{body}")

    extra = {
        "source_type": "post_with_comments",
        "subreddit": infer_subreddit_from_filename(filename)
    }
    if visited_value:
        extra["visited"] = visited_value

    obj["extra"] = extra
    return obj


def build_comment(author: str, text: str) -> dict:
    c = base_comment_object()
    c["data"] = text
    c["language"] = detect_language(text)
    c["extra"] = {
        "author": author
    }
    return c


def infer_subreddit_from_filename(filename: str) -> str:
    stem = Path(filename).stem.lower()

    if stem == "budapest":
        return "r/budapest"

    username = extract_username_from_filename(filename)
    if username:
        return normalize_subreddit(f"u/{prettify_username(username)}")

    return ""


def dedent_block(text: str) -> str:
    lines = text.splitlines()

    # Eldobjuk a teljesen üres szélső sorokat
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return ""

    indents = []
    for line in lines:
        if line.strip():
            indents.append(len(line) - len(line.lstrip(" ")))

    min_indent = min(indents) if indents else 0
    return "\n".join(line[min_indent:] if len(line) >= min_indent else line for line in lines)


def detect_file_type(text: str, filename: str) -> str:
    stripped = text.strip()

    if not stripped:
        return "unknown"

    # csak kommentek
    if "Comment:" in stripped and "Post:" not in stripped and "comment:" not in stripped:
        return "comment_only"

    # csak posztok
    if "Post:" in stripped and "subreddit:" in stripped and "title:" in stripped and "by " not in stripped:
        return "post_only"

    # posztok kommentekkel
    if "Post:" in stripped and "by " in stripped:
        return "post_with_comments"

    # fájlnév alapján fallback
    lower_name = filename.lower()
    if lower_name.endswith("_chats.txt"):
        return "comment_only"
    if lower_name.endswith("_posts.txt"):
        return "post_only"

    return "unknown"


def process_file(input_path: Path) -> list:
    with input_path.open("r", encoding="utf-8") as f:
        text = f.read()

    file_type = detect_file_type(text, input_path.name)

    if file_type == "comment_only":
        return parse_comment_only_file(text, input_path.name)
    elif file_type == "post_only":
        return parse_post_only_file(text, input_path.name)
    elif file_type == "post_with_comments":
        return parse_post_with_comments_file(text, input_path.name)
    else:
        print(f"[WARN] Ismeretlen formátum, kihagyva: {input_path.name}")
        return []


def save_json(output_path: Path, data: list) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Reddit-szerű txt fájlok konvertálása egységes forum_post JSON sémára."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Bemeneti mappa elérési útja"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Kimeneti mappa elérési útja"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Hiba: a bemeneti mappa nem létezik vagy nem mappa: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        print("Nincs feldolgozható .txt fájl a bemeneti mappában.")
        return

    print(f"Összesen {len(txt_files)} fájl feldolgozása indul...")

    for index, input_file in enumerate(txt_files, start=1):
        print(f"[{index}/{len(txt_files)}] Feldolgozás: {input_file.name}")

        try:
            converted = process_file(input_file)

            output_file = output_dir / f"{input_file.stem}.json"
            save_json(output_file, converted)

            print(f"    Kész: {output_file.name} ({len(converted)} rekord)")
        except Exception as e:
            print(f"    HIBA {input_file.name} feldolgozása közben: {e}")

    print("Feldolgozás befejezve.")


if __name__ == "__main__":
    main()

#python jsonconverter.py --input ./input --output ./output