#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from datetime import datetime
from pathlib import Path


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def detect_language(text: str) -> str:
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

    # pl. r/u_Kaiokensanpaida -> u/Kaiokensanpaida
    if value.startswith("r/u_"):
        return "u/" + value[len("r/u_"):]
    return value


def extract_username_from_filename(filename: str) -> str:
    stem = Path(filename).stem

    if stem.endswith("_chats"):
        return stem[:-6]
    if stem.endswith("_posts"):
        return stem[:-6]

    return stem


def prettify_username(username: str) -> str:
    special = {
        "izzystraveldiaries": "izzystraveldiaries",
        "kaiokensanpaida": "Kaiokensanpaida",
    }
    return special.get(username.lower(), username)


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
        "date_modified": now_iso(),
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


def strip_common_indent(lines: list[str]) -> str:
    # szélső üres sorok levágása
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return ""

    min_indent = None
    for line in lines:
        if line.strip():
            leading = len(line) - len(line.lstrip(" "))
            if min_indent is None or leading < min_indent:
                min_indent = leading

    if min_indent is None:
        min_indent = 0

    normalized = []
    for line in lines:
        if line.strip():
            normalized.append(line[min_indent:])
        else:
            normalized.append("")

    return "\n".join(normalized).strip()


def split_top_level_blocks(text: str, marker: str) -> list[list[str]]:
    """
    marker = "Comment:" vagy "Post:"
    A fájlt top-level blokkokra bontja.
    Csak a sor elején álló marker számít új blokknak.
    """
    lines = text.splitlines()
    blocks = []
    current = None

    for line in lines:
        if line.strip() == marker:
            if current:
                blocks.append(current)
            current = [line]
        else:
            if current is not None:
                current.append(line)

    if current:
        blocks.append(current)

    return blocks


def detect_file_type(text: str, filename: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "unknown"

    has_comment_marker = "\nComment:" in "\n" + stripped or stripped.startswith("Comment:")
    has_post_marker = "\nPost:" in "\n" + stripped or stripped.startswith("Post:")
    has_by_lines = "\nby " in "\n" + stripped or stripped.startswith("by ")

    # 1) csak kommentek
    if has_comment_marker and not has_post_marker:
        return "comment_only"

    # 2) posztok kommentekkel
    if has_post_marker and has_by_lines:
        return "post_with_comments"

    # 3) csak posztok
    if has_post_marker:
        return "post_only"

    # fallback fájlnév alapján
    lower_name = filename.lower()
    if lower_name.endswith("_chats.txt"):
        return "comment_only"
    if lower_name.endswith("_posts.txt"):
        return "post_only"

    return "unknown"


def parse_comment_block(block_lines: list[str], username: str) -> dict | None:
    """
    Formátum:
    Comment:
      subreddit: r/...
      body:
        szöveg...
    """
    subreddit = None
    body_lines = []
    in_body = False

    for i, raw in enumerate(block_lines[1:], start=1):  # az első sor maga a Comment:
        stripped = raw.strip()

        if stripped.startswith("subreddit:"):
            subreddit = stripped.split(":", 1)[1].strip()
            continue

        if stripped == "body:":
            in_body = True
            continue

        if in_body:
            body_lines.append(raw)

    body = strip_common_indent(body_lines)

    if subreddit is None and not body:
        return None

    obj = base_post_object()
    obj["title"] = ""
    obj["authors"] = make_author(username)
    obj["data"] = body
    obj["comments"] = []
    obj["language"] = detect_language(body)
    obj["extra"] = {
        "source_type": "comment",
        "subreddit": normalize_subreddit(subreddit or "")
    }

    return obj


def parse_comment_only_file(text: str, filename: str) -> list:
    username = prettify_username(extract_username_from_filename(filename))
    blocks = split_top_level_blocks(text, "Comment:")
    results = []

    for block in blocks:
        obj = parse_comment_block(block, username)
        if obj is not None:
            results.append(obj)

    return results


def parse_post_only_block(block_lines: list[str], username: str) -> dict | None:
    """
    Formátum:
    Post:
      subreddit: ...
      title: ...
      body:
        ...
    """
    subreddit = None
    title = ""
    body_lines = []
    in_body = False

    for raw in block_lines[1:]:
        stripped = raw.strip()

        if stripped.startswith("subreddit:"):
            subreddit = stripped.split(":", 1)[1].strip()
            continue

        if stripped.startswith("title:"):
            title = stripped.split(":", 1)[1].strip()
            continue

        if stripped == "body:":
            in_body = True
            continue

        if in_body:
            body_lines.append(raw)

    body = strip_common_indent(body_lines)

    if subreddit is None and not title and not body:
        return None

    obj = base_post_object()
    obj["title"] = title
    obj["authors"] = make_author(username)
    obj["data"] = body
    obj["comments"] = []
    obj["language"] = detect_language(title + "\n" + body)
    obj["extra"] = {
        "source_type": "post",
        "subreddit": normalize_subreddit(subreddit or "")
    }

    return obj


def parse_post_only_file(text: str, filename: str) -> list:
    username = prettify_username(extract_username_from_filename(filename))
    blocks = split_top_level_blocks(text, "Post:")
    results = []

    for block in blocks:
        obj = parse_post_only_block(block, username)
        if obj is not None:
            results.append(obj)

    return results


def build_comment(author: str, text: str) -> dict:
    c = base_comment_object()
    c["data"] = text.strip()
    c["language"] = detect_language(text)
    c["extra"] = {
        "author": author.strip()
    }
    return c


def infer_subreddit_from_filename(filename: str) -> str:
    stem = Path(filename).stem.lower()

    if stem == "budapest":
        return "r/budapest"

    username = extract_username_from_filename(filename)
    if username:
        pretty = prettify_username(username)
        return normalize_subreddit(f"u/{pretty}")

    return ""


def parse_post_with_comments_block(block_lines: list[str], filename: str) -> dict | None:
    """
    Formátum kb:
    Post:
    by USER: Cím
      body:
        ...
      comment:
        USER2: ...
        ...
      comment:
        USER3: ...
    """

    title = ""
    author = ""
    body_lines = []
    comments = []
    visited = None

    mode = None  # None | "body" | "comment"
    current_comment_author = None
    current_comment_lines = []

    def flush_comment():
        nonlocal current_comment_author, current_comment_lines, comments
        if current_comment_author is not None:
            text = "\n".join(current_comment_lines).strip()
            comments.append(build_comment(current_comment_author, text))
            current_comment_author = None
            current_comment_lines = []

    for raw in block_lines[1:]:  # első sor a Post:
        stripped = raw.strip()

        if not stripped:
            if mode == "body":
                body_lines.append("")
            elif mode == "comment" and current_comment_author is not None:
                current_comment_lines.append("")
            continue

        if stripped.startswith("by "):
            # pl. by Rude_Ant_9007: Kávé mellett munka?
            rest = stripped[3:]
            if ":" in rest:
                author_part, title_part = rest.split(":", 1)
                author = author_part.strip()
                title = title_part.strip()
            else:
                author = rest.strip()
                title = ""
            mode = None
            continue

        if stripped.lower().startswith("visited:"):
            visited = stripped.split(":", 1)[1].strip()
            continue

        if stripped == "body:":
            flush_comment()
            mode = "body"
            continue

        if stripped == "comment:":
            flush_comment()
            mode = "comment"
            continue

        if mode == "body":
            body_lines.append(raw)
            continue

        if mode == "comment":
            if current_comment_author is None:
                # első komment sor: username: szöveg
                if ":" in stripped:
                    c_author, c_text = stripped.split(":", 1)
                    current_comment_author = c_author.strip()
                    first_text = c_text.strip()
                    current_comment_lines = [first_text] if first_text else []
                else:
                    current_comment_author = "unknown"
                    current_comment_lines = [stripped]
            else:
                current_comment_lines.append(raw)

    flush_comment()

    body = strip_common_indent(body_lines)

    if not author and not title and not body and not comments:
        return None

    obj = base_post_object()
    obj["title"] = title
    obj["authors"] = make_author(author if author else prettify_username(extract_username_from_filename(filename)))
    obj["data"] = body
    obj["comments"] = comments
    obj["language"] = detect_language(title + "\n" + body)

    extra = {
        "source_type": "post_with_comments",
        "subreddit": infer_subreddit_from_filename(filename)
    }
    if visited:
        extra["visited"] = visited

    obj["extra"] = extra
    return obj


def parse_post_with_comments_file(text: str, filename: str) -> list:
    blocks = split_top_level_blocks(text, "Post:")
    results = []

    for block in blocks:
        obj = parse_post_with_comments_block(block, filename)
        if obj is not None:
            results.append(obj)

    return results


def process_file(input_path: Path) -> list:
    text = input_path.read_text(encoding="utf-8")
    file_type = detect_file_type(text, input_path.name)

    if file_type == "comment_only":
        return parse_comment_only_file(text, input_path.name)

    if file_type == "post_only":
        return parse_post_only_file(text, input_path.name)

    if file_type == "post_with_comments":
        return parse_post_with_comments_file(text, input_path.name)

    print(f"[WARN] Ismeretlen formátum, kihagyva: {input_path.name}")
    return []


def save_json(output_path: Path, data: list) -> None:
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Reddit jellegű txt fájlok konvertálása egységes forum_post JSON formátumba."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Bemeneti mappa"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Kimeneti mappa"
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

    for idx, input_file in enumerate(txt_files, start=1):
        print(f"[{idx}/{len(txt_files)}] Feldolgozás: {input_file.name}")

        try:
            converted = process_file(input_file)
            output_file = output_dir / f"{input_file.stem}.json"
            save_json(output_file, converted)
            print(f"    Kész: {output_file.name} | rekordok száma: {len(converted)}")
        except Exception as e:
            print(f"    HIBA: {input_file.name} -> {e}")

    print("Feldolgozás befejezve.")


if __name__ == "__main__":
    main()
    
#python jsonconverter.py --input ./input --output ./output