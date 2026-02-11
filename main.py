#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# ----------------------------
# Optional better language detection
# ----------------------------
_LANGDETECT_AVAILABLE = False
try:
    from langdetect import detect_langs  # type: ignore
    from langdetect import DetectorFactory  # type: ignore

    DetectorFactory.seed = 0
    _LANGDETECT_AVAILABLE = True
except Exception:
    detect_langs = None  # type: ignore


# ----------------------------
# Common text cleanup/tokenize
# ----------------------------
URL_RE = re.compile(r"https?://\S+")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
INLINE_CODE_RE = re.compile(r"`[^`]*`")
WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)  # unicode letters only

def clean_for_lang(text: str) -> str:
    text = URL_RE.sub(" ", text)
    text = MD_LINK_RE.sub(r"\1", text)
    text = INLINE_CODE_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_words(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text)]

def split_into_chunks(text: str) -> List[str]:
    chunks: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+", line)
        for p in parts:
            p = p.strip()
            if p:
                chunks.append(p)
    return chunks


# ----------------------------
# Heuristic scoring (HU vs EN)
# ----------------------------
HU_DIACRITICS = set("áéíóöőúüűÁÉÍÓÖŐÚÜŰ")

EN_STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","when","while","for","to","of","in","on","at","by","with",
    "as","is","are","was","were","be","been","being","do","does","did","doing","have","has","had","having",
    "i","you","he","she","it","we","they","me","him","her","us","them","my","your","his","their","our",
    "this","that","these","those","there","here","what","which","who","whom","why","how",
    "not","no","yes","so","very","just","also","too","more","most","less","least",
    "can","could","would","should","may","might","must","will","shall",
    "from","into","about","over","under","again","because",
}

HU_STOPWORDS = {
    "a","az","és","vagy","de","ha","akkor","mert","hogy","mint","is","sem","se","nem","igen","nincs","van","volt","lesz",
    "én","te","ő","mi","ti","ők","engem","nekem","tőlem","velem","veled","vele","velünk","veletek",
    "azt","ezt","itt","ott","ide","oda","innen","onnan","amikor","ahol","ami","aki","akik","mely","melyik","miért","hogyan",
    "már","még","csak","nagyon","túl","sok","kevés","kicsit","kb","szóval","persze",
    "meg","rá","le","fel","be","ki","el","át","össze","szét",
}

def _hu_ratio_heuristic(text: str) -> float:
    words = tokenize_words(text)
    if not words:
        return 0.0

    hu_sw = 0
    en_sw = 0
    hu_diac = 0

    for w in words:
        if w in HU_STOPWORDS:
            hu_sw += 1
        if w in EN_STOPWORDS:
            en_sw += 1
        if any(ch in HU_DIACRITICS for ch in w):
            hu_diac += 1

    hu_score = hu_sw + 1.5 * hu_diac
    en_score = en_sw
    unknown = len(words) - (hu_sw + en_sw)

    if hu_score >= en_score:
        hu_assigned = hu_score + 0.8 * unknown
    else:
        hu_assigned = hu_score + 0.2 * unknown

    return max(0.0, min(1.0, hu_assigned / len(words)))

def _lang_ratio_langdetect(text: str, lang_code: str) -> Optional[float]:
    if not _LANGDETECT_AVAILABLE or detect_langs is None:
        return None

    chunks = split_into_chunks(text)
    if not chunks:
        return 0.0

    lang_words = 0
    total_words = 0

    for ch in chunks:
        wcount = len(tokenize_words(ch))
        if wcount == 0:
            continue

        try:
            langs = detect_langs(ch)
        except Exception:
            continue

        if not langs:
            continue

        top = langs[0]
        total_words += wcount

        if getattr(top, "lang", "") == lang_code and getattr(top, "prob", 0.0) >= 0.70:
            lang_words += wcount

    if total_words == 0:
        return 0.0
    return lang_words / total_words

def hungarian_ratio(text: str, force_heuristic: bool = False) -> float:
    text = clean_for_lang(text)
    if not text:
        return 0.0

    if not force_heuristic:
        r = _lang_ratio_langdetect(text, "hu")
        if r is not None:
            return r

    return _hu_ratio_heuristic(text)

def english_ratio(text: str, force_heuristic: bool = False) -> float:
    # kept for non-subreddits mode compatibility (delete mostly EN)
    text = clean_for_lang(text)
    if not text:
        return 0.0

    if not force_heuristic:
        r = _lang_ratio_langdetect(text, "en")
        if r is not None:
            return r

    # simple "inverse-ish" heuristic: reuse HU heuristic + EN stopwords
    words = tokenize_words(text)
    if not words:
        return 0.0

    en_sw = sum(1 for w in words if w in EN_STOPWORDS)
    hu_sw = sum(1 for w in words if w in HU_STOPWORDS)
    hu_diac = sum(1 for w in words if any(ch in HU_DIACRITICS for ch in w))

    hu_score = hu_sw + 1.5 * hu_diac
    en_score = en_sw
    unknown = len(words) - (en_sw + hu_sw)
    if en_score > hu_score:
        en_assigned = en_score + 0.8 * unknown
    else:
        en_assigned = en_score + 0.2 * unknown

    return max(0.0, min(1.0, en_assigned / len(words)))


# ----------------------------
# Utilities
# ----------------------------
@dataclass
class Decision:
    delete: bool
    ratio: float
    preview: str
    kind: str

def make_preview(s: str, n: int = 140) -> str:
    s = s.strip().replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    return (s[:n] + "...") if len(s) > n else s

def unique_backup_path(path: Path) -> Path:
    base = path.with_suffix(path.suffix + ".bak")
    if not base.exists():
        return base
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_suffix(path.suffix + f".bak_{ts}")


# ============================================================
# MODE 1: "Comment:/Post:" exports (your original format)
# ============================================================
ENTRY_START_RE = re.compile(r"^(Comment|Post)\s*:\s*$")
FIELD_RE = re.compile(r"^\s{2}([A-Za-z0-9_]+):\s*(.*)$")

def split_segments_userexport(text: str) -> List[Tuple[str, str, Optional[str]]]:
    lines = text.splitlines(keepends=True)
    segments: List[Tuple[str, str, Optional[str]]] = []

    cur: List[str] = []
    cur_type: Optional[str] = None

    for line in lines:
        if ENTRY_START_RE.match(line) and line.lstrip() == line:
            if cur:
                if cur_type is None:
                    segments.append(("text", "".join(cur), None))
                else:
                    segments.append(("entry", cur_type, "".join(cur)))
                cur = []
            cur_type = ENTRY_START_RE.match(line).group(1)
            cur.append(line)
        else:
            cur.append(line)

    if cur:
        if cur_type is None:
            segments.append(("text", "".join(cur), None))
        else:
            segments.append(("entry", cur_type, "".join(cur)))

    return segments

def extract_multiline_field(entry_text: str, field: str) -> str:
    lines = entry_text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(rf"^\s{{2}}{re.escape(field)}:\s*(.*)$", line)
        if not m:
            continue

        rest = m.group(1)
        if rest.strip():
            return rest.strip()

        collected: List[str] = []
        j = i + 1
        while j < len(lines):
            l = lines[j]
            if FIELD_RE.match(l) and l.startswith("  "):
                break
            if ENTRY_START_RE.match(l) and l.lstrip() == l:
                break

            if l.startswith("    "):
                collected.append(l[4:])
            else:
                collected.append(l.lstrip())
            j += 1

        return "\n".join(collected).rstrip()
    return ""

def decide_userexport(entry_type: str, entry_text: str, threshold: float, force_heuristic: bool) -> Decision:
    title = extract_multiline_field(entry_text, "title")
    body = extract_multiline_field(entry_text, "body")

    if entry_type == "Post":
        judge_text = (title + "\n" + body).strip()
    else:
        judge_text = body.strip()

    if not judge_text or judge_text.strip().lower() in {"[removed]", "[deleted]"}:
        return Decision(False, 0.0, make_preview(judge_text), kind=entry_type)

    r_en = english_ratio(judge_text, force_heuristic=force_heuristic)
    return Decision(r_en >= threshold, r_en, make_preview(judge_text), kind=entry_type)

def process_file_userexport(
    in_path: Path,
    out_path: Path,
    threshold: float,
    show_deleted: bool,
    ask: bool,
    dry_run: bool,
    force_heuristic: bool,
) -> Tuple[int, int]:
    raw = in_path.read_text(encoding="utf-8", errors="replace")
    segments = split_segments_userexport(raw)

    kept_parts: List[str] = []
    total_entries = 0
    deleted_entries = 0

    for kind, a, b in segments:
        if kind == "text":
            kept_parts.append(a)
            continue

        entry_type = a
        entry_text = b or ""
        total_entries += 1

        dec = decide_userexport(entry_type, entry_text, threshold, force_heuristic)

        do_delete = dec.delete
        if do_delete and ask:
            print(f"\n[{in_path.name}] Candidate: {dec.kind} | en_ratio={dec.ratio:.2f}")
            print(f"Preview: {dec.preview}")
            ans = input("Delete this entry? [y/N] ").strip().lower()
            do_delete = (ans == "y")

        if do_delete:
            deleted_entries += 1
            if show_deleted:
                print(f"[DELETED] {in_path.name} | {dec.kind} | en_ratio={dec.ratio:.2f} | {dec.preview}")
        else:
            kept_parts.append(entry_text)

    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("".join(kept_parts), encoding="utf-8")

    return total_entries, deleted_entries


# ============================================================
# MODE 2: Subreddit dump files (=== r/... ===, Post:, comment:)
# ============================================================
SUBREDDIT_HEADER_RE = re.compile(r"^===\s*r/.+\s*===\s*$")
POST_START_RE = re.compile(r"^Post\s*:\s*$")
COMMENT_START_RE = re.compile(r"^\s{2}comment\s*:\s*$")
BYLINE_RE = re.compile(r"^by\s+([^:]+):\s*(.*)\s*$")
BODY_START_RE = re.compile(r"^\s{2}body\s*:\s*$")

def split_posts_subreddit(text: str) -> List[Tuple[str, str]]:
    """
    Returns list of segments: ("text", raw) or ("post", raw_post_block)
    """
    lines = text.splitlines(keepends=True)
    segs: List[Tuple[str, str]] = []

    cur: List[str] = []
    in_post = False

    def flush():
        nonlocal cur, in_post
        if not cur:
            return
        segs.append(("post" if in_post else "text", "".join(cur)))
        cur = []

    for line in lines:
        if POST_START_RE.match(line) and line.lstrip() == line:
            flush()
            in_post = True
            cur.append(line)
            continue

        # if a subreddit header appears, close current post (rare, but safe)
        if SUBREDDIT_HEADER_RE.match(line) and line.lstrip() == line and in_post:
            flush()
            in_post = False
            cur.append(line)
            continue

        cur.append(line)

        # If we are not in_post, keep accumulating text; if in_post, we just keep going until next Post: or header or EOF.

    flush()
    return segs

def split_post_into_pre_and_comments(post_block: str) -> Tuple[str, List[str]]:
    lines = post_block.splitlines(keepends=True)
    comment_idxs = [i for i, l in enumerate(lines) if COMMENT_START_RE.match(l)]
    if not comment_idxs:
        return post_block, []

    pre = "".join(lines[:comment_idxs[0]])
    comments: List[str] = []
    for idx_i, start in enumerate(comment_idxs):
        end = comment_idxs[idx_i + 1] if idx_i + 1 < len(comment_idxs) else len(lines)
        comments.append("".join(lines[start:end]))
    return pre, comments

def extract_subreddit_post_text(pre_block: str) -> str:
    """
    pre_block contains:
      Post:
      by user: TITLE
      (optional)  body:
        ....
    """
    lines = pre_block.splitlines()
    title = ""
    body_lines: List[str] = []

    # title from by-line
    for l in lines:
        m = BYLINE_RE.match(l.strip())
        if m:
            title = m.group(2).strip()
            break

    # body
    for i, l in enumerate(lines):
        if BODY_START_RE.match(l):
            j = i + 1
            while j < len(lines):
                if COMMENT_START_RE.match(lines[j]):
                    break
                # body lines are usually indented 4 spaces
                bl = lines[j]
                if bl.startswith("    "):
                    body_lines.append(bl[4:])
                else:
                    body_lines.append(bl.lstrip())
                j += 1
            break

    body = "\n".join(body_lines).rstrip()
    return (title + "\n" + body).strip()

def extract_subreddit_comment_text(comment_block: str) -> str:
    """
    comment_block:
      '  comment:\n'
      '    username: text...\n'
      '      continuation...\n'
    """
    lines = comment_block.splitlines()
    # find first content line after "  comment:"
    content_lines: List[str] = []
    started = False

    for l in lines:
        if COMMENT_START_RE.match(l):
            started = True
            continue
        if not started:
            continue

        # strip indentation (at least 4 spaces usually)
        raw = l
        if raw.startswith("    "):
            raw = raw[4:]
        else:
            raw = raw.lstrip()

        if not content_lines:
            # first line often: "user: text"
            if ":" in raw:
                _, txt = raw.split(":", 1)
                content_lines.append(txt.strip())
            else:
                content_lines.append(raw.strip())
        else:
            content_lines.append(raw.rstrip())

    return "\n".join([x for x in content_lines if x is not None]).strip()

def decide_subreddit_post(pre_block: str, threshold_hu: float, force_heuristic: bool) -> Decision:
    judge_text = extract_subreddit_post_text(pre_block)
    if not judge_text or judge_text.strip().lower() in {"[removed]", "[deleted]"}:
        return Decision(False, 0.0, make_preview(judge_text), kind="Post")

    r_hu = hungarian_ratio(judge_text, force_heuristic=force_heuristic)
    # Subreddits mode: delete if HU ratio below threshold
    return Decision(r_hu < threshold_hu, r_hu, make_preview(judge_text), kind="Post")

def decide_subreddit_comment(comment_block: str, threshold_hu: float, force_heuristic: bool) -> Decision:
    judge_text = extract_subreddit_comment_text(comment_block)
    if not judge_text or judge_text.strip().lower() in {"[removed]", "[deleted]"}:
        return Decision(False, 0.0, make_preview(judge_text), kind="Comment")

    r_hu = hungarian_ratio(judge_text, force_heuristic=force_heuristic)
    return Decision(r_hu < threshold_hu, r_hu, make_preview(judge_text), kind="Comment")

def process_file_subreddits(
    in_path: Path,
    out_path: Path,
    threshold_hu: float,
    show_deleted: bool,
    ask: bool,
    dry_run: bool,
    force_heuristic: bool,
) -> Tuple[int, int]:
    raw = in_path.read_text(encoding="utf-8", errors="replace")
    segs = split_posts_subreddit(raw)

    kept: List[str] = []
    total = 0
    deleted = 0

    for kind, block in segs:
        if kind == "text":
            kept.append(block)
            continue

        # post block
        pre, comments = split_post_into_pre_and_comments(block)
        total += 1

        post_dec = decide_subreddit_post(pre, threshold_hu, force_heuristic)
        do_delete_post = post_dec.delete

        if do_delete_post and ask:
            print(f"\n[{in_path.name}] Candidate POST | hu_ratio={post_dec.ratio:.2f}")
            print(f"Preview: {post_dec.preview}")
            ans = input("Delete this post (and its comments)? [y/N] ").strip().lower()
            do_delete_post = (ans == "y")

        if do_delete_post:
            deleted += 1
            if show_deleted:
                print(f"[DELETED] {in_path.name} | Post | hu_ratio={post_dec.ratio:.2f} | {post_dec.preview}")
            continue

        # keep post header/body
        kept.append(pre)

        # filter comments inside kept post
        for c in comments:
            total += 1
            c_dec = decide_subreddit_comment(c, threshold_hu, force_heuristic)
            do_delete_c = c_dec.delete

            if do_delete_c and ask:
                print(f"\n[{in_path.name}] Candidate COMMENT | hu_ratio={c_dec.ratio:.2f}")
                print(f"Preview: {c_dec.preview}")
                ans = input("Delete this comment? [y/N] ").strip().lower()
                do_delete_c = (ans == "y")

            if do_delete_c:
                deleted += 1
                if show_deleted:
                    print(f"[DELETED] {in_path.name} | Comment | hu_ratio={c_dec.ratio:.2f} | {c_dec.preview}")
            else:
                kept.append(c)

    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("".join(kept), encoding="utf-8")

    return total, deleted


# ----------------------------
# File iteration & main
# ----------------------------
def iter_files(inputfolder: Path, recursive: bool, pattern: str) -> List[Path]:
    return sorted(inputfolder.rglob(pattern) if recursive else inputfolder.glob(pattern))

def main() -> int:
    p = argparse.ArgumentParser(description="Filter Reddit exports: remove mostly-English entries or keep mostly-HU in subreddit dumps.")
    p.add_argument("-inputfolder", "--inputfolder", required=True, help="Folder that contains exported .txt files.")
    p.add_argument("--pattern", default="*.txt", help="Glob pattern for files (default: *.txt).")
    p.add_argument("--recursive", action="store_true", help="Process subfolders too.")
    p.add_argument("--threshold", type=float, default=0.90, help="Threshold (see --subreddits note).")

    p.add_argument("--show-deleted", action="store_true", help="Print deletions continuously to console.")
    p.add_argument("--ask", action="store_true", help="Ask confirmation for every candidate deletion (type 'y' to delete).")
    p.add_argument("--dry-run", action="store_true", help="Do not write files, only report what would be deleted.")

    p.add_argument("--inplace", action="store_true",
                   help="Overwrite files in-place (creates a .bak backup next to each file).")
    p.add_argument("--outputfolder", default=None,
                   help="Output folder for cleaned files (default: create '<inputfolder>/cleaned').")

    p.add_argument("--force-heuristic", action="store_true",
                   help="Do not use langdetect even if installed (use heuristic only).")

    p.add_argument("--subreddits", action="store_true",
                   help="Enable subreddit-dump mode (=== r/... ===, Post:, comment:). In this mode, --threshold means MIN HU ratio to KEEP (below it -> delete).")

    args = p.parse_args()

    in_dir = Path(args.inputfolder).expanduser().resolve()
    if not in_dir.exists() or not in_dir.is_dir():
        print(f"ERROR: inputfolder does not exist or not a directory: {in_dir}")
        return 2

    out_dir = in_dir if args.inplace else (Path(args.outputfolder).expanduser().resolve() if args.outputfolder else (in_dir / "cleaned"))

    files = iter_files(in_dir, args.recursive, args.pattern)
    if not files:
        print(f"No files matched: {in_dir} / {args.pattern}")
        return 0

    detector_info = "langdetect" if (_LANGDETECT_AVAILABLE and not args.force_heuristic) else "heuristic"
    print(f"Detector: {detector_info} | threshold={args.threshold:.2f} | subreddits={args.subreddits}")
    print(f"Files: {len(files)} | mode={'inplace' if args.inplace else 'outputfolder'} | dry_run={args.dry_run}")

    grand_total = 0
    grand_deleted = 0

    for f in files:
        rel = f.relative_to(in_dir)
        out_path = (out_dir / rel) if not args.inplace else f

        if args.inplace and not args.dry_run:
            bak = unique_backup_path(f)
            shutil.copy2(f, bak)

        if args.subreddits:
            total, deleted = process_file_subreddits(
                in_path=f,
                out_path=out_path,
                threshold_hu=args.threshold,
                show_deleted=args.show_deleted,
                ask=args.ask,
                dry_run=args.dry_run,
                force_heuristic=args.force_heuristic,
            )
        else:
            total, deleted = process_file_userexport(
                in_path=f,
                out_path=out_path,
                threshold=args.threshold,
                show_deleted=args.show_deleted,
                ask=args.ask,
                dry_run=args.dry_run,
                force_heuristic=args.force_heuristic,
            )

        grand_total += total
        grand_deleted += deleted

    print(f"\nDone. Items total={grand_total}, deleted={grand_deleted}, kept={grand_total - grand_deleted}")
    if args.dry_run:
        print("Dry-run mode: no files were written.")
    else:
        if args.inplace:
            print("In-place mode: originals were overwritten (backups created as *.bak*).")
        else:
            print(f"Cleaned files written to: {out_dir}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
