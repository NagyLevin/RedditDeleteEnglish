from __future__ import annotations

import argparse
import os
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
# Parsing helpers
# ----------------------------
ENTRY_START_RE = re.compile(r"^(Comment|Post)\s*:\s*$")
FIELD_RE = re.compile(r"^\s{2}([A-Za-z0-9_]+):\s*(.*)$")

URL_RE = re.compile(r"https?://\S+")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
INLINE_CODE_RE = re.compile(r"`[^`]*`")

WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)  # unicode letters only


def split_segments(text: str) -> List[Tuple[str, str, Optional[str]]]:
    """
    Splits input into segments:
      ("text", raw, None) for non-entry content (headers, etc.)
      ("entry", entry_type, raw_entry_text) for each Comment/Post entry
    """
    lines = text.splitlines(keepends=True)
    segments: List[Tuple[str, str, Optional[str]]] = []

    cur: List[str] = []
    cur_type: Optional[str] = None

    for line in lines:
        if ENTRY_START_RE.match(line) and line.lstrip() == line:
            # Start of a new entry
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
    """
    Extracts fields like 'subreddit', 'title', 'body'.
    Supports:
      title: single-line
      body:  multi-line, indented content under 'body:' line
    """
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

            # Stop at next field definition (2 spaces + key + :)
            if FIELD_RE.match(l) and l.startswith("  "):
                break

            # Stop if a new entry starts (shouldn't happen inside an entry, but safe)
            if ENTRY_START_RE.match(l) and l.lstrip() == l:
                break

            # Strip up to 4 leading spaces for body/title content lines
            if l.startswith("    "):
                collected.append(l[4:])
            else:
                collected.append(l.lstrip())

            j += 1

        return "\n".join(collected).rstrip()

    return ""


def clean_for_lang(text: str) -> str:
    # remove URLs, markdown link targets, inline code
    text = URL_RE.sub(" ", text)
    text = MD_LINK_RE.sub(r"\1", text)
    text = INLINE_CODE_RE.sub(" ", text)

    # collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_words(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


# ----------------------------
# Heuristic language scoring (fallback)
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


def english_ratio_heuristic(text: str) -> float:
    words = tokenize_words(text)
    if not words:
        return 0.0

    en_sw = 0
    hu_sw = 0
    hu_diac = 0

    for w in words:
        if w in EN_STOPWORDS:
            en_sw += 1
        if w in HU_STOPWORDS:
            hu_sw += 1
        if any(ch in HU_DIACRITICS for ch in w):
            hu_diac += 1

    # Strong HU signal from diacritics
    hu_score = hu_sw + 1.5 * hu_diac
    en_score = en_sw

    unknown = len(words) - (en_sw + hu_sw)
    if en_score > hu_score:
        # unknown words likely follow the dominant EN context
        en_assigned = en_score + 0.8 * unknown
    else:
        # unknown words likely NOT EN
        en_assigned = en_score + 0.2 * unknown

    return max(0.0, min(1.0, en_assigned / len(words)))


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


def english_ratio_langdetect(text: str) -> Optional[float]:
    if not _LANGDETECT_AVAILABLE or detect_langs is None:
        return None

    chunks = split_into_chunks(text)
    if not chunks:
        return 0.0

    en_words = 0
    total_words = 0

    for ch in chunks:
        # weight by word count
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

        # count as English if 'en' is top with decent confidence
        if getattr(top, "lang", "") == "en" and getattr(top, "prob", 0.0) >= 0.70:
            en_words += wcount

    if total_words == 0:
        return 0.0
    return en_words / total_words


def english_ratio(text: str, force_heuristic: bool = False) -> float:
    text = clean_for_lang(text)
    if not text:
        return 0.0

    if not force_heuristic:
        r = english_ratio_langdetect(text)
        if r is not None:
            return r

    return english_ratio_heuristic(text)


# ----------------------------
# Processing
# ----------------------------
@dataclass
class Decision:
    delete: bool
    ratio: float
    preview: str
    subreddit: str
    entry_type: str


def make_preview(s: str, n: int = 140) -> str:
    s = s.strip().replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    if len(s) > n:
        return s[:n] + "..."
    return s


def decide_entry(
    entry_type: str,
    entry_text: str,
    threshold: float,
    force_heuristic: bool,
) -> Decision:
    subreddit = extract_multiline_field(entry_text, "subreddit")
    title = extract_multiline_field(entry_text, "title")
    body = extract_multiline_field(entry_text, "body")

    # What text do we judge?
    if entry_type == "Post":
        judge_text = (title + "\n" + body).strip()
    else:
        judge_text = body.strip()

    # Special cases: empty/removed
    if not judge_text or judge_text.strip().lower() in {"[removed]", "[deleted]"}:
        return Decision(False, 0.0, make_preview(judge_text), subreddit, entry_type)

    ratio = english_ratio(judge_text, force_heuristic=force_heuristic)
    prev = make_preview(judge_text)
    return Decision(ratio >= threshold, ratio, prev, subreddit, entry_type)


def unique_backup_path(path: Path) -> Path:
    base = path.with_suffix(path.suffix + ".bak")
    if not base.exists():
        return base
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_suffix(path.suffix + f".bak_{ts}")


def process_file(
    in_path: Path,
    out_path: Path,
    threshold: float,
    show_deleted: bool,
    ask: bool,
    dry_run: bool,
    force_heuristic: bool,
) -> Tuple[int, int]:
    raw = in_path.read_text(encoding="utf-8", errors="replace")
    segments = split_segments(raw)

    kept_parts: List[str] = []
    total_entries = 0
    deleted_entries = 0

    for kind, a, b in segments:
        if kind == "text":
            kept_parts.append(a)
            continue

        # entry
        entry_type = a
        entry_text = b or ""
        total_entries += 1

        dec = decide_entry(entry_type, entry_text, threshold, force_heuristic)

        do_delete = dec.delete
        if do_delete and ask:
            print(f"\n[{in_path.name}] Candidate: {dec.entry_type} | {dec.subreddit or '(no subreddit)'} "
                  f"| en_ratio={dec.ratio:.2f}")
            print(f"Preview: {dec.preview}")
            ans = input("Delete this entry? [y/N] ").strip().lower()
            do_delete = (ans == "y")

        if do_delete:
            deleted_entries += 1
            if show_deleted:
                print(f"[DELETED] {in_path.name} | {dec.entry_type} | {dec.subreddit or '(no subreddit)'} "
                      f"| en_ratio={dec.ratio:.2f} | {dec.preview}")
            # skip appending
        else:
            kept_parts.append(entry_text)

    if dry_run:
        return total_entries, deleted_entries

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(kept_parts), encoding="utf-8")
    return total_entries, deleted_entries


def iter_files(inputfolder: Path, recursive: bool, pattern: str) -> List[Path]:
    if recursive:
        return sorted(inputfolder.rglob(pattern))
    return sorted(inputfolder.glob(pattern))


def main() -> int:
    p = argparse.ArgumentParser(description="Remove mostly-English Reddit entries (Comment/Post) from export files.")
    p.add_argument("-inputfolder", "--inputfolder", required=True, help="Folder that contains exported .txt files.")
    p.add_argument("--pattern", default="*.txt", help="Glob pattern for files (default: *.txt).")
    p.add_argument("--recursive", action="store_true", help="Process subfolders too.")
    p.add_argument("--threshold", type=float, default=0.90, help="Delete if English ratio >= threshold (default: 0.90).")

    p.add_argument("--show-deleted", action="store_true", help="Print deletions continuously to console.")
    p.add_argument("--ask", action="store_true", help="Ask confirmation for every candidate deletion (type 'y' to delete).")
    p.add_argument("--dry-run", action="store_true", help="Do not write files, only report what would be deleted.")

    p.add_argument("--inplace", action="store_true",
                   help="Overwrite files in-place (creates a .bak backup next to each file).")
    p.add_argument("--outputfolder", default=None,
                   help="Output folder for cleaned files (default: create '<inputfolder>/cleaned').")

    p.add_argument("--force-heuristic", action="store_true",
                   help="Do not use langdetect even if installed (use heuristic only).")

    args = p.parse_args()

    in_dir = Path(args.inputfolder).expanduser().resolve()
    if not in_dir.exists() or not in_dir.is_dir():
        print(f"ERROR: inputfolder does not exist or not a directory: {in_dir}")
        return 2

    if args.inplace:
        out_dir = in_dir
    else:
        out_dir = Path(args.outputfolder).expanduser().resolve() if args.outputfolder else (in_dir / "cleaned")

    files = iter_files(in_dir, args.recursive, args.pattern)
    if not files:
        print(f"No files matched: {in_dir} / {args.pattern}")
        return 0

    if _LANGDETECT_AVAILABLE and not args.force_heuristic:
        detector_info = "langdetect"
    else:
        detector_info = "heuristic(stopwords+diacritics)"
    print(f"Detector: {detector_info} | threshold={args.threshold:.2f}")
    print(f"Files: {len(files)} | mode={'inplace' if args.inplace else 'outputfolder'} | dry_run={args.dry_run}")

    grand_total = 0
    grand_deleted = 0

    for f in files:
        rel = f.relative_to(in_dir)
        out_path = (out_dir / rel) if not args.inplace else f

        # backup if inplace and not dry-run
        if args.inplace and not args.dry_run:
            bak = unique_backup_path(f)
            shutil.copy2(f, bak)

        total, deleted = process_file(
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

    print(f"\nDone. Entries total={grand_total}, deleted={grand_deleted}, kept={grand_total - grand_deleted}")
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
