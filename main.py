
from __future__ import annotations

import argparse
import re
import shutil
from collections import Counter
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
SECTION_RE = re.compile(r"^\s*\[(hu|en|else|other)\]\s*$", flags=re.IGNORECASE)

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
# Stopwords store (external file + learning)
# ----------------------------
HU_DIACRITICS = set("áéíóöőúüűÁÉÍÓÖŐÚÜŰ")

@dataclass
class StopwordsStore:
    path: Path
    hu: set[str]
    en: set[str]
    other: set[str]
    _mtime: Optional[float] = None

    @classmethod
    def from_path(cls, path: Path) -> "StopwordsStore":
        store = cls(path=path, hu=set(), en=set(), other=set(), _mtime=None)
        store.ensure_exists()
        store.load()
        return store

    def ensure_exists(self) -> None:
        if self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        template = (
            "# stopwords.txt\n"
            "# Format: INI-like sections. One word per line.\n"
            "# Lines starting with # or ; are comments.\n\n"
            "[hu]\n"
            "# Hungarian stopwords (learned / manual)\n\n"
            "[en]\n"
            "# English stopwords (manual)\n\n"
            "[else]\n"
            "# Other-language stopwords (learned / manual)\n"
        )
        self.path.write_text(template, encoding="utf-8")

    def refresh_if_changed(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            self.ensure_exists()
            mtime = self.path.stat().st_mtime

        if self._mtime is None or mtime > self._mtime:
            self.load()

    def load(self) -> None:
        self.hu.clear()
        self.en.clear()
        self.other.clear()

        current = "hu"
        try:
            raw = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        except FileNotFoundError:
            self.ensure_exists()
            raw = self.path.read_text(encoding="utf-8", errors="replace").splitlines()

        for line in raw:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue

            m = SECTION_RE.match(line)
            if m:
                sec = m.group(1).lower()
                current = "else" if sec == "other" else sec
                continue

            # strip inline comments
            line = re.split(r"\s[;#]", line, maxsplit=1)[0].strip()
            if not line:
                continue
            w = line.lower()

            if current == "hu":
                self.hu.add(w)
            elif current == "en":
                self.en.add(w)
            else:
                self.other.add(w)

        try:
            self._mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            self._mtime = None

    def save(self) -> None:
        def section(name: str, words: set[str], comment: str) -> str:
            lines = [f"[{name}]", f"# {comment}"]
            for w in sorted(words):
                lines.append(w)
            lines.append("")  # blank line
            return "\n".join(lines)

        content = []
        content.append("# stopwords.txt")
        content.append("# Auto-maintained by the Reddit filter script.")
        content.append("# Format: INI-like sections. One word per line.")
        content.append("# You can edit it manually; the script will reload changes automatically.")
        content.append("")
        content.append(section("hu", self.hu, "Hungarian stopwords (learned / manual)"))
        content.append(section("en", self.en, "English stopwords (manual)"))
        content.append(section("else", self.other, "Other-language stopwords (learned / manual)"))
        text = "\n".join(content).rstrip() + "\n"

        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(self.path)

        self._mtime = self.path.stat().st_mtime

    def add_words(self, section: str, words: set[str]) -> int:
        if not words:
            return 0

        section = section.lower()
        if section in {"other", "else"}:
            target = self.other
            section = "else"
        elif section == "en":
            target = self.en
        else:
            target = self.hu
            section = "hu"

        before = len(target)
        target.update(words)
        added = len(target) - before
        if added > 0:
            self.save()
            # Requirement: always load from stopwords.txt after adding (so next sentence uses it)
            self.load()
        return added


# ----------------------------
# Heuristic scoring (HU / EN / OTHER)
# ----------------------------
def _hu_ratio_heuristic(text: str, sw_hu: set[str], sw_en: set[str], sw_other: set[str]) -> float:
    words = tokenize_words(text)
    if not words:
        return 0.0

    hu_sw = 0
    foreign_sw = 0
    hu_diac = 0
    any_sets = 0

    for w in words:
        in_hu = w in sw_hu
        in_en = w in sw_en
        in_ot = w in sw_other
        if in_hu:
            hu_sw += 1
        if in_en or in_ot:
            foreign_sw += 1
        if in_hu or in_en or in_ot:
            any_sets += 1
        if any(ch in HU_DIACRITICS for ch in w):
            hu_diac += 1

    unknown = len(words) - any_sets
    hu_score = hu_sw + 1.5 * hu_diac
    foreign_score = foreign_sw

    if hu_score >= foreign_score:
        hu_assigned = hu_score + 0.8 * unknown
    else:
        hu_assigned = hu_score + 0.2 * unknown

    return max(0.0, min(1.0, float(hu_assigned) / len(words)))

def _en_ratio_heuristic(text: str, sw_hu: set[str], sw_en: set[str], sw_other: set[str]) -> float:
    words = tokenize_words(text)
    if not words:
        return 0.0

    en_sw = 0
    hu_sw = 0
    other_sw = 0
    hu_diac = 0
    any_sets = 0

    for w in words:
        in_hu = w in sw_hu
        in_en = w in sw_en
        in_ot = w in sw_other
        if in_en:
            en_sw += 1
        if in_hu:
            hu_sw += 1
        if in_ot:
            other_sw += 1
        if in_hu or in_en or in_ot:
            any_sets += 1
        if any(ch in HU_DIACRITICS for ch in w):
            hu_diac += 1

    unknown = len(words) - any_sets

    en_score = en_sw
    hu_score = hu_sw + 1.5 * hu_diac
    competitor = max(hu_score, other_sw)

    if en_score > competitor:
        en_assigned = en_score + 0.8 * unknown
    else:
        en_assigned = en_score + 0.2 * unknown

    return max(0.0, min(1.0, float(en_assigned) / len(words)))

def _other_ratio_heuristic(text: str, sw_hu: set[str], sw_en: set[str], sw_other: set[str]) -> float:
    words = tokenize_words(text)
    if not words:
        return 0.0

    other_sw = 0
    hu_sw = 0
    en_sw = 0
    hu_diac = 0
    any_sets = 0

    for w in words:
        in_hu = w in sw_hu
        in_en = w in sw_en
        in_ot = w in sw_other
        if in_ot:
            other_sw += 1
        if in_hu:
            hu_sw += 1
        if in_en:
            en_sw += 1
        if in_hu or in_en or in_ot:
            any_sets += 1
        if any(ch in HU_DIACRITICS for ch in w):
            hu_diac += 1

    unknown = len(words) - any_sets

    other_score = other_sw
    hu_score = hu_sw + 1.5 * hu_diac
    competitor = max(hu_score, en_sw)

    if other_score > competitor:
        other_assigned = other_score + 0.8 * unknown
    else:
        other_assigned = other_score + 0.2 * unknown

    return max(0.0, min(1.0, float(other_assigned) / len(words)))


def _lang_prob_ratio_langdetect(text: str, lang_code: str) -> Optional[float]:
    """
    Returns expected proportion of words in given language, using langdetect probabilities.
    Weighted by token count per sentence-ish chunk.
    """
    if not _LANGDETECT_AVAILABLE or detect_langs is None:
        return None

    chunks = split_into_chunks(text)
    if not chunks:
        return 0.0

    lang_words = 0.0
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

        p = 0.0
        for lp in langs:
            if getattr(lp, "lang", "") == lang_code:
                p = float(getattr(lp, "prob", 0.0))
                break

        total_words += wcount
        lang_words += p * wcount

    if total_words == 0:
        return 0.0
    return float(lang_words / total_words)

def _other_prob_ratio_langdetect(text: str) -> Optional[float]:
    """
    Expected 'other language' proportion = 1 - p(hu) - p(en).
    """
    if not _LANGDETECT_AVAILABLE or detect_langs is None:
        return None

    chunks = split_into_chunks(text)
    if not chunks:
        return 0.0

    other_words = 0.0
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

        p_hu = 0.0
        p_en = 0.0
        for lp in langs:
            lang = getattr(lp, "lang", "")
            prob = float(getattr(lp, "prob", 0.0))
            if lang == "hu":
                p_hu = prob
            elif lang == "en":
                p_en = prob

        p_other = max(0.0, min(1.0, 1.0 - p_hu - p_en))
        total_words += wcount
        other_words += p_other * wcount

    if total_words == 0:
        return 0.0
    return float(other_words / total_words)

def hungarian_ratio(text: str, sw: StopwordsStore, force_heuristic: bool = False) -> float:
    text = clean_for_lang(text)
    if not text:
        return 0.0

    h = _hu_ratio_heuristic(text, sw.hu, sw.en, sw.other)
    if force_heuristic:
        return h

    r = _lang_prob_ratio_langdetect(text, "hu")
    if r is None:
        return h

    return max(h, r)

def english_ratio(text: str, sw: StopwordsStore, force_heuristic: bool = False) -> float:
    text = clean_for_lang(text)
    if not text:
        return 0.0

    h = _en_ratio_heuristic(text, sw.hu, sw.en, sw.other)
    if force_heuristic:
        return h

    r = _lang_prob_ratio_langdetect(text, "en")
    if r is None:
        return h
    return max(h, r)

def other_ratio(text: str, sw: StopwordsStore, force_heuristic: bool = False) -> float:
    text = clean_for_lang(text)
    if not text:
        return 0.0

    h = _other_ratio_heuristic(text, sw.hu, sw.en, sw.other)
    if force_heuristic:
        return h

    r = _other_prob_ratio_langdetect(text)
    if r is None:
        return h
    return max(h, r)


# ----------------------------
# Utilities
# ----------------------------
@dataclass
class Decision:
    delete: bool
    hu_ratio: float
    en_ratio: float
    other_ratio: float
    preview: str
    kind: str
    verdict: str  # KEEP / DELETE / ASK

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

def learnable_words(
    judge_text: str,
    sw: StopwordsStore,
    target: str,
    min_freq: int = 2,
    max_len: int = 6,
) -> set[str]:
    """
    Words to add to stopwords based on an answered ambiguous decision.
    target='hu' or 'else'
    - only adds words that are not already in any section
    - prefers short and/or repeated words (to avoid polluting stopwords with content words)
    - for 'else' target: avoid Hungarian diacritics
    """
    cleaned = clean_for_lang(judge_text)
    toks = tokenize_words(cleaned)
    if not toks:
        return set()

    counts = Counter(toks)
    unknown = {w for w in counts if (w not in sw.hu and w not in sw.en and w not in sw.other)}

    if target.lower() in {"hu", "hungarian"}:
        out = set()
        for w in unknown:
            has_hu_diac = any(ch in HU_DIACRITICS for ch in w)
            if has_hu_diac:
                out.add(w)
                continue
            if len(w) <= max_len and counts[w] >= min_freq:
                out.add(w)
        return out

    out = set()
    for w in unknown:
        if any(ch in HU_DIACRITICS for ch in w):
            continue
        if len(w) <= max_len and counts[w] >= min_freq:
            out.add(w)
    return out

def decide_text(
    kind: str,
    judge_text: str,
    sw: StopwordsStore,
    threshold: float,
    force_heuristic: bool,
) -> Decision:
    preview = make_preview(judge_text)
    cleaned = clean_for_lang(judge_text)
    words = tokenize_words(cleaned)

    if len(words) == 0:
        return Decision(False, 1.0, 0.0, 0.0, preview, kind=kind, verdict="KEEP")

    r_hu = hungarian_ratio(judge_text, sw, force_heuristic=force_heuristic)
    r_en = english_ratio(judge_text, sw, force_heuristic=force_heuristic)
    r_ot = other_ratio(judge_text, sw, force_heuristic=force_heuristic)

    if r_hu >= threshold:
        return Decision(False, r_hu, r_en, r_ot, preview, kind=kind, verdict="KEEP")

    if (r_en >= threshold) or (r_ot >= threshold):
        return Decision(True, r_hu, r_en, r_ot, preview, kind=kind, verdict="DELETE")

    return Decision(False, r_hu, r_en, r_ot, preview, kind=kind, verdict="ASK")


# ============================================================
# MODE 1: "Comment:/Post:" exports (user export)
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

def process_file_userexport(
    in_path: Path,
    out_path: Path,
    threshold: float,
    show_deleted: bool,
    ask_uncertain: bool,
    ask_all_deletions: bool,
    dry_run: bool,
    force_heuristic: bool,
    sw: StopwordsStore,
    learn_min_freq: int,
    learn_max_len: int,
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

        title = extract_multiline_field(entry_text, "title")
        body = extract_multiline_field(entry_text, "body")

        judge_text = (title + "\n" + body).strip() if entry_type == "Post" else body.strip()

        if not judge_text or judge_text.strip().lower() in {"[removed]", "[deleted]"}:
            kept_parts.append(entry_text)
            continue

        sw.refresh_if_changed()
        dec = decide_text(entry_type, judge_text, sw, threshold, force_heuristic)

        do_delete = False
        if dec.verdict == "KEEP":
            do_delete = False
        elif dec.verdict == "DELETE":
            do_delete = True
        else:
            if ask_uncertain:
                print(f"\n[{in_path.name}] Bizonytalan: {dec.kind} | hu={dec.hu_ratio:.2f} en={dec.en_ratio:.2f} other={dec.other_ratio:.2f}")
                print(f"Preview: {dec.preview}")
                ans = input("Magyar szöveg? (y=marad+tanul, n=töröl+tanul) [y/N] ").strip().lower()
                if ans == "y":
                    do_delete = False
                    w = learnable_words(judge_text, sw, target="hu", min_freq=learn_min_freq, max_len=learn_max_len)
                    added = sw.add_words("hu", w)
                    if added:
                        print(f"  +HU stopwords: {added} új szó")
                else:
                    do_delete = True
                    w = learnable_words(judge_text, sw, target="else", min_freq=learn_min_freq, max_len=learn_max_len)
                    added = sw.add_words("else", w)
                    if added:
                        print(f"  +ELSE stopwords: {added} új szó")
            else:
                foreign = max(dec.en_ratio, dec.other_ratio)
                do_delete = foreign > dec.hu_ratio

        if do_delete and ask_all_deletions:
            print(f"\n[{in_path.name}] Törlés jelölt: {dec.kind} | hu={dec.hu_ratio:.2f} en={dec.en_ratio:.2f} other={dec.other_ratio:.2f}")
            print(f"Preview: {dec.preview}")
            ans = input("Biztosan töröljem? [y/N] ").strip().lower()
            do_delete = (ans == "y")

        if do_delete:
            deleted_entries += 1
            if show_deleted:
                print(f"[DELETED] {in_path.name} | {dec.kind} | hu={dec.hu_ratio:.2f} en={dec.en_ratio:.2f} other={dec.other_ratio:.2f} | {dec.preview}")
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

        if SUBREDDIT_HEADER_RE.match(line) and line.lstrip() == line and in_post:
            flush()
            in_post = False
            cur.append(line)
            continue

        cur.append(line)

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
    lines = pre_block.splitlines()
    title = ""
    body_lines: List[str] = []

    for l in lines:
        m = BYLINE_RE.match(l.strip())
        if m:
            title = m.group(2).strip()
            break

    for i, l in enumerate(lines):
        if BODY_START_RE.match(l):
            j = i + 1
            while j < len(lines):
                if COMMENT_START_RE.match(lines[j]):
                    break
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
    lines = comment_block.splitlines()
    content_lines: List[str] = []
    started = False

    for l in lines:
        if COMMENT_START_RE.match(l):
            started = True
            continue
        if not started:
            continue

        raw = l
        if raw.startswith("    "):
            raw = raw[4:]
        else:
            raw = raw.lstrip()

        if not content_lines:
            if ":" in raw:
                _, txt = raw.split(":", 1)
                content_lines.append(txt.strip())
            else:
                content_lines.append(raw.strip())
        else:
            content_lines.append(raw.rstrip())

    return "\n".join([x for x in content_lines if x is not None]).strip()

def process_file_subreddits(
    in_path: Path,
    out_path: Path,
    threshold: float,
    show_deleted: bool,
    ask_uncertain: bool,
    ask_all_deletions: bool,
    dry_run: bool,
    force_heuristic: bool,
    sw: StopwordsStore,
    learn_min_freq: int,
    learn_max_len: int,
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

        pre, comments = split_post_into_pre_and_comments(block)

        total += 1
        judge_post = extract_subreddit_post_text(pre)
        sw.refresh_if_changed()
        post_dec = decide_text("Post", judge_post, sw, threshold, force_heuristic)

        do_delete_post = False
        if post_dec.verdict == "KEEP":
            do_delete_post = False
        elif post_dec.verdict == "DELETE":
            do_delete_post = True
        else:
            if ask_uncertain:
                print(f"\n[{in_path.name}] Bizonytalan: POST | hu={post_dec.hu_ratio:.2f} en={post_dec.en_ratio:.2f} other={post_dec.other_ratio:.2f}")
                print(f"Preview: {post_dec.preview}")
                ans = input("Magyar poszt? (y=marad+tanul, n=töröl+tanul) [y/N] ").strip().lower()
                if ans == "y":
                    do_delete_post = False
                    w = learnable_words(judge_post, sw, target="hu", min_freq=learn_min_freq, max_len=learn_max_len)
                    added = sw.add_words("hu", w)
                    if added:
                        print(f"  +HU stopwords: {added} új szó")
                else:
                    do_delete_post = True
                    w = learnable_words(judge_post, sw, target="else", min_freq=learn_min_freq, max_len=learn_max_len)
                    added = sw.add_words("else", w)
                    if added:
                        print(f"  +ELSE stopwords: {added} új szó")
            else:
                foreign = max(post_dec.en_ratio, post_dec.other_ratio)
                do_delete_post = foreign > post_dec.hu_ratio

        if do_delete_post and ask_all_deletions:
            print(f"\n[{in_path.name}] Törlés jelölt: POST | hu={post_dec.hu_ratio:.2f} en={post_dec.en_ratio:.2f} other={post_dec.other_ratio:.2f}")
            print(f"Preview: {post_dec.preview}")
            ans = input("Biztosan töröljem a posztot (és kommenteket)? [y/N] ").strip().lower()
            do_delete_post = (ans == "y")

        if do_delete_post:
            deleted += 1
            if show_deleted:
                print(f"[DELETED] {in_path.name} | Post | hu={post_dec.hu_ratio:.2f} en={post_dec.en_ratio:.2f} other={post_dec.other_ratio:.2f} | {post_dec.preview}")
            continue

        kept.append(pre)

        for c in comments:
            total += 1
            judge_c = extract_subreddit_comment_text(c)
            sw.refresh_if_changed()
            c_dec = decide_text("Comment", judge_c, sw, threshold, force_heuristic)

            do_delete_c = False
            if c_dec.verdict == "KEEP":
                do_delete_c = False
            elif c_dec.verdict == "DELETE":
                do_delete_c = True
            else:
                if ask_uncertain:
                    print(f"\n[{in_path.name}] Bizonytalan: COMMENT | hu={c_dec.hu_ratio:.2f} en={c_dec.en_ratio:.2f} other={c_dec.other_ratio:.2f}")
                    print(f"Preview: {c_dec.preview}")
                    ans = input("Magyar komment? (y=marad+tanul, n=töröl+tanul) [y/N] ").strip().lower()
                    if ans == "y":
                        do_delete_c = False
                        w = learnable_words(judge_c, sw, target="hu", min_freq=learn_min_freq, max_len=learn_max_len)
                        added = sw.add_words("hu", w)
                        if added:
                            print(f"  +HU stopwords: {added} új szó")
                    else:
                        do_delete_c = True
                        w = learnable_words(judge_c, sw, target="else", min_freq=learn_min_freq, max_len=learn_max_len)
                        added = sw.add_words("else", w)
                        if added:
                            print(f"  +ELSE stopwords: {added} új szó")
                else:
                    foreign = max(c_dec.en_ratio, c_dec.other_ratio)
                    do_delete_c = foreign > c_dec.hu_ratio

            if do_delete_c and ask_all_deletions:
                print(f"\n[{in_path.name}] Törlés jelölt: COMMENT | hu={c_dec.hu_ratio:.2f} en={c_dec.en_ratio:.2f} other={c_dec.other_ratio:.2f}")
                print(f"Preview: {c_dec.preview}")
                ans = input("Biztosan töröljem? [y/N] ").strip().lower()
                do_delete_c = (ans == "y")

            if do_delete_c:
                deleted += 1
                if show_deleted:
                    print(f"[DELETED] {in_path.name} | Comment | hu={c_dec.hu_ratio:.2f} en={c_dec.en_ratio:.2f} other={c_dec.other_ratio:.2f} | {c_dec.preview}")
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
    p = argparse.ArgumentParser(
        description="Filter Reddit exports: keep Hungarian, delete English/other; learns stopwords from uncertain cases."
    )
    p.add_argument("-inputfolder", "--inputfolder", required=True, help="Folder that contains exported .txt files.")
    p.add_argument("--pattern", default="*.txt", help="Glob pattern for files (default: *.txt).")
    p.add_argument("--recursive", action="store_true", help="Process subfolders too.")
    p.add_argument("--threshold", type=float, default=0.90,
                   help="Decision threshold for HU/EN/OTHER ratio. HU>=T => keep. EN>=T or OTHER>=T => delete. Else => ask.")

    p.add_argument("--stopwords", default=None,
                   help="Path to stopwords.txt (INI-like with [hu]/[en]/[else]). Default: next to the script.")

    p.add_argument("--show-deleted", action="store_true", help="Print deletions continuously to console.")
    p.add_argument("--ask", action="store_true",
                   help="Legacy: ask confirmation for every deletion candidate (type 'y' to delete).")
    p.add_argument("--noask", action="store_true",
                   help="Do not ask in uncertain cases. Fallback: keep if HU score >= foreign score, else delete.")
    p.add_argument("--dry-run", action="store_true", help="Do not write files, only report what would be deleted.")

    p.add_argument("--inplace", action="store_true",
                   help="Overwrite files in-place (creates a .bak backup next to each file).")
    p.add_argument("--outputfolder", default=None,
                   help="Output folder for cleaned files (default: create '<inputfolder>/cleaned').")

    p.add_argument("--force-heuristic", action="store_true",
                   help="Do not use langdetect even if installed (use heuristic only).")

    p.add_argument("--subreddits", action="store_true",
                   help="Enable subreddit-dump mode (=== r/... ===, Post:, comment:).")

    p.add_argument("--learn-min-freq", type=int, default=2,
                   help="Learning: only add words that appear at least this many times in the judged text (default: 2).")
    p.add_argument("--learn-max-len", type=int, default=6,
                   help="Learning: only add words up to this length (default: 6).")

    args = p.parse_args()

    in_dir = Path(args.inputfolder).expanduser().resolve()
    if not in_dir.exists() or not in_dir.is_dir():
        print(f"ERROR: inputfolder does not exist or not a directory: {in_dir}")
        return 2

    out_dir = in_dir if args.inplace else (
        Path(args.outputfolder).expanduser().resolve()
        if args.outputfolder else (in_dir / "cleaned")
    )

    files = iter_files(in_dir, args.recursive, args.pattern)
    if not files:
        print(f"No files matched: {in_dir} / {args.pattern}")
        return 0

    stop_path = Path(args.stopwords).expanduser().resolve() if args.stopwords else Path(__file__).with_name("stopwords.txt").resolve()
    sw = StopwordsStore.from_path(stop_path)

    detector_info = "langdetect(prob)+heuristic" if (_LANGDETECT_AVAILABLE and not args.force_heuristic) else "heuristic"
    ask_uncertain = (not args.noask)
    ask_all_deletions = bool(args.ask) and (not args.noask)

    print(f"Detector: {detector_info} | threshold={args.threshold:.2f} | subreddits={args.subreddits}")
    print(f"Stopwords: {sw.path} | hu={len(sw.hu)} en={len(sw.en)} else={len(sw.other)}")
    print(f"Files: {len(files)} | mode={'inplace' if args.inplace else 'outputfolder'} | dry_run={args.dry_run}")
    print(f"Asking: uncertain={'ON' if ask_uncertain else 'OFF'} | all_deletions={'ON' if ask_all_deletions else 'OFF'}")

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
                threshold=args.threshold,
                show_deleted=args.show_deleted,
                ask_uncertain=ask_uncertain,
                ask_all_deletions=ask_all_deletions,
                dry_run=args.dry_run,
                force_heuristic=args.force_heuristic,
                sw=sw,
                learn_min_freq=args.learn_min_freq,
                learn_max_len=args.learn_max_len,
            )
        else:
            total, deleted = process_file_userexport(
                in_path=f,
                out_path=out_path,
                threshold=args.threshold,
                show_deleted=args.show_deleted,
                ask_uncertain=ask_uncertain,
                ask_all_deletions=ask_all_deletions,
                dry_run=args.dry_run,
                force_heuristic=args.force_heuristic,
                sw=sw,
                learn_min_freq=args.learn_min_freq,
                learn_max_len=args.learn_max_len,
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
