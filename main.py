from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Set

VERSION = "2026-02-16_v7 (double-guard: FORCE prompt if hu/en <= zero_eps even before delete)"

# ============================
# Optional langdetect
# ============================
_LANGDETECT_AVAILABLE = False
try:
    from langdetect import detect_langs  # type: ignore
    from langdetect import DetectorFactory  # type: ignore

    DetectorFactory.seed = 0
    _LANGDETECT_AVAILABLE = True
except Exception:
    detect_langs = None  # type: ignore


# ============================
# Text cleanup/tokenize
# ============================
URL_RE = re.compile(r"https?://\S+")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
INLINE_CODE_RE = re.compile(r"`[^`]*`")
WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)  # unicode letters only

# CJK/Hangul/Kana detection
CJK_RE = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF\u3040-\u30FF\uAC00-\uD7AF]")


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


def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


# ============================
# Default stopwords (seed)
# ============================
HU_DIACRITICS = set("áéíóöőúüűÁÉÍÓÖŐÚÜŰ")

DEFAULT_EN_STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","when","while","for","to","of","in","on","at","by","with",
    "as","is","are","was","were","be","been","being","do","does","did","doing","have","has","had","having",
    "i","you","he","she","it","we","they","me","him","her","us","them","my","your","his","their","our",
    "this","that","these","those","there","here","what","which","who","whom","why","how",
    "not","no","yes","so","very","just","also","too","more","most","less","least",
    "can","could","would","should","may","might","must","will","shall",
    "from","into","about","over","under","again","because",
}

DEFAULT_HU_STOPWORDS = {
    "a","az","és","vagy","de","ha","akkor","mert","hogy","mint","is","sem","se","nem","igen","nincs","van","volt","lesz",
    "én","te","ő","mi","ti","ők","engem","nekem","tőlem","velem","veled","vele","velünk","veletek",
    "azt","ezt","itt","ott","ide","oda","innen","onnan","amikor","ahol","ami","aki","akik","mely","melyik","miért","hogyan",
    "már","még","csak","nagyon","túl","sok","kevés","kicsit","kb","szóval","persze",
    "meg","rá","le","fel","be","ki","el","át","össze","szét",
}

EN_STOPWORDS: Set[str] = set(DEFAULT_EN_STOPWORDS)
HU_STOPWORDS: Set[str] = set(DEFAULT_HU_STOPWORDS)

STOPWORDS_PATH = Path("stopwords.txt").resolve()
_STOPWORDS_MTIME: Optional[float] = None


def _write_stopwords_file(path: Path, hu: Set[str], en: Set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# stopwords.txt\n")
        f.write("# Format:\n# [HU] then one word per line\n# [EN] then one word per line\n\n")
        f.write("[HU]\n")
        for w in sorted(hu):
            f.write(w + "\n")
        f.write("\n[EN]\n")
        for w in sorted(en):
            f.write(w + "\n")


def _read_stopwords_file(path: Path) -> Tuple[Set[str], Set[str]]:
    hu = set(DEFAULT_HU_STOPWORDS)
    en = set(DEFAULT_EN_STOPWORDS)

    if not path.exists():
        _write_stopwords_file(path, hu, en)
        return hu, en

    section: Optional[str] = None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if s.upper() == "[HU]":
            section = "HU"
            continue
        if s.upper() == "[EN]":
            section = "EN"
            continue
        w = s.lower()
        if not w:
            continue
        if section == "HU":
            hu.add(w)
        elif section == "EN":
            en.add(w)

    return hu, en


def reload_stopwords_if_changed(force: bool = False) -> None:
    global HU_STOPWORDS, EN_STOPWORDS, _STOPWORDS_MTIME

    if not STOPWORDS_PATH.exists():
        hu, en = _read_stopwords_file(STOPWORDS_PATH)
        HU_STOPWORDS, EN_STOPWORDS = hu, en
        _STOPWORDS_MTIME = STOPWORDS_PATH.stat().st_mtime if STOPWORDS_PATH.exists() else None
        return

    mtime = STOPWORDS_PATH.stat().st_mtime
    if force or (_STOPWORDS_MTIME is None) or (mtime != _STOPWORDS_MTIME):
        hu, en = _read_stopwords_file(STOPWORDS_PATH)
        HU_STOPWORDS, EN_STOPWORDS = hu, en
        _STOPWORDS_MTIME = mtime


def learn_stopwords_from_text(text: str, lang: str) -> None:
    reload_stopwords_if_changed()
    words = tokenize_words(clean_for_lang(text))
    words = [w for w in words if len(w) >= 2]
    if not words:
        return

    if lang.upper() == "HU":
        HU_STOPWORDS.update(words)
    elif lang.upper() == "EN":
        EN_STOPWORDS.update(words)
    else:
        return

    _write_stopwords_file(STOPWORDS_PATH, HU_STOPWORDS, EN_STOPWORDS)
    reload_stopwords_if_changed(force=True)


def stopwords_match_ratios(text: str) -> Tuple[float, float, int]:
    reload_stopwords_if_changed()
    words = tokenize_words(clean_for_lang(text))
    if not words:
        return 0.0, 0.0, 0
    hu = sum(1 for w in words if w in HU_STOPWORDS)
    en = sum(1 for w in words if w in EN_STOPWORDS)
    total = len(words)
    return hu / total, en / total, total


# ============================
# Language ratios
# ============================
def _hu_ratio_heuristic(text: str) -> float:
    words = tokenize_words(text)
    if not words:
        return 0.0

    hu_sw = sum(1 for w in words if w in HU_STOPWORDS)
    en_sw = sum(1 for w in words if w in EN_STOPWORDS)
    hu_diac = sum(1 for w in words if any(ch in HU_DIACRITICS for ch in w))

    hu_score = hu_sw + 1.5 * hu_diac
    en_score = en_sw
    unknown = len(words) - (hu_sw + en_sw)

    if hu_score >= en_score:
        hu_assigned = hu_score + 0.8 * unknown
    else:
        hu_assigned = hu_score + 0.2 * unknown

    return max(0.0, min(1.0, hu_assigned / len(words)))


def _lang_prob_ratio_langdetect(text: str, lang_code: str) -> Optional[float]:
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


def detect_top_lang(text: str) -> Optional[Tuple[str, float]]:
    if not _LANGDETECT_AVAILABLE or detect_langs is None:
        return None
    cleaned = clean_for_lang(text)
    if not cleaned:
        return None
    try:
        langs = detect_langs(cleaned)
    except Exception:
        return None
    if not langs:
        return None
    top = langs[0]
    return (getattr(top, "lang", ""), float(getattr(top, "prob", 0.0)))


def hungarian_ratio(text: str, force_heuristic: bool = False) -> float:
    reload_stopwords_if_changed()
    t = clean_for_lang(text)
    if not t:
        return 0.0
    h = _hu_ratio_heuristic(t)
    if force_heuristic:
        return h
    r = _lang_prob_ratio_langdetect(t, "hu")
    return h if r is None else max(h, r)


def english_ratio(text: str, force_heuristic: bool = False) -> float:
    reload_stopwords_if_changed()
    t = clean_for_lang(text)
    if not t:
        return 0.0

    if not force_heuristic:
        r = _lang_prob_ratio_langdetect(t, "en")
        if r is not None:
            return r

    words = tokenize_words(t)
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


# ============================
# Decisions
# ============================
@dataclass
class Decision:
    delete: bool
    ratio_hu: float
    ratio_en: float
    preview: str
    kind: str
    reason: str  # hu, en, other, keep, hu_stopwords, en_stopwords, ask, ask_zero


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


def decide_two_stage(
    text: str,
    threshold: float,
    margin: float,
    stopwords_threshold: float,
    zero_eps: float,
    force_heuristic: bool,
    kind: str,
) -> Decision:
    prev = make_preview(text)
    cleaned = clean_for_lang(text)
    words = tokenize_words(cleaned)

    if not words:
        return Decision(False, 1.0, 0.0, prev, kind, "keep")

    if has_cjk(cleaned):
        return Decision(True, 0.0, 0.0, prev, kind, "other")

    # stopwords dominance first
    hu_sw_ratio, en_sw_ratio, total = stopwords_match_ratios(text)
    if total > 0:
        if hu_sw_ratio >= stopwords_threshold:
            return Decision(False, hu_sw_ratio, en_sw_ratio, prev, kind, "hu_stopwords")
        if en_sw_ratio >= stopwords_threshold:
            return Decision(True, hu_sw_ratio, en_sw_ratio, prev, kind, "en_stopwords")

    r_hu = hungarian_ratio(text, force_heuristic=force_heuristic)
    r_en = english_ratio(text, force_heuristic=force_heuristic)

    # primary ask_zero
    if (r_hu <= zero_eps) and (r_en <= zero_eps):
        return Decision(False, r_hu, r_en, prev, kind, "ask_zero")

    if r_hu >= threshold:
        return Decision(False, r_hu, r_en, prev, kind, "hu")
    if r_en >= threshold:
        return Decision(True, r_hu, r_en, prev, kind, "en")

    # other-language only after ask_zero
    if not force_heuristic:
        top = detect_top_lang(text)
        if top is not None:
            lang, prob = top
            if prob >= 0.70 and lang and (lang not in {"hu", "en"}):
                return Decision(True, r_hu, r_en, prev, kind, "other")

    if max(r_hu, r_en) >= (threshold - margin):
        return Decision(False, r_hu, r_en, prev, kind, "ask")

    return Decision(True, r_hu, r_en, prev, kind, "other")


def handle_prompt(dec: Decision, original_text: str, noask: bool) -> Tuple[bool, str]:
    """
    y => keep + learn HU
    n/Enter => delete + learn EN
    h => delete, no learning
    --noask:
      ask_zero => KEEP
      ask => DELETE
    """
    if noask:
        if dec.reason == "ask_zero":
            return False, ""
        return True, ""

    if dec.reason == "ask_zero":
        print("\n[ASK_ZERO] hu és en ~0 -> KÖTELEZŐ kézi elbírálás.")
    else:
        print("\n[AMBIGUOUS] Küszöb környéke -> kézi elbírálás.")

    print(f"Kind: {dec.kind} | hu={dec.ratio_hu:.6f} | en={dec.ratio_en:.6f} | reason={dec.reason}")
    print(f"Preview: {dec.preview}")

    ans = input("Döntés? (y=megtart+tanul HU, n=töröl+tanul EN, h=töröl(nincs tanulás)) [y/N/h]: ").strip().lower()

    if ans == "y":
        learn_stopwords_from_text(original_text, "HU")
        return False, "HU"
    if ans == "h":
        return True, ""
    learn_stopwords_from_text(original_text, "EN")
    return True, "EN"


# ============================
# Visited
# ============================
def load_visited(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: Set[str] = set()
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        out.add(s)
    return out


def append_visited(path: Path, rel_posix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(rel_posix + "\n")


# ============================================================
# MODE 2: Subreddit dump files
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
                body_lines.append(bl[4:] if bl.startswith("    ") else bl.lstrip())
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

        raw = l[4:] if l.startswith("    ") else l.lstrip()
        if not content_lines:
            if ":" in raw:
                _, txt = raw.split(":", 1)
                content_lines.append(txt.strip())
            else:
                content_lines.append(raw.strip())
        else:
            content_lines.append(raw.rstrip())

    return "\n".join(content_lines).strip()


def unique_backup_path(path: Path) -> Path:
    base = path.with_suffix(path.suffix + ".bak")
    if not base.exists():
        return base
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_suffix(path.suffix + f".bak_{ts}")


def process_file_subreddits(
    in_path: Path,
    out_path: Path,
    threshold: float,
    margin: float,
    stopwords_threshold: float,
    zero_eps: float,
    show_deleted: bool,
    confirm_all_deletions: bool,
    noask: bool,
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

        pre, comments = split_post_into_pre_and_comments(block)

        # POST
        total += 1
        post_text = extract_subreddit_post_text(pre)
        post_dec = decide_two_stage(
            post_text, threshold, margin, stopwords_threshold, zero_eps, force_heuristic, kind="Post"
        )

        # initial decision
        if post_dec.reason in {"ask", "ask_zero"}:
            delete_post, _ = handle_prompt(post_dec, post_text, noask=noask)
        else:
            delete_post = post_dec.reason in {"en", "other", "en_stopwords"}

        # !!! SECOND GUARD: if about to delete but hu/en ~0 => FORCE ask (unless --noask)
        if delete_post and (not noask) and (post_dec.ratio_hu <= zero_eps) and (post_dec.ratio_en <= zero_eps):
            print("\n[FORCE ASK] Törlés előtt: hu/en ~0, ezért most KÖTELEZŐ kérdés (double-guard).")
            forced = Decision(False, post_dec.ratio_hu, post_dec.ratio_en, post_dec.preview, "Post", "ask_zero")
            delete_post, _ = handle_prompt(forced, post_text, noask=noask)

        if delete_post and (not noask) and confirm_all_deletions and (post_dec.reason not in {"ask", "ask_zero"}):
            print(f"\n[{in_path.name}] Candidate DELETE: Post | hu={post_dec.ratio_hu:.6f} | en={post_dec.ratio_en:.6f} | reason={post_dec.reason}")
            print(f"Preview: {post_dec.preview}")
            ans = input("Töröljem a posztot (kommentekkel együtt)? [y/N] ").strip().lower()
            delete_post = (ans == "y")

        if show_deleted:
            if (not delete_post) and post_dec.reason == "hu_stopwords":
                print(f"[KEPT(stopwords)] {in_path.name} | Post | hu={post_dec.ratio_hu:.6f} | en={post_dec.ratio_en:.6f} | reason={post_dec.reason} | {post_dec.preview}")
            if delete_post:
                tag = "DELETED(stopwords)" if post_dec.reason == "en_stopwords" else "DELETED"
                print(f"[{tag}] {in_path.name} | Post | hu={post_dec.ratio_hu:.6f} | en={post_dec.ratio_en:.6f} | reason={post_dec.reason} | {post_dec.preview}")

        if delete_post:
            deleted += 1
            continue

        kept.append(pre)

        # COMMENTS
        for c in comments:
            total += 1
            c_text = extract_subreddit_comment_text(c)
            c_dec = decide_two_stage(
                c_text, threshold, margin, stopwords_threshold, zero_eps, force_heuristic, kind="Comment"
            )

            if c_dec.reason in {"ask", "ask_zero"}:
                delete_c, _ = handle_prompt(c_dec, c_text, noask=noask)
            else:
                delete_c = c_dec.reason in {"en", "other", "en_stopwords"}

            # second guard for comments too
            if delete_c and (not noask) and (c_dec.ratio_hu <= zero_eps) and (c_dec.ratio_en <= zero_eps):
                print("\n[FORCE ASK] Törlés előtt: hu/en ~0, ezért most KÖTELEZŐ kérdés (double-guard).")
                forced = Decision(False, c_dec.ratio_hu, c_dec.ratio_en, c_dec.preview, "Comment", "ask_zero")
                delete_c, _ = handle_prompt(forced, c_text, noask=noask)

            if delete_c and (not noask) and confirm_all_deletions and (c_dec.reason not in {"ask", "ask_zero"}):
                print(f"\n[{in_path.name}] Candidate DELETE: Comment | hu={c_dec.ratio_hu:.6f} | en={c_dec.ratio_en:.6f} | reason={c_dec.reason}")
                print(f"Preview: {c_dec.preview}")
                ans = input("Töröljem a kommentet? [y/N] ").strip().lower()
                delete_c = (ans == "y")

            if show_deleted:
                if (not delete_c) and c_dec.reason == "hu_stopwords":
                    print(f"[KEPT(stopwords)] {in_path.name} | Comment | hu={c_dec.ratio_hu:.6f} | en={c_dec.ratio_en:.6f} | reason={c_dec.reason} | {c_dec.preview}")
                if delete_c:
                    tag = "DELETED(stopwords)" if c_dec.reason == "en_stopwords" else "DELETED"
                    print(f"[{tag}] {in_path.name} | Comment | hu={c_dec.ratio_hu:.6f} | en={c_dec.ratio_en:.6f} | reason={c_dec.reason} | {c_dec.preview}")

            if delete_c:
                deleted += 1
            else:
                kept.append(c)

    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("".join(kept), encoding="utf-8")

    return total, deleted


# ============================
# File iteration & main
# ============================
def iter_files(inputfolder: Path, recursive: bool, pattern: str) -> List[Path]:
    return sorted(inputfolder.rglob(pattern) if recursive else inputfolder.glob(pattern))


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def main() -> int:
    p = argparse.ArgumentParser(
        description="Filter Reddit exports: keep HU, delete EN/other; adaptive stopwords + visited."
    )
    p.add_argument("--version", action="version", version=VERSION)

    p.add_argument("-inputfolder", "--inputfolder", required=True, help="Folder that contains exported .txt files.")
    p.add_argument("--pattern", default="*.txt", help="Glob pattern for files (default: *.txt).")
    p.add_argument("--recursive", action="store_true", help="Process subfolders too.")

    p.add_argument("--threshold", type=float, default=0.75)
    p.add_argument("--margin", type=float, default=0.10)
    p.add_argument("--stopwords-threshold", type=float, default=0.80)
    p.add_argument("--zero-eps", type=float, default=0.02)

    p.add_argument("--show-deleted", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--inplace", action="store_true")
    p.add_argument("--outputfolder", default=None)

    p.add_argument("--force-heuristic", action="store_true")
    p.add_argument("--subreddits", action="store_true")

    p.add_argument("--ask", action="store_true")
    p.add_argument("--noask", action="store_true",
                   help="Disable ALL prompts. ask_zero => keep, ask => delete.")

    args = p.parse_args()

    in_dir = Path(args.inputfolder).expanduser().resolve()
    if not in_dir.exists() or not in_dir.is_dir():
        print(f"ERROR: inputfolder does not exist or not a directory: {in_dir}")
        return 2

    out_dir = in_dir if args.inplace else (
        Path(args.outputfolder).expanduser().resolve()
        if args.outputfolder else (in_dir / "cleaned")
    )

    visited_path = Path("visited.txt").resolve()
    visited = load_visited(visited_path)

    reload_stopwords_if_changed(force=True)

    detector_info = "langdetect(prob)" if (_LANGDETECT_AVAILABLE and not args.force_heuristic) else "heuristic"
    print(f"Version: {VERSION}")
    print(
        f"Detector: {detector_info} | threshold={args.threshold:.2f} | margin={args.margin:.2f} | "
        f"stopwords_threshold={args.stopwords_threshold:.2f} | zero_eps={args.zero_eps:.6f} | subreddits={args.subreddits}"
    )
    print(f"Stopwords file (root): {STOPWORDS_PATH}")
    print(f"Visited file (root): {visited_path} | entries={len(visited)}")

    files_all = iter_files(in_dir, args.recursive, args.pattern)

    files: List[Path] = []
    for f in files_all:
        if f.resolve() in {visited_path, STOPWORDS_PATH}:
            continue
        if (not args.inplace) and is_under(f, out_dir):
            continue
        files.append(f)

    if not files:
        print(f"No files matched: {in_dir} / {args.pattern}")
        return 0

    grand_total = 0
    grand_deleted = 0

    for f in files:
        rel_posix = f.relative_to(in_dir).as_posix()
        if rel_posix in visited:
            print(f"[SKIP visited] {rel_posix}")
            continue

        out_path = (out_dir / f.relative_to(in_dir)) if not args.inplace else f

        try:
            if args.inplace and not args.dry_run:
                bak = unique_backup_path(f)
                shutil.copy2(f, bak)

            if args.subreddits:
                total, deleted = process_file_subreddits(
                    in_path=f,
                    out_path=out_path,
                    threshold=args.threshold,
                    margin=args.margin,
                    stopwords_threshold=args.stopwords_threshold,
                    zero_eps=args.zero_eps,
                    show_deleted=args.show_deleted,
                    confirm_all_deletions=(args.ask and (not args.noask)),
                    noask=args.noask,
                    dry_run=args.dry_run,
                    force_heuristic=args.force_heuristic,
                )
            else:
                raise RuntimeError("This v7 snippet focuses on --subreddits mode. Use your prior userexport handler or ask me to merge it in.")

            grand_total += total
            grand_deleted += deleted

            if not args.dry_run:
                append_visited(visited_path, rel_posix)
                visited.add(rel_posix)

        except Exception as e:
            print(f"[ERROR] Failed processing {rel_posix}: {e}")

    print("\n--- Summary ---")
    print(f"Items total={grand_total}, deleted={grand_deleted}, kept={grand_total - grand_deleted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
