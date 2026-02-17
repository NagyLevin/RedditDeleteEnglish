from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Set

VERSION = "2026-02-16_v9 (removed stopwords; langdetect HU -> hunspell -> delete)"

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
# Optional Hunspell (pyhun / phunspell)
# ============================
_HUNSPELL_AVAILABLE = False
_HUNSPELL_ENGINE = None
_HUNSPELL_ENGINE_NAME = ""


def _try_init_hunspell() -> None:
    """
    Prefer phunspell (pure python + bundled dictionaries in many setups).
    Fallback: hunspell binding if available and dict files found.
    """
    global _HUNSPELL_AVAILABLE, _HUNSPELL_ENGINE, _HUNSPELL_ENGINE_NAME

    # 1) phunspell
    try:
        import phunspell  # type: ignore

        _HUNSPELL_ENGINE = phunspell.Phunspell("hu_HU")
        _HUNSPELL_AVAILABLE = True
        _HUNSPELL_ENGINE_NAME = "phunspell(hu_HU)"
        return
    except Exception:
        pass

    # 2) hunspell binding (needs aff/dic files)
    try:
        from hunspell import HunSpell  # type: ignore

        aff_dic = _find_hunspell_files("hu_HU")
        if aff_dic is None:
            return
        aff, dic = aff_dic
        _HUNSPELL_ENGINE = HunSpell(str(dic), str(aff))
        _HUNSPELL_AVAILABLE = True
        _HUNSPELL_ENGINE_NAME = f"hunspell({dic.name})"
        return
    except Exception:
        return


def _find_hunspell_files(lang: str) -> Optional[Tuple[Path, Path]]:
    # Common locations (Linux/macOS custom). Windows users can pass files next to script or cwd.
    candidates = [
        Path.cwd(),
        Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd(),
        Path("/usr/share/hunspell"),
        Path("/usr/share/myspell"),
        Path("/usr/share/myspell/dicts"),
        Path("/Library/Spelling"),
    ]
    for d in candidates:
        aff = d / f"{lang}.aff"
        dic = d / f"{lang}.dic"
        if aff.exists() and dic.exists():
            return aff, dic
    return None


def hunspell_lookup(word: str) -> bool:
    if not _HUNSPELL_AVAILABLE or _HUNSPELL_ENGINE is None:
        return False

    # phunspell: .lookup(str)->bool
    if hasattr(_HUNSPELL_ENGINE, "lookup"):
        try:
            return bool(_HUNSPELL_ENGINE.lookup(word))
        except Exception:
            return False

    # hunspell binding: .spell(str)->bool
    if hasattr(_HUNSPELL_ENGINE, "spell"):
        try:
            return bool(_HUNSPELL_ENGINE.spell(word))
        except Exception:
            return False

    return False


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


def make_preview(s: str, n: int = 140) -> str:
    s = s.strip().replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    return (s[:n] + "...") if len(s) > n else s


# ============================
# Langdetect helpers
# ============================
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


# ============================
# Hunspell score
# ============================
def hunspell_hu_ratio(text: str, min_word_len: int = 3) -> Optional[float]:
    """
    Returns ratio of words recognized by Hungarian Hunspell dictionary.
    None if hunspell engine not available.
    """
    if not _HUNSPELL_AVAILABLE:
        return None

    words = tokenize_words(clean_for_lang(text))
    words = [w for w in words if len(w) >= min_word_len]
    if not words:
        return 0.0

    ok = 0
    for w in words:
        # try word as-is
        if hunspell_lookup(w):
            ok += 1
            continue
        # try a simple de-hyphenation split (common in exports)
        if "-" in w:
            parts = [p for p in w.split("-") if p]
            if parts and all(hunspell_lookup(p) for p in parts):
                ok += 1

    return ok / len(words)


# ============================
# Decisions
# ============================
@dataclass
class Decision:
    delete: bool
    ld_hu: float
    ld_top_lang: str
    ld_top_prob: float
    hs_hu: float
    preview: str
    kind: str
    reason: str  # hu_langdetect, hu_hunspell, langdetect_xx, unknown, cjk, keep, ask, ask_zero


def decide_langdetect_then_hunspell(
    text: str,
    ld_hu_threshold: float,
    ld_any_threshold: float,
    hunspell_threshold: float,
    margin: float,
    zero_eps: float,
    min_words: int,
    hunspell_min_word_len: int,
    kind: str,
    force_heuristic: bool,  # kept for CLI compatibility; no effect here
) -> Decision:
    prev = make_preview(text)
    cleaned = clean_for_lang(text)
    words = tokenize_words(cleaned)

    if not words:
        return Decision(False, 0.0, "", 0.0, 0.0, prev, kind, "keep")

    if has_cjk(cleaned):
        return Decision(True, 0.0, "", 0.0, 0.0, prev, kind, "cjk")

    # If extremely short, langdetect can be noisy; still try both then fall back.
    ld_hu = 0.0
    ld_top_lang = ""
    ld_top_prob = 0.0

    if _LANGDETECT_AVAILABLE and not force_heuristic:
        r = _lang_prob_ratio_langdetect(cleaned, "hu")
        ld_hu = float(r) if r is not None else 0.0

        top = detect_top_lang(cleaned)
        if top is not None:
            ld_top_lang, ld_top_prob = top

        # 1) primary: langdetect says HU strongly
        if ld_hu >= ld_hu_threshold:
            return Decision(False, ld_hu, ld_top_lang, ld_top_prob, 0.0, prev, kind, "hu_langdetect")

    # 2) fallback: hunspell ratio
    hs = hunspell_hu_ratio(cleaned, min_word_len=hunspell_min_word_len)
    hs_hu = float(hs) if hs is not None else 0.0

    if hs is not None and hs_hu >= hunspell_threshold:
        return Decision(False, ld_hu, ld_top_lang, ld_top_prob, hs_hu, prev, kind, "hu_hunspell")

    # ask/ask_zero behavior (optional)
    # ask_zero: no evidence (langdetect ~0 AND hunspell ~0) OR too few words
    too_short = len([w for w in words if len(w) >= hunspell_min_word_len]) < min_words
    if (ld_hu <= zero_eps and hs_hu <= zero_eps) or too_short:
        return Decision(False, ld_hu, ld_top_lang, ld_top_prob, hs_hu, prev, kind, "ask_zero")

    # ask: near thresholds
    near_ld = (ld_hu_threshold - margin) <= ld_hu < ld_hu_threshold
    near_hs = (hunspell_threshold - margin) <= hs_hu < hunspell_threshold
    if near_ld or near_hs:
        return Decision(False, ld_hu, ld_top_lang, ld_top_prob, hs_hu, prev, kind, "ask")

    # otherwise delete
    if ld_top_lang and ld_top_prob >= ld_any_threshold and ld_top_lang != "hu":
        return Decision(True, ld_hu, ld_top_lang, ld_top_prob, hs_hu, prev, kind, f"langdetect_{ld_top_lang}")

    return Decision(True, ld_hu, ld_top_lang, ld_top_prob, hs_hu, prev, kind, "unknown")


def handle_prompt(dec: Decision, noask: bool) -> bool:
    """
    Returns delete? bool
    --noask:
      ask_zero => DELETE (because user requested: if none recognized => delete)
      ask => DELETE
    """
    if noask:
        return True  # both ask / ask_zero -> delete

    if dec.reason == "ask_zero":
        print("\n[ASK_ZERO] Nem megbízható / túl rövid / nincs elég bizonyíték -> kézi döntés.")
    else:
        print("\n[AMBIGUOUS] Küszöb környéke -> kézi döntés.")

    print(
        f"Kind: {dec.kind} | ld_hu={dec.ld_hu:.6f} | "
        f"top={dec.ld_top_lang}:{dec.ld_top_prob:.3f} | hs_hu={dec.hs_hu:.6f} | reason={dec.reason}"
    )
    print(f"Preview: {dec.preview}")

    ans = input("Döntés? (y=megtart, n/Enter=töröl) [y/N]: ").strip().lower()
    if ans == "y":
        return False
    return True


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


def unique_backup_path(path: Path) -> Path:
    base = path.with_suffix(path.suffix + ".bak")
    if not base.exists():
        return base
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_suffix(path.suffix + f".bak_{ts}")


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


def process_file_subreddits(
    in_path: Path,
    out_path: Path,
    ld_hu_threshold: float,
    ld_any_threshold: float,
    hunspell_threshold: float,
    margin: float,
    zero_eps: float,
    min_words: int,
    hunspell_min_word_len: int,
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
        post_dec = decide_langdetect_then_hunspell(
            post_text,
            ld_hu_threshold=ld_hu_threshold,
            ld_any_threshold=ld_any_threshold,
            hunspell_threshold=hunspell_threshold,
            margin=margin,
            zero_eps=zero_eps,
            min_words=min_words,
            hunspell_min_word_len=hunspell_min_word_len,
            kind="Post",
            force_heuristic=force_heuristic,
        )

        if post_dec.reason in {"ask", "ask_zero"}:
            delete_post = handle_prompt(post_dec, noask=noask)
        else:
            delete_post = post_dec.delete

        # optional confirm for non-ambiguous deletions
        if delete_post and (not noask) and confirm_all_deletions and (post_dec.reason not in {"ask", "ask_zero"}):
            print(
                f"\n[{in_path.name}] Candidate DELETE: Post | ld_hu={post_dec.ld_hu:.6f} | "
                f"top={post_dec.ld_top_lang}:{post_dec.ld_top_prob:.3f} | hs_hu={post_dec.hs_hu:.6f} | reason={post_dec.reason}"
            )
            print(f"Preview: {post_dec.preview}")
            ans = input("Töröljem a posztot (kommentekkel együtt)? [y/N] ").strip().lower()
            delete_post = (ans == "y")

        if show_deleted:
            if delete_post:
                print(
                    f"[DELETED] {in_path.name} | Post | ld_hu={post_dec.ld_hu:.6f} | "
                    f"top={post_dec.ld_top_lang}:{post_dec.ld_top_prob:.3f} | hs_hu={post_dec.hs_hu:.6f} | "
                    f"reason={post_dec.reason} | {post_dec.preview}"
                )
            else:
                if post_dec.reason in {"hu_langdetect", "hu_hunspell"}:
                    print(
                        f"[KEPT] {in_path.name} | Post | ld_hu={post_dec.ld_hu:.6f} | "
                        f"top={post_dec.ld_top_lang}:{post_dec.ld_top_prob:.3f} | hs_hu={post_dec.hs_hu:.6f} | "
                        f"reason={post_dec.reason} | {post_dec.preview}"
                    )

        if delete_post:
            deleted += 1
            continue

        kept.append(pre)

        # COMMENTS
        for c in comments:
            total += 1
            c_text = extract_subreddit_comment_text(c)
            c_dec = decide_langdetect_then_hunspell(
                c_text,
                ld_hu_threshold=ld_hu_threshold,
                ld_any_threshold=ld_any_threshold,
                hunspell_threshold=hunspell_threshold,
                margin=margin,
                zero_eps=zero_eps,
                min_words=min_words,
                hunspell_min_word_len=hunspell_min_word_len,
                kind="Comment",
                force_heuristic=force_heuristic,
            )

            if c_dec.reason in {"ask", "ask_zero"}:
                delete_c = handle_prompt(c_dec, noask=noask)
            else:
                delete_c = c_dec.delete

            if delete_c and (not noask) and confirm_all_deletions and (c_dec.reason not in {"ask", "ask_zero"}):
                print(
                    f"\n[{in_path.name}] Candidate DELETE: Comment | ld_hu={c_dec.ld_hu:.6f} | "
                    f"top={c_dec.ld_top_lang}:{c_dec.ld_top_prob:.3f} | hs_hu={c_dec.hs_hu:.6f} | reason={c_dec.reason}"
                )
                print(f"Preview: {c_dec.preview}")
                ans = input("Töröljem a kommentet? [y/N] ").strip().lower()
                delete_c = (ans == "y")

            if show_deleted:
                if delete_c:
                    print(
                        f"[DELETED] {in_path.name} | Comment | ld_hu={c_dec.ld_hu:.6f} | "
                        f"top={c_dec.ld_top_lang}:{c_dec.ld_top_prob:.3f} | hs_hu={c_dec.hs_hu:.6f} | "
                        f"reason={c_dec.reason} | {c_dec.preview}"
                    )
                else:
                    if c_dec.reason in {"hu_langdetect", "hu_hunspell"}:
                        print(
                            f"[KEPT] {in_path.name} | Comment | ld_hu={c_dec.ld_hu:.6f} | "
                            f"top={c_dec.ld_top_lang}:{c_dec.ld_top_prob:.3f} | hs_hu={c_dec.hs_hu:.6f} | "
                            f"reason={c_dec.reason} | {c_dec.preview}"
                        )

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
    _try_init_hunspell()

    p = argparse.ArgumentParser(
        description="Filter Reddit exports: keep HU via langdetect, fallback to Hunspell, else delete; visited + backups."
    )
    p.add_argument("--version", action="version", version=VERSION)

    p.add_argument("-inputfolder", "--inputfolder", required=True, help="Folder that contains exported .txt files.")
    p.add_argument("--pattern", default="*.txt", help="Glob pattern for files (default: *.txt).")
    p.add_argument("--recursive", action="store_true", help="Process subfolders too.")

    # Detection thresholds
    p.add_argument("--ld-hu-threshold", "--threshold", dest="ld_hu_threshold", type=float, default=0.70,
                   help="Keep if langdetect HU probability >= this.")
    p.add_argument("--ld-any-threshold", type=float, default=0.70,
                   help="If langdetect top prob >= this and not HU, reason will be langdetect_xx.")
    p.add_argument("--hunspell-threshold", "--stopwords-threshold", dest="hunspell_threshold", type=float, default=0.55,
                   help="Keep if Hunspell HU word ratio >= this.")

    # Ambiguity handling (optional prompts)
    p.add_argument("--margin", type=float, default=0.10, help="Near-threshold band that triggers ask.")
    p.add_argument("--zero-eps", type=float, default=0.02, help="Near-zero evidence threshold for ask_zero.")
    p.add_argument("--min-words", type=int, default=4, help="If too few words, treat as ask_zero.")
    p.add_argument("--hunspell-min-word-len", type=int, default=3, help="Ignore shorter words for Hunspell ratio.")

    p.add_argument("--show-deleted", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--inplace", action="store_true")
    p.add_argument("--outputfolder", default=None)

    # kept for compatibility with your earlier CLI; no effect in this version
    p.add_argument("--force-heuristic", action="store_true")

    p.add_argument("--subreddits", action="store_true")

    p.add_argument("--ask", action="store_true")
    p.add_argument("--noask", action="store_true",
                   help="Disable ALL prompts. ask_zero/ask => DELETE (per your request).")

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

    detector_info = "langdetect" if _LANGDETECT_AVAILABLE and not args.force_heuristic else "langdetect(unavailable)"
    hunspell_info = _HUNSPELL_ENGINE_NAME if _HUNSPELL_AVAILABLE else "hunspell(unavailable)"

    print(f"Version: {VERSION}")
    print(
        f"Detectors: {detector_info} -> {hunspell_info} | "
        f"ld_hu_threshold={args.ld_hu_threshold:.2f} | hunspell_threshold={args.hunspell_threshold:.2f} | "
        f"margin={args.margin:.2f} | zero_eps={args.zero_eps:.6f} | subreddits={args.subreddits}"
    )
    print(f"Visited file (root): {visited_path} | entries={len(visited)}")

    if not _LANGDETECT_AVAILABLE:
        print("WARN: langdetect not available. HU detection will rely on Hunspell only.")
    if not _HUNSPELL_AVAILABLE:
        print("WARN: Hunspell not available. Fallback step is disabled (install phunspell or hunspell + hu_HU dict).")

    files_all = iter_files(in_dir, args.recursive, args.pattern)

    files: List[Path] = []
    for f in files_all:
        if f.resolve() == visited_path:
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
                    ld_hu_threshold=args.ld_hu_threshold,
                    ld_any_threshold=args.ld_any_threshold,
                    hunspell_threshold=args.hunspell_threshold,
                    margin=args.margin,
                    zero_eps=args.zero_eps,
                    min_words=args.min_words,
                    hunspell_min_word_len=args.hunspell_min_word_len,
                    show_deleted=args.show_deleted,
                    confirm_all_deletions=(args.ask and (not args.noask)),
                    noask=args.noask,
                    dry_run=args.dry_run,
                    force_heuristic=args.force_heuristic,
                )
            else:
                raise RuntimeError("This v9 code focuses on --subreddits mode (as your v7 snippet did).")

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
