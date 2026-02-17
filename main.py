from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Set

VERSION = "2026-02-17_v16 (after manual item-KEEP -> word-collect; ask only for words unknown to ALL detectors; szotar.txt in script root + auto-create)"

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
# Optional Hunspell (HU + EN)
# ============================
_HUNSPELL_AVAILABLE = False
_HUNSPELL_ENGINE = None
_HUNSPELL_ENGINE_NAME = ""

_HUNSPELL_EN_AVAILABLE = False
_HUNSPELL_EN_ENGINE = None
_HUNSPELL_EN_ENGINE_NAME = ""


def _find_hunspell_files(lang: str) -> Optional[Tuple[Path, Path]]:
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


def _try_init_hunspell_hu() -> None:
    global _HUNSPELL_AVAILABLE, _HUNSPELL_ENGINE, _HUNSPELL_ENGINE_NAME

    try:
        import phunspell  # type: ignore

        _HUNSPELL_ENGINE = phunspell.Phunspell("hu_HU")
        _HUNSPELL_AVAILABLE = True
        _HUNSPELL_ENGINE_NAME = "phunspell(hu_HU)"
        return
    except Exception:
        pass

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


def _try_init_hunspell_en() -> None:
    global _HUNSPELL_EN_AVAILABLE, _HUNSPELL_EN_ENGINE, _HUNSPELL_EN_ENGINE_NAME

    try:
        import phunspell  # type: ignore

        _HUNSPELL_EN_ENGINE = phunspell.Phunspell("en_US")
        _HUNSPELL_EN_AVAILABLE = True
        _HUNSPELL_EN_ENGINE_NAME = "phunspell(en_US)"
        return
    except Exception:
        pass

    try:
        from hunspell import HunSpell  # type: ignore

        aff_dic = _find_hunspell_files("en_US")
        if aff_dic is None:
            return
        aff, dic = aff_dic
        _HUNSPELL_EN_ENGINE = HunSpell(str(dic), str(aff))
        _HUNSPELL_EN_AVAILABLE = True
        _HUNSPELL_EN_ENGINE_NAME = f"hunspell({dic.name})"
        return
    except Exception:
        return


def hunspell_hu_lookup(word: str) -> bool:
    if not _HUNSPELL_AVAILABLE or _HUNSPELL_ENGINE is None:
        return False

    if hasattr(_HUNSPELL_ENGINE, "lookup"):
        try:
            return bool(_HUNSPELL_ENGINE.lookup(word))
        except Exception:
            return False

    if hasattr(_HUNSPELL_ENGINE, "spell"):
        try:
            return bool(_HUNSPELL_ENGINE.spell(word))
        except Exception:
            return False

    return False


def hunspell_en_lookup(word: str) -> bool:
    if not _HUNSPELL_EN_AVAILABLE or _HUNSPELL_EN_ENGINE is None:
        return False

    if hasattr(_HUNSPELL_EN_ENGINE, "lookup"):
        try:
            return bool(_HUNSPELL_EN_ENGINE.lookup(word))
        except Exception:
            return False

    if hasattr(_HUNSPELL_EN_ENGINE, "spell"):
        try:
            return bool(_HUNSPELL_EN_ENGINE.spell(word))
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


def detect_word_top_lang(word: str) -> Tuple[str, float]:
    """
    Single-word langdetect: noisy, so we only trust if prob >= --word-lang-prob.
    """
    w = word.strip().lower()
    if (not _LANGDETECT_AVAILABLE) or (detect_langs is None) or (len(w) < 4):
        return "", 0.0
    try:
        langs = detect_langs(w)
    except Exception:
        return "", 0.0
    if not langs:
        return "", 0.0
    top = langs[0]
    return (getattr(top, "lang", ""), float(getattr(top, "prob", 0.0)))


# ============================
# Hunspell HU ratio
# ============================
def hunspell_hu_ratio(text: str, min_word_len: int = 3) -> Optional[float]:
    if not _HUNSPELL_AVAILABLE:
        return None

    words = tokenize_words(clean_for_lang(text))
    words = [w for w in words if len(w) >= min_word_len]
    if not words:
        return 0.0

    ok = 0
    for w in words:
        if hunspell_hu_lookup(w):
            ok += 1
            continue
        if "-" in w:
            parts = [p for p in w.split("-") if p]
            if parts and all(hunspell_hu_lookup(p) for p in parts):
                ok += 1

    return ok / len(words)


# ============================
# szotar.txt (HU/IDEGEN) - script root forced + auto-create
# ============================
DICT_HU_HEADER = "[HU]"
DICT_FOREIGN_HEADER = "[IDEGEN]"


def load_szotar(path: Path) -> Tuple[Set[str], Set[str]]:
    if not path.exists():
        return set(), set()

    hu: Set[str] = set()
    foreign: Set[str] = set()
    section = ""

    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        up = s.upper()
        if up == DICT_HU_HEADER:
            section = "HU"
            continue
        if up == DICT_FOREIGN_HEADER:
            section = "FOREIGN"
            continue

        w = s.lower()
        if section == "HU":
            hu.add(w)
        elif section == "FOREIGN":
            foreign.add(w)
        else:
            hu.add(w)

    return hu, foreign


def save_szotar(path: Path, hu: Set[str], foreign: Set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append(f"# szotar.txt autogenerated | {now}\n")
    lines.append(f"{DICT_HU_HEADER}\n")
    for w in sorted(hu):
        lines.append(w + "\n")
    lines.append("\n")
    lines.append(f"{DICT_FOREIGN_HEADER}\n")
    for w in sorted(foreign):
        lines.append(w + "\n")
    path.write_text("".join(lines), encoding="utf-8")


class SzotarStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.hu, self.foreign = load_szotar(path)
        self._dirty = False

        # ALWAYS create file at startup if missing
        if not self.path.exists():
            save_szotar(self.path, self.hu, self.foreign)

    def save_if_dirty(self) -> None:
        if self._dirty:
            save_szotar(self.path, self.hu, self.foreign)
            self._dirty = False

    def add_hu(self, w: str) -> None:
        w = w.lower()
        if w not in self.hu:
            self.hu.add(w)
            self._dirty = True

    def add_foreign(self, w: str) -> None:
        w = w.lower()
        if w not in self.foreign:
            self.foreign.add(w)
            self._dirty = True

    def contains_any(self, w: str) -> bool:
        w = w.lower()
        return (w in self.hu) or (w in self.foreign)


def ratio_in_set(words: List[str], s: Set[str], min_len: int) -> float:
    eligible = [w for w in words if len(w) >= min_len]
    if not eligible:
        return 0.0
    hit = sum(1 for w in eligible if w in s)
    return hit / len(eligible)


# ============================
# Word eligibility for prompting
# ============================
def is_known_by_any_detector(word: str, *, word_lang_prob: float) -> bool:
    """
    True => do NOT ask user about this word.
    We skip prompting if ANY detector recognizes it:
      - HU hunspell
      - EN hunspell
      - langdetect(word) says *any* language with prob >= word_lang_prob
    """
    w = word.strip().lower()
    if not w:
        return True

    if _HUNSPELL_AVAILABLE and hunspell_hu_lookup(w):
        return True

    if _HUNSPELL_EN_AVAILABLE and hunspell_en_lookup(w):
        return True

    lang, prob = detect_word_top_lang(w)
    if lang and prob >= word_lang_prob:
        return True

    return False


def collect_words_from_text_to_szotar(
    text: str,
    *,
    szotar: SzotarStore,
    min_word_len: int,
    word_lang_prob: float,
    noask: bool,
) -> None:
    """
    Ask word-by-word ONLY if:
      - not in szotar (HU or IDEGEN)
      - NOT recognized by any detector (HU hunspell / EN hunspell / langdetect(word)>=word_lang_prob for any lang)
    Then prompt:
      Enter/y -> add to HU
      n       -> add to IDEGEN
      s       -> skip
    """
    if noask:
        return

    cleaned = clean_for_lang(text)
    words = tokenize_words(cleaned)
    words = [w for w in words if len(w) >= min_word_len]
    if not words:
        return

    seen: Set[str] = set()
    ordered: List[str] = []
    for w in words:
        if w not in seen:
            seen.add(w)
            ordered.append(w)

    changed = False
    asked_any = False

    for w in ordered:
        if szotar.contains_any(w):
            continue

        # if any detector recognizes -> do NOT ask
        if is_known_by_any_detector(w, word_lang_prob=word_lang_prob):
            continue

        asked_any = True
        ans = input(f"Új/ISMERETLEN szó: '{w}' | HU szótárba? (Enter/y=igen, n=IDEGEN, s=skip) [Y/n/s]: ").strip().lower()
        if ans in {"", "y"}:
            szotar.add_hu(w)
            changed = True
        elif ans == "n":
            szotar.add_foreign(w)
            changed = True
        else:
            pass

    if asked_any and changed:
        szotar.save_if_dirty()


# ============================
# ITEM-level decision: AUTO keep only if HU strong; AUTO delete only if foreign strong; else ASK user
# ============================
@dataclass
class Decision:
    delete: bool
    ld_hu: float
    ld_top_lang: str
    ld_top_prob: float
    hs_hu: float
    dict_hu_ratio: float
    dict_foreign_ratio: float
    en_prob: float
    preview: str
    kind: str
    reason: str  # hu_..., foreign_..., ask, ask_zero


def decide_item(
    text: str,
    *,
    kind: str,
    force_heuristic: bool,
    # HU thresholds
    ld_hu_threshold: float,
    ld_any_threshold: float,
    hunspell_threshold: float,
    hunspell_min_word_len: int,
    # foreign thresholds
    en_item_threshold: float,
    # dictionary thresholds
    szotar: SzotarStore,
    dict_hu_threshold: float,
    dict_foreign_threshold: float,
    dict_min_word_len: int,
    # ask heuristics
    zero_eps: float,
    min_words: int,
) -> Decision:
    prev = make_preview(text)
    cleaned = clean_for_lang(text)
    words = tokenize_words(cleaned)

    if not words:
        return Decision(
            delete=False, ld_hu=0.0, ld_top_lang="", ld_top_prob=0.0, hs_hu=0.0,
            dict_hu_ratio=0.0, dict_foreign_ratio=0.0, en_prob=0.0,
            preview=prev, kind=kind, reason="ask_zero"
        )

    if has_cjk(cleaned):
        return Decision(
            delete=True, ld_hu=0.0, ld_top_lang="", ld_top_prob=0.0, hs_hu=0.0,
            dict_hu_ratio=0.0, dict_foreign_ratio=1.0, en_prob=0.0,
            preview=prev, kind=kind, reason="foreign_cjk"
        )

    dict_hu_ratio = ratio_in_set(words, szotar.hu, dict_min_word_len)
    dict_foreign_ratio = ratio_in_set(words, szotar.foreign, dict_min_word_len)

    ld_hu = 0.0
    ld_top_lang = ""
    ld_top_prob = 0.0
    en_prob = 0.0

    if _LANGDETECT_AVAILABLE and (not force_heuristic) and detect_langs is not None:
        r_hu = _lang_prob_ratio_langdetect(cleaned, "hu")
        ld_hu = float(r_hu) if r_hu is not None else 0.0

        r_en = _lang_prob_ratio_langdetect(cleaned, "en")
        en_prob = float(r_en) if r_en is not None else 0.0

        top = detect_top_lang(cleaned)
        if top is not None:
            ld_top_lang, ld_top_prob = top

    hs = hunspell_hu_ratio(cleaned, min_word_len=hunspell_min_word_len)
    hs_hu = float(hs) if hs is not None else 0.0

    hu_strong = (
        (dict_hu_ratio >= dict_hu_threshold)
        or (ld_hu >= ld_hu_threshold)
        or (ld_top_lang == "hu" and ld_top_prob >= ld_any_threshold)
        or (hs is not None and hs_hu >= hunspell_threshold)
    )
    if hu_strong:
        return Decision(
            delete=False,
            ld_hu=ld_hu, ld_top_lang=ld_top_lang, ld_top_prob=ld_top_prob, hs_hu=hs_hu,
            dict_hu_ratio=dict_hu_ratio, dict_foreign_ratio=dict_foreign_ratio, en_prob=en_prob,
            preview=prev, kind=kind, reason="hu_strong_keep"
        )

    foreign_strong = (
        (dict_foreign_ratio >= dict_foreign_threshold)
        or (en_prob >= en_item_threshold)
        or (ld_top_lang == "en" and ld_top_prob >= ld_any_threshold)
        or (ld_top_lang not in {"", "hu"} and ld_top_prob >= ld_any_threshold)
    )
    if foreign_strong:
        reason = "foreign_strong_delete"
        if dict_foreign_ratio >= dict_foreign_threshold:
            reason = "foreign_dict_delete"
        elif (ld_top_lang == "en" and ld_top_prob >= ld_any_threshold) or (en_prob >= en_item_threshold):
            reason = "foreign_en_delete"
        else:
            reason = f"foreign_top_{ld_top_lang}_delete"

        return Decision(
            delete=True,
            ld_hu=ld_hu, ld_top_lang=ld_top_lang, ld_top_prob=ld_top_prob, hs_hu=hs_hu,
            dict_hu_ratio=dict_hu_ratio, dict_foreign_ratio=dict_foreign_ratio, en_prob=en_prob,
            preview=prev, kind=kind, reason=reason
        )

    eligible_words = [w for w in words if len(w) >= hunspell_min_word_len]
    too_short = len(eligible_words) < min_words
    near_zero = (
        (ld_hu <= zero_eps)
        and (hs_hu <= zero_eps)
        and (en_prob <= zero_eps)
        and (dict_hu_ratio <= zero_eps)
        and (dict_foreign_ratio <= zero_eps)
        and (ld_top_prob <= zero_eps)
    )

    return Decision(
        delete=False,
        ld_hu=ld_hu, ld_top_lang=ld_top_lang, ld_top_prob=ld_top_prob, hs_hu=hs_hu,
        dict_hu_ratio=dict_hu_ratio, dict_foreign_ratio=dict_foreign_ratio, en_prob=en_prob,
        preview=prev, kind=kind, reason=("ask_zero" if (too_short or near_zero) else "ask")
    )


def handle_item_prompt(dec: Decision, noask: bool) -> bool:
    """
    return delete?
      y = keep
      else = delete
    if --noask: keep ambiguous (delete only if foreign_strong)
    """
    if noask:
        return False

    if dec.reason == "ask_zero":
        print("\n[ASK_ZERO] Túl kevés / nincs elég bizonyíték -> kézi döntés.")
    else:
        print("\n[ASK] Nem egyértelmű (nem HU>=thr és nem foreign>=thr) -> kézi döntés.")

    print(
        f"Kind: {dec.kind} | reason={dec.reason} | "
        f"ld_hu={dec.ld_hu:.6f} | top={dec.ld_top_lang}:{dec.ld_top_prob:.3f} | en_prob={dec.en_prob:.6f} | "
        f"hs_hu={dec.hs_hu:.6f} | dict_hu={dec.dict_hu_ratio:.3f} | dict_foreign={dec.dict_foreign_ratio:.3f}"
    )
    print(f"Preview: {dec.preview}")

    ans = input("Döntés? (y=MEGTART, n/Enter=TÖRÖL) [y/N]: ").strip().lower()
    return (ans != "y")


# ============================
# Sentence-level filtering (EN + szotar) + optional word collection (only unknown)
# ============================
@dataclass
class SentDecision:
    keep: bool
    reason: str
    en_prob: float
    top_lang: str
    top_prob: float
    foreign_ratio: float
    hu_ratio: float
    preview: str


def decide_sentence(
    sentence: str,
    *,
    ld_any_threshold: float,
    en_sentence_threshold: float,
    margin: float,
    dict_foreign_threshold: float,
    dict_hu_threshold: float,
    dict_min_word_len: int,
    szotar: SzotarStore,
) -> SentDecision:
    prev = make_preview(sentence)
    cleaned = clean_for_lang(sentence)

    if not cleaned:
        return SentDecision(True, "empty", 0.0, "", 0.0, 0.0, 0.0, prev)

    if has_cjk(cleaned):
        return SentDecision(False, "cjk", 0.0, "", 0.0, 1.0, 0.0, prev)

    words = tokenize_words(cleaned)

    foreign_ratio = ratio_in_set(words, szotar.foreign, dict_min_word_len)
    if foreign_ratio >= dict_foreign_threshold:
        return SentDecision(False, "idegen_list", 0.0, "", 0.0, foreign_ratio, 0.0, prev)

    hu_ratio = ratio_in_set(words, szotar.hu, dict_min_word_len)
    if hu_ratio >= dict_hu_threshold:
        return SentDecision(True, "kept_bc_szotar", 0.0, "", 0.0, foreign_ratio, hu_ratio, prev)

    en_prob = 0.0
    top_lang = ""
    top_prob = 0.0

    if _LANGDETECT_AVAILABLE and detect_langs is not None:
        r = _lang_prob_ratio_langdetect(cleaned, "en")
        en_prob = float(r) if r is not None else 0.0
        top = detect_top_lang(cleaned)
        if top is not None:
            top_lang, top_prob = top

        strong_en = (en_prob >= en_sentence_threshold) or (top_lang == "en" and top_prob >= ld_any_threshold)
        if strong_en:
            return SentDecision(False, "en_strong", en_prob, top_lang, top_prob, foreign_ratio, hu_ratio, prev)

        near_en = ((en_sentence_threshold - margin) <= en_prob < en_sentence_threshold) or (
            top_lang == "en" and (ld_any_threshold - margin) <= top_prob < ld_any_threshold
        )
        if near_en:
            return SentDecision(True, "en_ambiguous_ask", en_prob, top_lang, top_prob, foreign_ratio, hu_ratio, prev)

    return SentDecision(True, "keep", en_prob, top_lang, top_prob, foreign_ratio, hu_ratio, prev)


def prompt_keep_ambiguous_sentence(sd: SentDecision, kind: str, noask: bool) -> bool:
    if noask:
        return True

    print("\n[AMBIGUOUS EN SENTENCE] Küszöb közelében -> kézi döntés.")
    print(
        f"Kind: {kind} | en_prob={sd.en_prob:.6f} | top={sd.top_lang}:{sd.top_prob:.3f} | "
        f"foreign_ratio={sd.foreign_ratio:.3f} | hu_ratio={sd.hu_ratio:.3f}"
    )
    print(f"Preview: {sd.preview}")
    ans = input("Döntés? (y=MEGTART, n/Enter=TÖRÖL) [y/N]: ").strip().lower()
    return ans == "y"


def filter_text_sentences(
    text: str,
    *,
    context_kind: str,
    file_name: str,
    show_deleted: bool,
    noask: bool,
    ld_any_threshold: float,
    en_sentence_threshold: float,
    margin: float,
    dict_foreign_threshold: float,
    dict_hu_threshold: float,
    dict_min_word_len: int,
    # NEW: if user keeps ambiguous sentence, collect ONLY unknown words
    dict_collect_min_word_len: int,
    word_lang_prob: float,
    szotar: SzotarStore,
) -> str:
    chunks = split_into_chunks(text)
    if not chunks:
        return text

    kept: List[str] = []

    for ch in chunks:
        sd = decide_sentence(
            ch,
            ld_any_threshold=ld_any_threshold,
            en_sentence_threshold=en_sentence_threshold,
            margin=margin,
            dict_foreign_threshold=dict_foreign_threshold,
            dict_hu_threshold=dict_hu_threshold,
            dict_min_word_len=dict_min_word_len,
            szotar=szotar,
        )

        if sd.reason == "kept_bc_szotar":
            print(f"[KEPT, bc szotar.txt] {file_name} | {context_kind} | hu_ratio={sd.hu_ratio:.3f} | {sd.preview}")
            kept.append(ch)
            continue

        if sd.reason == "en_ambiguous_ask":
            keep_it = prompt_keep_ambiguous_sentence(sd, kind=f"{file_name} | {context_kind}", noask=noask)
            if keep_it:
                kept.append(ch)
                # collect unknown words from this kept ambiguous sentence
                collect_words_from_text_to_szotar(
                    ch,
                    szotar=szotar,
                    min_word_len=dict_collect_min_word_len,
                    word_lang_prob=word_lang_prob,
                    noask=noask,
                )
            else:
                if show_deleted:
                    print(
                        f"[DELETED_SENT] {file_name} | {context_kind} | reason=en_ambiguous_user_delete | "
                        f"en_prob={sd.en_prob:.3f} top={sd.top_lang}:{sd.top_prob:.3f} | {sd.preview}"
                    )
            continue

        if not sd.keep:
            if show_deleted:
                print(
                    f"[DELETED_SENT] {file_name} | {context_kind} | reason={sd.reason} | "
                    f"en_prob={sd.en_prob:.3f} top={sd.top_lang}:{sd.top_prob:.3f} | "
                    f"foreign_ratio={sd.foreign_ratio:.3f} hu_ratio={sd.hu_ratio:.3f} | {sd.preview}"
                )
            continue

        kept.append(ch)

    return "\n".join(kept).strip()


# ============================
# Visited / backups
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
# Subreddit dump parsing
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


def rewrite_post_pre_block(
    pre_block: str,
    *,
    file_name: str,
    show_deleted: bool,
    noask: bool,
    ld_any_threshold: float,
    en_sentence_threshold: float,
    margin: float,
    dict_foreign_threshold: float,
    dict_hu_threshold: float,
    dict_min_word_len: int,
    dict_collect_min_word_len: int,
    word_lang_prob: float,
    szotar: SzotarStore,
) -> str:
    lines = pre_block.splitlines(keepends=True)
    out: List[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        m = BYLINE_RE.match(line.strip())
        if m:
            user = m.group(1)
            title = m.group(2)
            new_title = filter_text_sentences(
                title,
                context_kind="PostTitle",
                file_name=file_name,
                show_deleted=show_deleted,
                noask=noask,
                ld_any_threshold=ld_any_threshold,
                en_sentence_threshold=en_sentence_threshold,
                margin=margin,
                dict_foreign_threshold=dict_foreign_threshold,
                dict_hu_threshold=dict_hu_threshold,
                dict_min_word_len=dict_min_word_len,
                dict_collect_min_word_len=dict_collect_min_word_len,
                word_lang_prob=word_lang_prob,
                szotar=szotar,
            )
            eol = "\n" if line.endswith("\n") else ""
            out.append(f"by {user}: {new_title}{eol}")
            i += 1
            continue

        if BODY_START_RE.match(line):
            out.append(line)
            i += 1
            body_lines: List[str] = []
            while i < len(lines) and (not COMMENT_START_RE.match(lines[i])):
                body_lines.append(lines[i])
                i += 1

            raw_body_parts: List[str] = []
            for bl in body_lines:
                txt = bl[4:] if bl.startswith("    ") else bl.lstrip()
                raw_body_parts.append(txt.rstrip("\n"))
            raw_body = "\n".join(raw_body_parts).strip()

            new_body = filter_text_sentences(
                raw_body,
                context_kind="PostBody",
                file_name=file_name,
                show_deleted=show_deleted,
                noask=noask,
                ld_any_threshold=ld_any_threshold,
                en_sentence_threshold=en_sentence_threshold,
                margin=margin,
                dict_foreign_threshold=dict_foreign_threshold,
                dict_hu_threshold=dict_hu_threshold,
                dict_min_word_len=dict_min_word_len,
                dict_collect_min_word_len=dict_collect_min_word_len,
                word_lang_prob=word_lang_prob,
                szotar=szotar,
            )

            if new_body.strip():
                for ln in new_body.splitlines():
                    out.append("    " + ln.rstrip() + "\n")
            continue

        out.append(line)
        i += 1

    return "".join(out)


def rewrite_comment_block(
    comment_block: str,
    *,
    file_name: str,
    show_deleted: bool,
    noask: bool,
    ld_any_threshold: float,
    en_sentence_threshold: float,
    margin: float,
    dict_foreign_threshold: float,
    dict_hu_threshold: float,
    dict_min_word_len: int,
    dict_collect_min_word_len: int,
    word_lang_prob: float,
    szotar: SzotarStore,
) -> Optional[str]:
    lines = comment_block.splitlines(keepends=True)
    if not lines:
        return comment_block

    start_idx = None
    for i, l in enumerate(lines):
        if COMMENT_START_RE.match(l):
            start_idx = i
            break
    if start_idx is None:
        return comment_block

    author_prefix = ""
    for j in range(start_idx + 1, len(lines)):
        raw = lines[j]
        stripped = (raw[4:] if raw.startswith("    ") else raw.lstrip()).rstrip("\n")
        if stripped.strip() == "":
            continue
        if ":" in stripped:
            author = stripped.split(":", 1)[0].strip()
            author_prefix = author + ":"
        break

    comment_text = extract_subreddit_comment_text(comment_block)
    new_text = filter_text_sentences(
        comment_text,
        context_kind="CommentBody",
        file_name=file_name,
        show_deleted=show_deleted,
        noask=noask,
        ld_any_threshold=ld_any_threshold,
        en_sentence_threshold=en_sentence_threshold,
        margin=margin,
        dict_foreign_threshold=dict_foreign_threshold,
        dict_hu_threshold=dict_hu_threshold,
        dict_min_word_len=dict_min_word_len,
        dict_collect_min_word_len=dict_collect_min_word_len,
        word_lang_prob=word_lang_prob,
        szotar=szotar,
    )

    if not new_text.strip():
        return None

    new_lines: List[str] = []
    new_lines.extend(lines[: start_idx + 1])

    text_lines = new_text.splitlines()
    if author_prefix:
        first_line = text_lines[0] if text_lines else ""
        new_lines.append("    " + author_prefix + ((" " + first_line) if first_line else "") + "\n")
        for ln in text_lines[1:]:
            new_lines.append("    " + ln.rstrip() + "\n")
    else:
        for ln in text_lines:
            new_lines.append("    " + ln.rstrip() + "\n")

    return "".join(new_lines)


def process_file_subreddits(
    in_path: Path,
    out_path: Path,
    *,
    # item thresholds
    ld_hu_threshold: float,
    ld_any_threshold: float,
    hunspell_threshold: float,
    hunspell_min_word_len: int,
    en_item_threshold: float,
    dict_foreign_threshold: float,
    dict_hu_threshold: float,
    dict_min_word_len: int,
    zero_eps: float,
    min_words: int,
    # sentence thresholds
    en_sentence_threshold: float,
    margin: float,
    # word collection thresholds
    dict_collect_min_word_len: int,
    word_lang_prob: float,
    # runtime
    show_deleted: bool,
    noask: bool,
    dry_run: bool,
    force_heuristic: bool,
    szotar: SzotarStore,
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

        # ---- POST item-level decision
        total += 1
        post_text = extract_subreddit_post_text(pre)
        post_dec = decide_item(
            post_text,
            kind="Post",
            force_heuristic=force_heuristic,
            ld_hu_threshold=ld_hu_threshold,
            ld_any_threshold=ld_any_threshold,
            hunspell_threshold=hunspell_threshold,
            hunspell_min_word_len=hunspell_min_word_len,
            en_item_threshold=en_item_threshold,
            szotar=szotar,
            dict_hu_threshold=dict_hu_threshold,
            dict_foreign_threshold=dict_foreign_threshold,
            dict_min_word_len=dict_min_word_len,
            zero_eps=zero_eps,
            min_words=min_words,
        )

        manual_post_keep = False
        if post_dec.reason in {"ask", "ask_zero"}:
            delete_post = handle_item_prompt(post_dec, noask=noask)
            manual_post_keep = (not delete_post)
        else:
            delete_post = post_dec.delete

        if show_deleted:
            tag = "DELETED" if delete_post else "KEPT"
            print(
                f"[{tag}] {in_path.name} | Post | "
                f"ld_hu={post_dec.ld_hu:.6f} | top={post_dec.ld_top_lang}:{post_dec.ld_top_prob:.3f} | "
                f"en_prob={post_dec.en_prob:.6f} | hs_hu={post_dec.hs_hu:.6f} | "
                f"dict_hu={post_dec.dict_hu_ratio:.3f} | dict_foreign={post_dec.dict_foreign_ratio:.3f} | "
                f"reason={post_dec.reason} | {post_dec.preview}"
            )

        if delete_post:
            deleted += 1
            continue

        # sentence-level cleanup inside post
        new_pre = rewrite_post_pre_block(
            pre,
            file_name=in_path.name,
            show_deleted=show_deleted,
            noask=noask,
            ld_any_threshold=ld_any_threshold,
            en_sentence_threshold=en_sentence_threshold,
            margin=margin,
            dict_foreign_threshold=dict_foreign_threshold,
            dict_hu_threshold=dict_hu_threshold,
            dict_min_word_len=dict_min_word_len,
            dict_collect_min_word_len=dict_collect_min_word_len,
            word_lang_prob=word_lang_prob,
            szotar=szotar,
        )

        new_post_text = extract_subreddit_post_text(new_pre).strip()
        if not new_post_text:
            if show_deleted:
                print(f"[DELETED] {in_path.name} | Post | reason=post_became_empty_after_sentence_filter")
            deleted += 1
            continue

        # NEW: if user manually kept ambiguous post -> word-collect from the kept (filtered) post text
        if manual_post_keep:
            collect_words_from_text_to_szotar(
                new_post_text,
                szotar=szotar,
                min_word_len=dict_collect_min_word_len,
                word_lang_prob=word_lang_prob,
                noask=noask,
            )

        kept.append(new_pre)

        # ---- COMMENTS
        for c in comments:
            total += 1
            c_text = extract_subreddit_comment_text(c)
            c_dec = decide_item(
                c_text,
                kind="Comment",
                force_heuristic=force_heuristic,
                ld_hu_threshold=ld_hu_threshold,
                ld_any_threshold=ld_any_threshold,
                hunspell_threshold=hunspell_threshold,
                hunspell_min_word_len=hunspell_min_word_len,
                en_item_threshold=en_item_threshold,
                szotar=szotar,
                dict_hu_threshold=dict_hu_threshold,
                dict_foreign_threshold=dict_foreign_threshold,
                dict_min_word_len=dict_min_word_len,
                zero_eps=zero_eps,
                min_words=min_words,
            )

            manual_comment_keep = False
            if c_dec.reason in {"ask", "ask_zero"}:
                delete_c = handle_item_prompt(c_dec, noask=noask)
                manual_comment_keep = (not delete_c)
            else:
                delete_c = c_dec.delete

            if show_deleted:
                tag = "DELETED" if delete_c else "KEPT"
                print(
                    f"[{tag}] {in_path.name} | Comment | "
                    f"ld_hu={c_dec.ld_hu:.6f} | top={c_dec.ld_top_lang}:{c_dec.ld_top_prob:.3f} | "
                    f"en_prob={c_dec.en_prob:.6f} | hs_hu={c_dec.hs_hu:.6f} | "
                    f"dict_hu={c_dec.dict_hu_ratio:.3f} | dict_foreign={c_dec.dict_foreign_ratio:.3f} | "
                    f"reason={c_dec.reason} | {c_dec.preview}"
                )

            if delete_c:
                deleted += 1
                continue

            new_c = rewrite_comment_block(
                c,
                file_name=in_path.name,
                show_deleted=show_deleted,
                noask=noask,
                ld_any_threshold=ld_any_threshold,
                en_sentence_threshold=en_sentence_threshold,
                margin=margin,
                dict_foreign_threshold=dict_foreign_threshold,
                dict_hu_threshold=dict_hu_threshold,
                dict_min_word_len=dict_min_word_len,
                dict_collect_min_word_len=dict_collect_min_word_len,
                word_lang_prob=word_lang_prob,
                szotar=szotar,
            )

            if new_c is None:
                if show_deleted:
                    print(f"[DELETED] {in_path.name} | Comment | reason=comment_became_empty_after_sentence_filter")
                deleted += 1
                continue

            # NEW: if user manually kept ambiguous comment -> word-collect from the kept (filtered) comment text
            if manual_comment_keep:
                new_c_text = extract_subreddit_comment_text(new_c).strip()
                if new_c_text:
                    collect_words_from_text_to_szotar(
                        new_c_text,
                        szotar=szotar,
                        min_word_len=dict_collect_min_word_len,
                        word_lang_prob=word_lang_prob,
                        noask=noask,
                    )

            kept.append(new_c)

    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("".join(kept), encoding="utf-8")

    return total, deleted


# ============================
# Root helpers
# ============================
def script_root_dir() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd().resolve()


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
    _try_init_hunspell_hu()
    _try_init_hunspell_en()

    p = argparse.ArgumentParser(
        description="AUTO keep only HU>=thr; AUTO delete only foreign>=thr; else ASK. After manual KEEP -> word prompts for words unknown to ALL detectors."
    )
    p.add_argument("--version", action="version", version=VERSION)

    p.add_argument("-inputfolder", "--inputfolder", required=True)
    p.add_argument("--pattern", default="*.txt")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--subreddits", action="store_true", required=True)

    # thresholds (default ~0.8 everywhere)
    p.add_argument("--ld-hu-threshold", "--threshold", dest="ld_hu_threshold", type=float, default=0.80)
    p.add_argument("--ld-any-threshold", type=float, default=0.80)
    p.add_argument("--hunspell-threshold", type=float, default=0.55)
    p.add_argument("--hunspell-min-word-len", type=int, default=3)

    p.add_argument("--en-item-threshold", type=float, default=0.80)
    p.add_argument("--en-sent-threshold", type=float, default=0.80)

    # ask heuristics
    p.add_argument("--zero-eps", type=float, default=0.02)
    p.add_argument("--min-words", type=int, default=4)
    p.add_argument("--margin", type=float, default=0.10)

    # szotar
    p.add_argument("--szotar-file", type=str, default="szotar.txt",
                   help="If relative, resolved relative to script root (main.py folder).")
    p.add_argument("--dict-foreign-threshold", type=float, default=0.35)
    p.add_argument("--dict-hu-threshold", type=float, default=0.35)
    p.add_argument("--dict-min-word-len", type=int, default=1)

    # word-collection
    p.add_argument("--dict-collect-min-word-len", type=int, default=4,
                   help="Min word length for asking user to add unknown words.")
    p.add_argument("--word-lang-prob", type=float, default=0.80,
                   help="If langdetect(word) returns prob >= this for ANY language, we treat it as known and we do NOT ask about it.")

    # runtime
    p.add_argument("--show-deleted", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--inplace", action="store_true")
    p.add_argument("--outputfolder", default=None)
    p.add_argument("--force-heuristic", action="store_true")
    p.add_argument("--noask", action="store_true", help="Disable prompts (ambiguous => KEEP, and NO word collection prompts).")

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

    root = script_root_dir()
    raw_szotar = Path(args.szotar_file).expanduser()
    szotar_path = (root / raw_szotar).resolve() if not raw_szotar.is_absolute() else raw_szotar.resolve()
    szotar = SzotarStore(szotar_path)

    detector_info = "langdetect" if _LANGDETECT_AVAILABLE and not args.force_heuristic else "langdetect(unavailable)"
    hunspell_info = _HUNSPELL_ENGINE_NAME if _HUNSPELL_AVAILABLE else "hunspell_hu(unavailable)"
    hunspell_en_info = _HUNSPELL_EN_ENGINE_NAME if _HUNSPELL_EN_AVAILABLE else "hunspell_en(unavailable)"

    print(f"Version: {VERSION}")
    print(f"Detectors: {detector_info} -> {hunspell_info} | EN: {hunspell_en_info}")
    print(f"Szotar (script root): {szotar_path} | exists={szotar_path.exists()} | HU={len(szotar.hu)} | IDEGEN={len(szotar.foreign)}")
    print(f"Visited: {visited_path} | entries={len(visited)}")
    print(
        f"THR: ld_hu={args.ld_hu_threshold:.2f} | ld_any={args.ld_any_threshold:.2f} | "
        f"en_item={args.en_item_threshold:.2f} | en_sent={args.en_sent_threshold:.2f} | "
        f"dict_foreign={args.dict_foreign_threshold:.2f} | dict_hu={args.dict_hu_threshold:.2f} | "
        f"word_lang_prob={args.word_lang_prob:.2f}"
    )

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

            total, deleted = process_file_subreddits(
                in_path=f,
                out_path=out_path,
                ld_hu_threshold=args.ld_hu_threshold,
                ld_any_threshold=args.ld_any_threshold,
                hunspell_threshold=args.hunspell_threshold,
                hunspell_min_word_len=args.hunspell_min_word_len,
                en_item_threshold=args.en_item_threshold,
                dict_foreign_threshold=args.dict_foreign_threshold,
                dict_hu_threshold=args.dict_hu_threshold,
                dict_min_word_len=args.dict_min_word_len,
                zero_eps=args.zero_eps,
                min_words=args.min_words,
                en_sentence_threshold=args.en_sent_threshold,
                margin=args.margin,
                dict_collect_min_word_len=args.dict_collect_min_word_len,
                word_lang_prob=args.word_lang_prob,
                show_deleted=args.show_deleted,
                noask=args.noask,
                dry_run=args.dry_run,
                force_heuristic=args.force_heuristic,
                szotar=szotar,
            )

            grand_total += total
            grand_deleted += deleted

            if not args.dry_run:
                append_visited(visited_path, rel_posix)
                visited.add(rel_posix)
                szotar.save_if_dirty()

        except Exception as e:
            print(f"[ERROR] Failed processing {rel_posix}: {e}")

    print("\n--- Summary ---")
    print(f"Items total={grand_total}, deleted={grand_deleted}, kept={grand_total - grand_deleted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
