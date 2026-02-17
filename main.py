from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Set

VERSION = "2026-02-17_v25 (unified word prompt: HU? yes->HU, no->IDEGEN; emoji+links+punct-noise delete)"

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
# Text cleanup/tokenize basics
# ============================
URL_HTTP_RE = re.compile(r"https?://\S+", flags=re.IGNORECASE)
URL_WWW_RE = re.compile(r"\bwww\.\S+", flags=re.IGNORECASE)
URL_RE = re.compile(r"(https?://\S+|\bwww\.\S+)", flags=re.IGNORECASE)

MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
INLINE_CODE_RE = re.compile(r"`[^`]*`")

WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
SINGLE_TOKEN_RE = re.compile(r"[0-9A-Za-z\u00C0-\u024F\u1E00-\u1EFF]+", flags=re.UNICODE)

CJK_RE = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF\u3040-\u30FF\uAC00-\uD7AF]")


# ============================
# Emoji / emoticon removal (FULL)
# ============================
_REGEXMOD_AVAILABLE = False
try:
    import regex as _regex  # type: ignore

    _REGEXMOD_AVAILABLE = True
except Exception:
    _regex = None  # type: ignore

_ASCII_EMOTICON_RE = re.compile(
    r"(?i)(:\-\)|:\)|:\-\(|:\(|;\-\)|;\)|:d|:\-d|:p|:\-p|:\/|:\\|<3)",
    flags=re.UNICODE,
)
_KEYCAP_RE = re.compile(r"[0-9#*]\uFE0F?\u20E3", flags=re.UNICODE)
_EMOJI_JOINERS_RE = re.compile(r"[\u200D\uFE0E\uFE0F]", flags=re.UNICODE)

_EMOJI_FALLBACK_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001FAFF"
    "\U0001F000-\U0001F02F"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "\U0001F3FB-\U0001F3FF"
    "]",
    flags=re.UNICODE,
)

if _REGEXMOD_AVAILABLE:
    _EMOJI_PROP_RE = _regex.compile(
        r"(?:\p{Extended_Pictographic}|\p{Emoji_Presentation}|\p{Emoji}\uFE0F)"
        r"(?:\u200D(?:\p{Extended_Pictographic}|\p{Emoji_Presentation}|\p{Emoji}\uFE0F))*"
        r"(?:\p{Emoji_Modifier})?",
        flags=_regex.UNICODE,
    )
    _EMOJI_MOD_RE = _regex.compile(r"\p{Emoji_Modifier}", flags=_regex.UNICODE)
else:
    _EMOJI_PROP_RE = None  # type: ignore
    _EMOJI_MOD_RE = None  # type: ignore


def contains_emoji_or_emoticon(text: str) -> bool:
    if not text:
        return False
    if _ASCII_EMOTICON_RE.search(text):
        return True
    if _KEYCAP_RE.search(text):
        return True
    if _REGEXMOD_AVAILABLE and _EMOJI_PROP_RE is not None:
        return bool(_EMOJI_PROP_RE.search(text))
    return bool(_EMOJI_FALLBACK_RE.search(text))


def _remove_emojis_core(text: str) -> str:
    t = text
    t = _ASCII_EMOTICON_RE.sub(" ", t)
    t = _KEYCAP_RE.sub(" ", t)

    if _REGEXMOD_AVAILABLE and _EMOJI_PROP_RE is not None:
        t = _EMOJI_PROP_RE.sub(" ", t)
        if _EMOJI_MOD_RE is not None:
            t = _EMOJI_MOD_RE.sub(" ", t)
    else:
        t = _EMOJI_FALLBACK_RE.sub(" ", t)

    t = _EMOJI_JOINERS_RE.sub(" ", t)
    return t


def strip_emojis_emoticons_for_detection(text: str) -> str:
    t = _remove_emojis_core(text)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def strip_emojis_emoticons_for_output(text: str) -> str:
    t = _remove_emojis_core(text)
    t = re.sub(r"[ \t]+", " ", t).strip()
    return t


def is_emoji_only_or_leftover_punct(text: str) -> bool:
    if not contains_emoji_or_emoticon(text):
        return False

    t = text
    t = MD_LINK_RE.sub(" ", t)
    t = URL_RE.sub(" ", t)
    t = strip_emojis_emoticons_for_output(t)

    t2 = re.sub(r"[\s\.\,\!\?\:\;\-\_\(\)\[\]\{\}\"\'`]+", "", t)
    return (t2 == "")


# ============================
# Punctuation / symbol-only noise ( ... +++ !!! ::: etc )
# ============================
_PUNCT_SYMBOL_ONLY_RE = re.compile(
    r"^[\s"
    r"\.\u2026"
    r",;:!?\u00A1\u00BF"
    r"\-\u2010\u2011\u2012\u2013\u2014\u2015"
    r"_"
    r"\(\)\[\]\{\}"
    r"\"\'`"
    r"~@#\$%\^&\*\+=\|\\/<>"
    r"]+$",
    flags=re.UNICODE,
)


def is_punct_only_noise(text: str) -> bool:
    if not text or not text.strip():
        return False

    t = text
    t = MD_LINK_RE.sub(" ", t)
    t = URL_RE.sub(" ", t)
    t = strip_emojis_emoticons_for_output(t).strip()

    if not t:
        return False

    return bool(_PUNCT_SYMBOL_ONLY_RE.match(t))


# ============================
# Other helpers
# ============================
def clean_for_lang(text: str) -> str:
    text = strip_emojis_emoticons_for_detection(text)
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


def is_link_dominant(text: str, max_leftover_words: int = 2) -> bool:
    t = text.strip()
    if not t:
        return False

    has_link = bool(URL_HTTP_RE.search(t) or URL_WWW_RE.search(t) or MD_LINK_RE.search(t))
    if not has_link:
        return False

    t2 = MD_LINK_RE.sub(" ", t)
    t2 = URL_RE.sub(" ", t2)
    t2 = strip_emojis_emoticons_for_detection(t2)
    t2 = re.sub(r"[\s\[\]\(\)<>.,;:!?'\"`*_=-]+", " ", t2).strip()
    if not t2:
        return True

    leftover_words = tokenize_words(t2)
    return len(leftover_words) <= max_leftover_words


def single_word_token(text: str) -> Optional[str]:
    t = text.strip()
    if not t:
        return None
    t2 = MD_LINK_RE.sub(" ", t)
    t2 = URL_RE.sub(" ", t2)
    t2 = strip_emojis_emoticons_for_detection(t2)
    t2 = re.sub(r"\s+", " ", t2).strip()
    if not t2:
        return None
    if " " in t2:
        return None
    m = SINGLE_TOKEN_RE.fullmatch(t2)
    if not m:
        return None
    return m.group(0).lower()


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
# szotar.txt (HU/IDEGEN)
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
        if not self.path.exists():
            save_szotar(self.path, self.hu, self.foreign)

    def save_if_dirty(self) -> None:
        if self._dirty:
            save_szotar(self.path, self.hu, self.foreign)
            self._dirty = False

    def contains_hu(self, w: str) -> bool:
        return w.lower() in self.hu

    def contains_foreign(self, w: str) -> bool:
        return w.lower() in self.foreign

    def add_hu(self, w: str) -> None:
        w = w.lower()
        if w in self.foreign:
            self.foreign.discard(w)
            self._dirty = True
        if w not in self.hu:
            self.hu.add(w)
            self._dirty = True

    def add_foreign(self, w: str) -> None:
        w = w.lower()
        if w in self.hu:
            self.hu.discard(w)
            self._dirty = True
        if w not in self.foreign:
            self.foreign.add(w)
            self._dirty = True


def ratio_in_set(words: List[str], s: Set[str], min_len: int) -> float:
    eligible = [w for w in words if len(w) >= min_len]
    if not eligible:
        return 0.0
    hit = sum(1 for w in eligible if w in s)
    return hit / len(eligible)


def apply_single_word_manual_to_szotar(
    text: str,
    *,
    szotar: SzotarStore,
    user_kept: bool,
) -> bool:
    w = single_word_token(text)
    if w is None:
        return False

    if user_kept:
        szotar.add_hu(w)
        szotar.save_if_dirty()
        print(f"[DICT AUTO] single-word manual KEEP -> [HU]: {w}")
    else:
        szotar.add_foreign(w)
        szotar.save_if_dirty()
        print(f"[DICT AUTO] single-word manual DELETE -> [IDEGEN]: {w}")
    return True


def collect_words_from_text_to_szotar(
    text: str,
    *,
    szotar: SzotarStore,
    min_word_len: int,
    noask: bool,
) -> None:
    """
    UNIFIED PROMPT:
      "HU listába?"  Enter/y -> HU, n -> IDEGEN
    EN hunspell csak HINT a kérdésben, de NEM dönt automatikusan.
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

    for w in ordered:
        if szotar.contains_hu(w) or szotar.contains_foreign(w):
            continue

        # HU hunspell ismeri -> ne szótárazzuk feleslegesen
        if _HUNSPELL_AVAILABLE and hunspell_hu_lookup(w):
            continue

        hint = ""
        if _HUNSPELL_EN_AVAILABLE and hunspell_en_lookup(w):
            hint = " (EN hunspell: angol?)"

        ans = input(f"Új szó: '{w}'{hint} | HU listába? (Enter/y=igen, n=IDEGEN) [Y/n]: ").strip().lower()
        if ans in {"", "y"}:
            szotar.add_hu(w)
            changed = True
        elif ans == "n":
            szotar.add_foreign(w)
            changed = True
        else:
            # ha valami mást ír be: idegennek vesszük
            szotar.add_foreign(w)
            changed = True

    if changed:
        szotar.save_if_dirty()


# ============================
# Combined scoring (exclusive allocation)
# ============================
def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def combined_hu_foreign_scores(
    *,
    dict_hu_ratio: float,
    dict_foreign_ratio: float,
    hs_hu: float,
    ld_hu: float,
    top_lang: str,
    top_prob: float,
    en_prob: float,
) -> Tuple[float, float, float, float]:
    top_hu = top_prob if top_lang == "hu" else 0.0
    top_foreign = top_prob if (top_lang not in {"", "hu"}) else 0.0

    hu_raw = dict_hu_ratio + hs_hu + ld_hu + top_hu
    foreign_raw = dict_foreign_ratio + en_prob + top_foreign

    if hu_raw >= foreign_raw:
        hu_score = clamp01(hu_raw)
        foreign_score = clamp01(min(1.0 - hu_score, foreign_raw))
    else:
        foreign_score = clamp01(foreign_raw)
        hu_score = clamp01(min(1.0 - foreign_score, hu_raw))

    return hu_score, foreign_score, hu_raw, foreign_raw


# ============================
# ITEM-level decision
# ============================
@dataclass
class Decision:
    delete: bool
    reason: str

    ld_hu: float
    top_lang: str
    top_prob: float
    en_prob: float
    hs_hu: float
    dict_hu_ratio: float
    dict_foreign_ratio: float

    hu_score: float
    foreign_score: float
    hu_raw: float
    foreign_raw: float

    preview: str
    kind: str


def decide_item(
    text: str,
    *,
    kind: str,
    force_heuristic: bool,
    main_threshold: float,
    ld_any_threshold: float,
    hunspell_min_word_len: int,
    en_item_threshold: float,
    szotar: SzotarStore,
    dict_min_word_len: int,
    zero_eps: float,
    min_words: int,
) -> Decision:
    prev = make_preview(text)

    # auto delete: emoji-only or emoji+punct leftover
    if is_emoji_only_or_leftover_punct(text):
        return Decision(
            delete=True,
            reason="emoji_only_delete",
            ld_hu=0.0,
            top_lang="",
            top_prob=0.0,
            en_prob=0.0,
            hs_hu=0.0,
            dict_hu_ratio=0.0,
            dict_foreign_ratio=0.0,
            hu_score=0.0,
            foreign_score=1.0,
            hu_raw=0.0,
            foreign_raw=1.0,
            preview=prev,
            kind=kind,
        )

    # auto delete: punctuation/symbol-only noise: ... +++ !!! ::: etc
    if is_punct_only_noise(text):
        return Decision(
            delete=True,
            reason="punct_only_delete",
            ld_hu=0.0,
            top_lang="",
            top_prob=0.0,
            en_prob=0.0,
            hs_hu=0.0,
            dict_hu_ratio=0.0,
            dict_foreign_ratio=0.0,
            hu_score=0.0,
            foreign_score=1.0,
            hu_raw=0.0,
            foreign_raw=1.0,
            preview=prev,
            kind=kind,
        )

    # auto delete: link-dominant
    if is_link_dominant(text):
        return Decision(
            delete=True,
            reason="link_dominant_delete",
            ld_hu=0.0,
            top_lang="",
            top_prob=0.0,
            en_prob=0.0,
            hs_hu=0.0,
            dict_hu_ratio=0.0,
            dict_foreign_ratio=0.0,
            hu_score=0.0,
            foreign_score=1.0,
            hu_raw=0.0,
            foreign_raw=1.0,
            preview=prev,
            kind=kind,
        )

    # auto delete: CJK
    if has_cjk(text):
        return Decision(
            delete=True,
            reason="foreign_cjk",
            ld_hu=0.0,
            top_lang="",
            top_prob=0.0,
            en_prob=0.0,
            hs_hu=0.0,
            dict_hu_ratio=0.0,
            dict_foreign_ratio=1.0,
            hu_score=0.0,
            foreign_score=1.0,
            hu_raw=0.0,
            foreign_raw=1.0,
            preview=prev,
            kind=kind,
        )

    # single-word => szotar first
    sw = single_word_token(text)
    if sw is not None:
        if szotar.contains_foreign(sw):
            return Decision(
                delete=True,
                reason="single_word_szotar_foreign_delete",
                ld_hu=0.0,
                top_lang="",
                top_prob=0.0,
                en_prob=0.0,
                hs_hu=0.0,
                dict_hu_ratio=0.0,
                dict_foreign_ratio=1.0,
                hu_score=0.0,
                foreign_score=1.0,
                hu_raw=0.0,
                foreign_raw=1.0,
                preview=prev,
                kind=kind,
            )
        if szotar.contains_hu(sw):
            return Decision(
                delete=False,
                reason="single_word_szotar_hu_keep",
                ld_hu=0.0,
                top_lang="",
                top_prob=0.0,
                en_prob=0.0,
                hs_hu=0.0,
                dict_hu_ratio=1.0,
                dict_foreign_ratio=0.0,
                hu_score=1.0,
                foreign_score=0.0,
                hu_raw=1.0,
                foreign_raw=0.0,
                preview=prev,
                kind=kind,
            )

    cleaned = clean_for_lang(text)
    words = tokenize_words(cleaned)
    if not words:
        return Decision(
            delete=False,
            reason="ask_zero",
            ld_hu=0.0,
            top_lang="",
            top_prob=0.0,
            en_prob=0.0,
            hs_hu=0.0,
            dict_hu_ratio=0.0,
            dict_foreign_ratio=0.0,
            hu_score=0.0,
            foreign_score=0.0,
            hu_raw=0.0,
            foreign_raw=0.0,
            preview=prev,
            kind=kind,
        )

    dict_hu_ratio = ratio_in_set(words, szotar.hu, dict_min_word_len)
    dict_foreign_ratio = ratio_in_set(words, szotar.foreign, dict_min_word_len)

    ld_hu = 0.0
    top_lang = ""
    top_prob = 0.0
    en_prob = 0.0

    if _LANGDETECT_AVAILABLE and (not force_heuristic) and detect_langs is not None:
        r_hu = _lang_prob_ratio_langdetect(cleaned, "hu")
        ld_hu = float(r_hu) if r_hu is not None else 0.0

        r_en = _lang_prob_ratio_langdetect(cleaned, "en")
        en_prob = float(r_en) if r_en is not None else 0.0

        top = detect_top_lang(cleaned)
        if top is not None:
            top_lang, top_prob = top

    hs = hunspell_hu_ratio(cleaned, min_word_len=hunspell_min_word_len)
    hs_hu = float(hs) if hs is not None else 0.0

    hu_score, foreign_score, hu_raw, foreign_raw = combined_hu_foreign_scores(
        dict_hu_ratio=dict_hu_ratio,
        dict_foreign_ratio=dict_foreign_ratio,
        hs_hu=hs_hu,
        ld_hu=ld_hu,
        top_lang=top_lang,
        top_prob=top_prob,
        en_prob=en_prob,
    )

    if hu_score >= main_threshold:
        return Decision(
            delete=False,
            reason="hu_combined_keep",
            ld_hu=ld_hu,
            top_lang=top_lang,
            top_prob=top_prob,
            en_prob=en_prob,
            hs_hu=hs_hu,
            dict_hu_ratio=dict_hu_ratio,
            dict_foreign_ratio=dict_foreign_ratio,
            hu_score=hu_score,
            foreign_score=foreign_score,
            hu_raw=hu_raw,
            foreign_raw=foreign_raw,
            preview=prev,
            kind=kind,
        )

    if foreign_score >= main_threshold:
        return Decision(
            delete=True,
            reason="foreign_combined_delete",
            ld_hu=ld_hu,
            top_lang=top_lang,
            top_prob=top_prob,
            en_prob=en_prob,
            hs_hu=hs_hu,
            dict_hu_ratio=dict_hu_ratio,
            dict_foreign_ratio=dict_foreign_ratio,
            hu_score=hu_score,
            foreign_score=foreign_score,
            hu_raw=hu_raw,
            foreign_raw=foreign_raw,
            preview=prev,
            kind=kind,
        )

    eligible_words = [w for w in words if len(w) >= hunspell_min_word_len]
    too_short = len(eligible_words) < min_words
    near_zero = (
        (ld_hu <= zero_eps)
        and (hs_hu <= zero_eps)
        and (en_prob <= zero_eps)
        and (dict_hu_ratio <= zero_eps)
        and (dict_foreign_ratio <= zero_eps)
        and (top_prob <= zero_eps)
    )
    reason = "ask_zero" if (too_short or near_zero) else "ask"

    return Decision(
        delete=False,
        reason=reason,
        ld_hu=ld_hu,
        top_lang=top_lang,
        top_prob=top_prob,
        en_prob=en_prob,
        hs_hu=hs_hu,
        dict_hu_ratio=dict_hu_ratio,
        dict_foreign_ratio=dict_foreign_ratio,
        hu_score=hu_score,
        foreign_score=foreign_score,
        hu_raw=hu_raw,
        foreign_raw=foreign_raw,
        preview=prev,
        kind=kind,
    )


def handle_item_prompt(dec: Decision, noask: bool) -> Tuple[bool, bool]:
    """
    returns (delete?, user_kept?)
    y = keep, else delete
    if --noask: keep ambiguous
    """
    if noask:
        return False, True

    print("\n[ASK] Nem egyértelmű (HU_score<thr és FOREIGN_score<thr) -> kézi döntés.")
    print(
        f"Kind: {dec.kind} | reason={dec.reason} | "
        f"HU_score={dec.hu_score:.3f} (raw={dec.hu_raw:.3f}) | "
        f"FOREIGN_score={dec.foreign_score:.3f} (raw={dec.foreign_raw:.3f}) | "
        f"ld_hu={dec.ld_hu:.6f} | top={dec.top_lang}:{dec.top_prob:.3f} | en_prob={dec.en_prob:.6f} | "
        f"hs_hu={dec.hs_hu:.6f} | dict_hu={dec.dict_hu_ratio:.3f} | dict_foreign={dec.dict_foreign_ratio:.3f}"
    )
    print(f"Preview: {dec.preview}")

    ans = input("Döntés? (y=MEGTART, n/Enter=TÖRÖL) [y/N]: ").strip().lower()
    user_kept = (ans == "y")
    return (not user_kept), user_kept


# ============================
# Sentence-level decision
# ============================
@dataclass
class SentDecision:
    keep: bool
    reason: str

    ld_hu: float
    top_lang: str
    top_prob: float
    en_prob: float
    hs_hu: float
    dict_hu_ratio: float
    dict_foreign_ratio: float

    hu_score: float
    foreign_score: float
    hu_raw: float
    foreign_raw: float

    preview: str


def decide_sentence(
    sentence: str,
    *,
    main_threshold: float,
    hunspell_min_word_len: int,
    dict_min_word_len: int,
    szotar: SzotarStore,
    force_heuristic: bool,
) -> SentDecision:
    prev = make_preview(sentence)

    if is_emoji_only_or_leftover_punct(sentence):
        return SentDecision(
            keep=False,
            reason="emoji_only_delete",
            ld_hu=0.0,
            top_lang="",
            top_prob=0.0,
            en_prob=0.0,
            hs_hu=0.0,
            dict_hu_ratio=0.0,
            dict_foreign_ratio=0.0,
            hu_score=0.0,
            foreign_score=1.0,
            hu_raw=0.0,
            foreign_raw=1.0,
            preview=prev,
        )

    if is_punct_only_noise(sentence):
        return SentDecision(
            keep=False,
            reason="punct_only_delete",
            ld_hu=0.0,
            top_lang="",
            top_prob=0.0,
            en_prob=0.0,
            hs_hu=0.0,
            dict_hu_ratio=0.0,
            dict_foreign_ratio=0.0,
            hu_score=0.0,
            foreign_score=1.0,
            hu_raw=0.0,
            foreign_raw=1.0,
            preview=prev,
        )

    if is_link_dominant(sentence):
        return SentDecision(
            keep=False,
            reason="link_dominant_delete",
            ld_hu=0.0,
            top_lang="",
            top_prob=0.0,
            en_prob=0.0,
            hs_hu=0.0,
            dict_hu_ratio=0.0,
            dict_foreign_ratio=0.0,
            hu_score=0.0,
            foreign_score=1.0,
            hu_raw=0.0,
            foreign_raw=1.0,
            preview=prev,
        )

    cleaned = clean_for_lang(sentence)
    if not cleaned:
        return SentDecision(
            keep=True,
            reason="empty",
            ld_hu=0.0,
            top_lang="",
            top_prob=0.0,
            en_prob=0.0,
            hs_hu=0.0,
            dict_hu_ratio=0.0,
            dict_foreign_ratio=0.0,
            hu_score=0.0,
            foreign_score=0.0,
            hu_raw=0.0,
            foreign_raw=0.0,
            preview=prev,
        )

    if has_cjk(cleaned):
        return SentDecision(
            keep=False,
            reason="foreign_cjk",
            ld_hu=0.0,
            top_lang="",
            top_prob=0.0,
            en_prob=0.0,
            hs_hu=0.0,
            dict_hu_ratio=0.0,
            dict_foreign_ratio=1.0,
            hu_score=0.0,
            foreign_score=1.0,
            hu_raw=0.0,
            foreign_raw=1.0,
            preview=prev,
        )

    # single-word => szotar first
    sw = single_word_token(sentence)
    if sw is not None:
        if szotar.contains_foreign(sw):
            return SentDecision(
                keep=False,
                reason="single_word_szotar_foreign_delete",
                ld_hu=0.0,
                top_lang="",
                top_prob=0.0,
                en_prob=0.0,
                hs_hu=0.0,
                dict_hu_ratio=0.0,
                dict_foreign_ratio=1.0,
                hu_score=0.0,
                foreign_score=1.0,
                hu_raw=0.0,
                foreign_raw=1.0,
                preview=prev,
            )
        if szotar.contains_hu(sw):
            return SentDecision(
                keep=True,
                reason="single_word_szotar_hu_keep",
                ld_hu=0.0,
                top_lang="",
                top_prob=0.0,
                en_prob=0.0,
                hs_hu=0.0,
                dict_hu_ratio=1.0,
                dict_foreign_ratio=0.0,
                hu_score=1.0,
                foreign_score=0.0,
                hu_raw=1.0,
                foreign_raw=0.0,
                preview=prev,
            )

    words = tokenize_words(cleaned)
    dict_hu_ratio = ratio_in_set(words, szotar.hu, dict_min_word_len)
    dict_foreign_ratio = ratio_in_set(words, szotar.foreign, dict_min_word_len)

    ld_hu = 0.0
    top_lang = ""
    top_prob = 0.0
    en_prob = 0.0

    if _LANGDETECT_AVAILABLE and (not force_heuristic) and detect_langs is not None:
        r_hu = _lang_prob_ratio_langdetect(cleaned, "hu")
        ld_hu = float(r_hu) if r_hu is not None else 0.0

        r_en = _lang_prob_ratio_langdetect(cleaned, "en")
        en_prob = float(r_en) if r_en is not None else 0.0

        top = detect_top_lang(cleaned)
        if top is not None:
            top_lang, top_prob = top

    hs = hunspell_hu_ratio(cleaned, min_word_len=hunspell_min_word_len)
    hs_hu = float(hs) if hs is not None else 0.0

    hu_score, foreign_score, hu_raw, foreign_raw = combined_hu_foreign_scores(
        dict_hu_ratio=dict_hu_ratio,
        dict_foreign_ratio=dict_foreign_ratio,
        hs_hu=hs_hu,
        ld_hu=ld_hu,
        top_lang=top_lang,
        top_prob=top_prob,
        en_prob=en_prob,
    )

    if foreign_score >= main_threshold:
        return SentDecision(
            keep=False,
            reason="foreign_combined_delete",
            ld_hu=ld_hu,
            top_lang=top_lang,
            top_prob=top_prob,
            en_prob=en_prob,
            hs_hu=hs_hu,
            dict_hu_ratio=dict_hu_ratio,
            dict_foreign_ratio=dict_foreign_ratio,
            hu_score=hu_score,
            foreign_score=foreign_score,
            hu_raw=hu_raw,
            foreign_raw=foreign_raw,
            preview=prev,
        )

    return SentDecision(
        keep=True,
        reason="keep",
        ld_hu=ld_hu,
        top_lang=top_lang,
        top_prob=top_prob,
        en_prob=en_prob,
        hs_hu=hs_hu,
        dict_hu_ratio=dict_hu_ratio,
        dict_foreign_ratio=dict_foreign_ratio,
        hu_score=hu_score,
        foreign_score=foreign_score,
        hu_raw=hu_raw,
        foreign_raw=foreign_raw,
        preview=prev,
    )


def filter_text_sentences(
    text: str,
    *,
    context_kind: str,
    file_name: str,
    show_deleted: bool,
    noask: bool,
    main_threshold: float,
    hunspell_min_word_len: int,
    dict_min_word_len: int,
    dict_collect_min_word_len: int,
    szotar: SzotarStore,
    force_heuristic: bool,
) -> str:
    chunks = split_into_chunks(text)
    if not chunks:
        return strip_emojis_emoticons_for_output(text)

    kept: List[str] = []
    for ch in chunks:
        sd = decide_sentence(
            ch,
            main_threshold=main_threshold,
            hunspell_min_word_len=hunspell_min_word_len,
            dict_min_word_len=dict_min_word_len,
            szotar=szotar,
            force_heuristic=force_heuristic,
        )
        if not sd.keep:
            if show_deleted:
                print(
                    f"[DELETED_SENT] {file_name} | {context_kind} | reason={sd.reason} | "
                    f"FOREIGN_score={sd.foreign_score:.3f} (raw={sd.foreign_raw:.3f}) | "
                    f"HU_score={sd.hu_score:.3f} (raw={sd.hu_raw:.3f}) | top={sd.top_lang}:{sd.top_prob:.3f} | "
                    f"dict_foreign={sd.dict_foreign_ratio:.3f} dict_hu={sd.dict_hu_ratio:.3f} | {sd.preview}"
                )
            continue
        kept.append(strip_emojis_emoticons_for_output(ch))

    return "\n".join([k for k in kept if k.strip()]).strip()


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
    main_threshold: float,
    hunspell_min_word_len: int,
    dict_min_word_len: int,
    dict_collect_min_word_len: int,
    szotar: SzotarStore,
    force_heuristic: bool,
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
                main_threshold=main_threshold,
                hunspell_min_word_len=hunspell_min_word_len,
                dict_min_word_len=dict_min_word_len,
                dict_collect_min_word_len=dict_collect_min_word_len,
                szotar=szotar,
                force_heuristic=force_heuristic,
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
                main_threshold=main_threshold,
                hunspell_min_word_len=hunspell_min_word_len,
                dict_min_word_len=dict_min_word_len,
                dict_collect_min_word_len=dict_collect_min_word_len,
                szotar=szotar,
                force_heuristic=force_heuristic,
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
    main_threshold: float,
    hunspell_min_word_len: int,
    dict_min_word_len: int,
    dict_collect_min_word_len: int,
    szotar: SzotarStore,
    force_heuristic: bool,
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
        main_threshold=main_threshold,
        hunspell_min_word_len=hunspell_min_word_len,
        dict_min_word_len=dict_min_word_len,
        dict_collect_min_word_len=dict_collect_min_word_len,
        szotar=szotar,
        force_heuristic=force_heuristic,
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
    main_threshold: float,
    ld_any_threshold: float,
    hunspell_min_word_len: int,
    en_item_threshold: float,
    dict_min_word_len: int,
    zero_eps: float,
    min_words: int,
    dict_collect_min_word_len: int,
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

        # POST
        total += 1
        post_text = extract_subreddit_post_text(pre)
        post_dec = decide_item(
            post_text,
            kind="Post",
            force_heuristic=force_heuristic,
            main_threshold=main_threshold,
            ld_any_threshold=ld_any_threshold,
            hunspell_min_word_len=hunspell_min_word_len,
            en_item_threshold=en_item_threshold,
            szotar=szotar,
            dict_min_word_len=dict_min_word_len,
            zero_eps=zero_eps,
            min_words=min_words,
        )

        manual_post = False
        delete_post = post_dec.delete
        user_kept_post = False

        if post_dec.reason in {"ask", "ask_zero"}:
            manual_post = True
            delete_post, user_kept_post = handle_item_prompt(post_dec, noask=noask)
            apply_single_word_manual_to_szotar(post_text, szotar=szotar, user_kept=user_kept_post)

        if show_deleted:
            tag = "DELETED" if delete_post else "KEPT"
            print(
                f"[{tag}] {in_path.name} | Post | reason={post_dec.reason} | "
                f"HU_score={post_dec.hu_score:.3f} (raw={post_dec.hu_raw:.3f}) | "
                f"FOREIGN_score={post_dec.foreign_score:.3f} (raw={post_dec.foreign_raw:.3f}) | "
                f"ld_hu={post_dec.ld_hu:.6f} | top={post_dec.top_lang}:{post_dec.top_prob:.3f} | en_prob={post_dec.en_prob:.6f} | "
                f"hs_hu={post_dec.hs_hu:.6f} | dict_hu={post_dec.dict_hu_ratio:.3f} | dict_foreign={post_dec.dict_foreign_ratio:.3f} | "
                f"{post_dec.preview}"
            )

        if delete_post:
            deleted += 1
            continue

        new_pre = rewrite_post_pre_block(
            pre,
            file_name=in_path.name,
            show_deleted=show_deleted,
            noask=noask,
            main_threshold=main_threshold,
            hunspell_min_word_len=hunspell_min_word_len,
            dict_min_word_len=dict_min_word_len,
            dict_collect_min_word_len=dict_collect_min_word_len,
            szotar=szotar,
            force_heuristic=force_heuristic,
        )

        if not extract_subreddit_post_text(new_pre).strip():
            if show_deleted:
                print(f"[DELETED] {in_path.name} | Post | reason=post_became_empty_after_sentence_filter")
            deleted += 1
            continue

        if manual_post and user_kept_post and single_word_token(post_text) is None:
            collect_words_from_text_to_szotar(
                post_text,
                szotar=szotar,
                min_word_len=dict_collect_min_word_len,
                noask=noask,
            )

        kept.append(new_pre)

        # COMMENTS
        for c in comments:
            total += 1
            c_text = extract_subreddit_comment_text(c)
            c_dec = decide_item(
                c_text,
                kind="Comment",
                force_heuristic=force_heuristic,
                main_threshold=main_threshold,
                ld_any_threshold=ld_any_threshold,
                hunspell_min_word_len=hunspell_min_word_len,
                en_item_threshold=en_item_threshold,
                szotar=szotar,
                dict_min_word_len=dict_min_word_len,
                zero_eps=zero_eps,
                min_words=min_words,
            )

            manual_comment = False
            delete_c = c_dec.delete
            user_kept_comment = False

            if c_dec.reason in {"ask", "ask_zero"}:
                manual_comment = True
                delete_c, user_kept_comment = handle_item_prompt(c_dec, noask=noask)
                apply_single_word_manual_to_szotar(c_text, szotar=szotar, user_kept=user_kept_comment)

            if show_deleted:
                tag = "DELETED" if delete_c else "KEPT"
                print(
                    f"[{tag}] {in_path.name} | Comment | reason={c_dec.reason} | "
                    f"HU_score={c_dec.hu_score:.3f} (raw={c_dec.hu_raw:.3f}) | "
                    f"FOREIGN_score={c_dec.foreign_score:.3f} (raw={c_dec.foreign_raw:.3f}) | "
                    f"ld_hu={c_dec.ld_hu:.6f} | top={c_dec.top_lang}:{c_dec.top_prob:.3f} | en_prob={c_dec.en_prob:.6f} | "
                    f"hs_hu={c_dec.hs_hu:.6f} | dict_hu={c_dec.dict_hu_ratio:.3f} | dict_foreign={c_dec.dict_foreign_ratio:.3f} | "
                    f"{c_dec.preview}"
                )

            if delete_c:
                deleted += 1
                continue

            new_c = rewrite_comment_block(
                c,
                file_name=in_path.name,
                show_deleted=show_deleted,
                noask=noask,
                main_threshold=main_threshold,
                hunspell_min_word_len=hunspell_min_word_len,
                dict_min_word_len=dict_min_word_len,
                dict_collect_min_word_len=dict_collect_min_word_len,
                szotar=szotar,
                force_heuristic=force_heuristic,
            )

            if new_c is None:
                if show_deleted:
                    print(f"[DELETED] {in_path.name} | Comment | reason=comment_became_empty_after_sentence_filter")
                deleted += 1
                continue

            if manual_comment and user_kept_comment and single_word_token(c_text) is None:
                collect_words_from_text_to_szotar(
                    c_text,
                    szotar=szotar,
                    min_word_len=dict_collect_min_word_len,
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
        description="HU vs FOREIGN combined scoring; auto delete emoji-only, punct/symbol-only, and link-dominant; szotar in script root."
    )
    p.add_argument("--version", action="version", version=VERSION)

    p.add_argument("-inputfolder", "--inputfolder", required=True)
    p.add_argument("--pattern", default="*.txt")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--subreddits", action="store_true")

    p.add_argument("--threshold", type=float, default=0.80, help="Main HU/FOREIGN combined score threshold.")
    p.add_argument("--ld-any-threshold", type=float, default=0.80)
    p.add_argument("--hunspell-min-word-len", type=int, default=3)
    p.add_argument("--en-item-threshold", type=float, default=0.80)

    p.add_argument("--zero-eps", type=float, default=0.02)
    p.add_argument("--min-words", type=int, default=4)

    p.add_argument(
        "--szotar-file",
        type=str,
        default="szotar.txt",
        help="If relative, resolved relative to script root (main.py folder).",
    )
    p.add_argument("--dict-min-word-len", type=int, default=1)
    p.add_argument("--dict-collect-min-word-len", type=int, default=4)

    p.add_argument("--show-deleted", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--inplace", action="store_true")
    p.add_argument("--outputfolder", default=None)
    p.add_argument("--force-heuristic", action="store_true")
    p.add_argument("--noask", action="store_true", help="Disable prompts (ambiguous => KEEP, and NO word collection).")

    args = p.parse_args()

    if not args.subreddits:
        print("ERROR: This script currently supports only --subreddits mode.")
        return 2

    in_dir = Path(args.inputfolder).expanduser().resolve()
    if not in_dir.exists() or not in_dir.is_dir():
        print(f"ERROR: inputfolder does not exist or not a directory: {in_dir}")
        return 2

    out_dir = in_dir if args.inplace else (
        Path(args.outputfolder).expanduser().resolve()
        if args.outputfolder
        else (in_dir / "cleaned")
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
    emoji_info = "regex(emoji-props)" if _REGEXMOD_AVAILABLE else "fallback(emoji-ranges)"

    print(f"Version: {VERSION}")
    print(f"Detectors: {detector_info} -> {hunspell_info} | EN: {hunspell_en_info} | Emoji: {emoji_info}")
    print(f"Szotar (script root): {szotar_path} | exists={szotar_path.exists()} | HU={len(szotar.hu)} | IDEGEN={len(szotar.foreign)}")
    print(f"Visited: {visited_path} | entries={len(visited)}")
    print(f"THR: main={args.threshold:.2f} | dict_min_word_len={args.dict_min_word_len}")

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
                main_threshold=args.threshold,
                ld_any_threshold=args.ld_any_threshold,
                hunspell_min_word_len=args.hunspell_min_word_len,
                en_item_threshold=args.en_item_threshold,
                dict_min_word_len=args.dict_min_word_len,
                zero_eps=args.zero_eps,
                min_words=args.min_words,
                dict_collect_min_word_len=args.dict_collect_min_word_len,
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
