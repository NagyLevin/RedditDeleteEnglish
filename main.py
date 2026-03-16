#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlsplit

from langdetect import DetectorFactory, LangDetectException, detect_langs

try:
    import phunspell
except ImportError:
    phunspell = None

try:
    from lingua import Language, LanguageDetectorBuilder
except ImportError:
    Language = None
    LanguageDetectorBuilder = None

DetectorFactory.seed = 0

BODY_RE = re.compile(r"^\s*body:\s*$")
COMMENT_RE = re.compile(r"^\s*comment:\s*$")
BY_LINE_RE = re.compile(r"^(\s*by\s+[^:\n\r]+:)(.*?)(\r?\n)?$", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű]+(?:[-'][A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű]+)*")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+|www\.[^)\s]+)\)", re.IGNORECASE)
MENTION_RE = re.compile(r"/?(?:r/|u/)[A-Za-z0-9_\-]+", re.IGNORECASE)
NUMBER_ONLY_RE = re.compile(
    r"^\s*[+\-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*(?:[%€$£¥])?\s*$",
    re.UNICODE,
)
EMOJI_CHAR_CLASS = r"[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\u2600-\u27BF\u200d\uFE0F]"
EMOJI_ONLY_RE = re.compile(rf"^\s*(?:{EMOJI_CHAR_CLASS}+\s*)+$", re.UNICODE)
PUNCT_ONLY_RE = re.compile(r"^\s*[\W_]+\s*$", re.UNICODE)
NUMERIC_SYMBOL_ONLY_RE = re.compile(
    r"^\s*[\d\s.,:;!?¿¡+\-*/=×xX%‰€$£¥<>()\[\]{}|\\^~`'\"*_&#@]+\s*$",
    re.UNICODE,
)
MEDIA_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif")


@dataclass
class Block:
    kind: str  # raw | body | comment
    header: list[str]
    content: list[str]


@dataclass
class AnalysisResult:
    score: float
    keep: bool
    reason: str
    langdetect_hu: Optional[float]
    phunspell_ratio: Optional[float]
    lingua_hu: Optional[float]
    word_count: int
    normalized_text: str
    forced_delete: bool = False


class HungarianDetectors:
    def __init__(self) -> None:
        self.spell = None
        self.lingua_detector = None
        self.phunspell_error = None

        if phunspell is not None:
            try:
                self.spell = phunspell.Phunspell("hu_HU")
            except Exception as exc:  # pragma: no cover - environment dependent
                self.phunspell_error = f"phunspell hu_HU nem elérhető: {exc}"

        if LanguageDetectorBuilder is not None and Language is not None:
            try:
                self.lingua_detector = (
                    LanguageDetectorBuilder.from_languages(
                        Language.HUNGARIAN,
                        Language.ENGLISH,
                        Language.GERMAN,
                        Language.ROMANIAN,
                        Language.SLOVAK,
                        Language.CROATIAN,
                        Language.SERBIAN,
                        Language.POLISH,
                        Language.CZECH,
                        Language.ITALIAN,
                        Language.FRENCH,
                        Language.SPANISH,
                    ).build()
                )
            except Exception:
                self.lingua_detector = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TXT fájlok body/comment részeinek magyar nyelv szerinti szűrése."
    )
    parser.add_argument("--inputmappa", required=True, help="Bemeneti mappa")
    parser.add_argument("--out", required=True, help="Kimeneti mappa")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum összpontszám a megtartáshoz (0.0 - 1.0). Alapértelmezett: 0.5",
    )
    parser.add_argument(
        "--showkomment",
        action="store_true",
        help="A megtartott body/comment blokkokról és sor-szintű takarításról is írjon részletes logot.",
    )
    return parser.parse_args()



def read_text(path: Path) -> str:
    encodings = ("utf-8", "utf-8-sig", "cp1250", "latin-1")
    last_error = None
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError(
        last_error.encoding if last_error else "unknown",
        b"",
        0,
        1,
        f"Nem sikerült beolvasni a fájlt: {path}",
    )



def sanitize_text(text: str) -> str:
    text = MARKDOWN_LINK_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = re.sub(EMOJI_CHAR_CLASS, " ", text)
    text = re.sub(r"[_*`>#\[\](){}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



def tokenize_words(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text)]



def is_block_start(line: str) -> bool:
    return bool(BODY_RE.match(line) or COMMENT_RE.match(line))



def parse_blocks(text: str) -> list[Block]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return []

    blocks: list[Block] = [Block(kind="raw", header=[lines[0]], content=[])]
    i = 1
    n = len(lines)

    while i < n:
        line = lines[i]
        if BODY_RE.match(line):
            header = [line]
            i += 1
            content: list[str] = []
            while i < n and not is_block_start(lines[i]):
                content.append(lines[i])
                i += 1
            blocks.append(Block(kind="body", header=header, content=content))
            continue

        if COMMENT_RE.match(line):
            header = [line]
            i += 1
            content = []
            while i < n and not is_block_start(lines[i]):
                content.append(lines[i])
                i += 1
            blocks.append(Block(kind="comment", header=header, content=content))
            continue

        blocks.append(Block(kind="raw", header=[line], content=[]))
        i += 1

    return blocks



def extract_text_from_block(block: Block) -> str:
    if block.kind == "comment":
        if not block.content:
            return ""
        pieces: list[str] = []
        first = block.content[0].strip()
        if ":" in first:
            _, after = first.split(":", 1)
            if after.strip():
                pieces.append(after.strip())
        elif first:
            pieces.append(first)
        for line in block.content[1:]:
            stripped = line.strip()
            if stripped:
                pieces.append(stripped)
        return " ".join(pieces).strip()

    pieces = [line.strip() for line in block.content if line.strip()]
    return " ".join(pieces).strip()



def langdetect_hu_probability(text: str) -> Optional[float]:
    try:
        langs = detect_langs(text)
    except LangDetectException:
        return None
    for entry in langs:
        if entry.lang == "hu":
            return float(entry.prob)
    return 0.0



def phunspell_ratio(words: list[str], spell) -> Optional[float]:
    if spell is None:
        return None
    filtered = [w for w in words if len(w) >= 2]
    if not filtered:
        return None
    valid = 0
    for word in filtered:
        try:
            if spell.lookup(word):
                valid += 1
        except Exception:
            continue
    return valid / len(filtered) if filtered else None



def lingua_hu_probability(text: str, detector) -> Optional[float]:
    if detector is None or Language is None:
        return None
    try:
        return float(detector.compute_language_confidence(text, Language.HUNGARIAN))
    except Exception:
        return None



def combined_score(lang_score: Optional[float], spell_ratio: Optional[float], lingua_score: Optional[float]) -> float:
    weighted_parts: list[tuple[float, float]] = []
    if lang_score is not None:
        weighted_parts.append((0.5, lang_score))
    if spell_ratio is not None:
        weighted_parts.append((0.35, spell_ratio))
    if lingua_score is not None:
        weighted_parts.append((0.15, lingua_score))

    if not weighted_parts:
        return 0.0

    total_weight = sum(weight for weight, _ in weighted_parts)
    return sum(weight * value for weight, value in weighted_parts) / total_weight



def is_emoji_only_text(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped and EMOJI_ONLY_RE.fullmatch(stripped))



def is_punct_only_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if WORD_RE.search(stripped):
        return False
    return bool(PUNCT_ONLY_RE.fullmatch(stripped))



def is_number_only_text(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped and NUMBER_ONLY_RE.fullmatch(stripped))



def is_numeric_symbol_only_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped or not any(ch.isdigit() for ch in stripped):
        return False
    if URL_RE.search(stripped) or MARKDOWN_LINK_RE.search(stripped):
        return False
    if WORD_RE.search(stripped):
        return False
    return bool(NUMERIC_SYMBOL_ONLY_RE.fullmatch(stripped))



def url_is_direct_media(url: str) -> bool:
    candidate = url.strip()
    if candidate.lower().startswith("www."):
        candidate = "https://" + candidate
    try:
        path = urlsplit(candidate).path.lower()
    except Exception:
        return False
    return path.endswith(MEDIA_EXTENSIONS)



def normalize_markdown_links(text: str) -> str:
    return MARKDOWN_LINK_RE.sub(lambda m: f" {m.group(2)} ", text)



def is_link_only_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    expanded = normalize_markdown_links(stripped)
    urls = URL_RE.findall(expanded)
    if not urls:
        return False
    rest = URL_RE.sub(" ", expanded)
    rest = re.sub(EMOJI_CHAR_CLASS, " ", rest)
    rest = re.sub(r"[\s\-–—•·|,;:!?¿¡()\[\]{}<>\"'`*_~]+", " ", rest)
    rest = re.sub(r"\s+", " ", rest).strip()
    return rest == ""



def is_media_only_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    expanded = normalize_markdown_links(stripped)
    urls = URL_RE.findall(expanded)
    if not urls:
        return False
    rest = URL_RE.sub(" ", expanded)
    rest = re.sub(r"\s+", " ", rest).strip()
    return rest == "" and all(url_is_direct_media(url) for url in urls)



def compact_token_for_mention_check(text: str) -> str:
    compact = MARKDOWN_LINK_RE.sub(" ", text)
    compact = re.sub(EMOJI_CHAR_CLASS, "", compact)
    compact = re.sub(r"[^A-Za-z0-9_/\-]", "", compact)
    if compact.startswith("/"):
        compact = compact[1:]
    return compact



def is_mention_only_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    compact = compact_token_for_mention_check(stripped)
    return bool(re.fullmatch(r"(?:r|u)/[A-Za-z0-9_\-]+", compact, flags=re.IGNORECASE))



def classify_noise_line(text: str) -> Optional[str]:
    stripped = text.strip()
    if not stripped:
        return None
    if is_emoji_only_text(stripped):
        return "csak emoji(k)ből álló sor"
    if is_punct_only_text(stripped):
        return "csak írásjel/szimbólum sor"
    if is_numeric_symbol_only_text(stripped):
        return "csak szám+szimbólum sor"
    return None



def clean_block_lines(block: Block) -> tuple[Block, list[tuple[str, str]]]:
    if not block.content:
        return block, []

    removed: list[tuple[str, str]] = []
    new_content: list[str] = []

    for idx, original_line in enumerate(block.content):
        if block.kind == "comment" and idx == 0:
            line_no_nl = original_line.rstrip("\r\n")
            newline = original_line[len(line_no_nl):]
            if ":" in line_no_nl:
                prefix, payload = line_no_nl.split(":", 1)
                reason = classify_noise_line(payload)
                if reason is not None:
                    removed.append((payload.strip(), reason))
                    new_content.append(prefix + ":" + newline)
                    continue
            new_content.append(original_line)
            continue

        reason = classify_noise_line(original_line)
        if reason is not None:
            removed.append((original_line.strip(), reason))
            continue
        new_content.append(original_line)

    return Block(kind=block.kind, header=list(block.header), content=new_content), removed



def analyze_text(text: str, detectors: HungarianDetectors, threshold: float) -> AnalysisResult:
    stripped = text.strip()

    if not stripped:
        return AnalysisResult(
            score=0.0,
            keep=False,
            reason="azonnali törlés: a blokk kiürült vagy üres",
            langdetect_hu=None,
            phunspell_ratio=None,
            lingua_hu=None,
            word_count=0,
            normalized_text="",
            forced_delete=True,
        )

    if is_media_only_text(stripped):
        return AnalysisResult(
            score=0.0,
            keep=False,
            reason="azonnali törlés: a blokk csak közvetlen kép/gif link(ek)ből áll",
            langdetect_hu=None,
            phunspell_ratio=None,
            lingua_hu=None,
            word_count=0,
            normalized_text=stripped,
            forced_delete=True,
        )

    if is_link_only_text(stripped):
        return AnalysisResult(
            score=0.0,
            keep=False,
            reason="azonnali törlés: a blokk csak link(ek)ből áll",
            langdetect_hu=None,
            phunspell_ratio=None,
            lingua_hu=None,
            word_count=0,
            normalized_text=stripped,
            forced_delete=True,
        )

    if is_mention_only_text(stripped):
        return AnalysisResult(
            score=0.0,
            keep=False,
            reason="azonnali törlés: a blokk csak subreddit/user hivatkozásból áll",
            langdetect_hu=None,
            phunspell_ratio=None,
            lingua_hu=None,
            word_count=0,
            normalized_text=stripped,
            forced_delete=True,
        )

    if is_emoji_only_text(stripped):
        return AnalysisResult(
            score=0.0,
            keep=False,
            reason="azonnali törlés: a blokk csak emoji(k)ból áll",
            langdetect_hu=None,
            phunspell_ratio=None,
            lingua_hu=None,
            word_count=0,
            normalized_text=stripped,
            forced_delete=True,
        )

    if is_punct_only_text(stripped):
        return AnalysisResult(
            score=0.0,
            keep=False,
            reason="azonnali törlés: a blokk csak írásjelekből/szimbólumokból áll",
            langdetect_hu=None,
            phunspell_ratio=None,
            lingua_hu=None,
            word_count=0,
            normalized_text=stripped,
            forced_delete=True,
        )

    if is_number_only_text(stripped) or is_numeric_symbol_only_text(stripped):
        return AnalysisResult(
            score=0.0,
            keep=False,
            reason="azonnali törlés: a blokk csak szám/számformátum/szám+szimbólum érték",
            langdetect_hu=None,
            phunspell_ratio=None,
            lingua_hu=None,
            word_count=0,
            normalized_text=stripped,
            forced_delete=True,
        )

    cleaned = sanitize_text(stripped)
    words = tokenize_words(cleaned)

    if cleaned and not words:
        return AnalysisResult(
            score=0.0,
            keep=False,
            reason="azonnali törlés: nincs benne értelmezhető természetes nyelvi szó",
            langdetect_hu=None,
            phunspell_ratio=None,
            lingua_hu=None,
            word_count=0,
            normalized_text=cleaned,
            forced_delete=True,
        )

    if not cleaned or not words:
        return AnalysisResult(
            score=0.0,
            keep=False,
            reason="azonnali törlés: a blokk kiürült a takarítás után",
            langdetect_hu=None,
            phunspell_ratio=None,
            lingua_hu=None,
            word_count=0,
            normalized_text=cleaned,
            forced_delete=True,
        )

    ldetect = langdetect_hu_probability(cleaned)
    ps_ratio = phunspell_ratio(words, detectors.spell)
    linga = lingua_hu_probability(cleaned, detectors.lingua_detector)
    score = combined_score(ldetect, ps_ratio, linga)
    keep = score >= threshold

    parts = [
        f"összpontszám={score:.3f}",
        f"küszöb={threshold:.3f}",
        f"langdetect.hu={ldetect if ldetect is not None else 'n/a'}",
        f"phunspell={ps_ratio if ps_ratio is not None else 'n/a'}",
        f"lingua.hu={linga if linga is not None else 'n/a'}",
        f"szavak={len(words)}",
    ]

    if len(words) <= 3 and keep and (ps_ratio == 0.0 or ps_ratio is None) and (ldetect is None or ldetect < threshold):
        keep = False
        parts.append("rövid szöveg-heurisztika=elutasítva")

    reason = ", ".join(parts)

    return AnalysisResult(
        score=score,
        keep=keep,
        reason=reason,
        langdetect_hu=ldetect,
        phunspell_ratio=ps_ratio,
        lingua_hu=linga,
        word_count=len(words),
        normalized_text=cleaned,
    )



def format_excerpt(text: str, max_len: int = 120) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."



def log_line_cleanup(file_path: Path, block_label: str, line_text: str, reason: str) -> None:
    excerpt = format_excerpt(line_text)
    print(
        f"[SOR TÖRÖLVE] {file_path} | blokk={block_label} | ok={reason} | szöveg='{excerpt}'",
        file=sys.stdout,
    )



def log_decision(action: str, file_path: Path, block_label: str, text_for_excerpt: str, result: AnalysisResult) -> None:
    excerpt = format_excerpt(result.normalized_text or text_for_excerpt)
    print(
        f"[{action}] {file_path} | blokk={block_label} | {result.reason} | szöveg='{excerpt}'",
        file=sys.stdout,
    )



def rebuild_text(blocks: Iterable[Block]) -> str:
    pieces: list[str] = []
    for block in blocks:
        pieces.extend(block.header)
        pieces.extend(block.content)
    return "".join(pieces)



def analyze_by_line(line: str, detectors: HungarianDetectors, threshold: float) -> tuple[bool, AnalysisResult]:
    match = BY_LINE_RE.match(line)
    if not match:
        dummy = AnalysisResult(
            score=1.0,
            keep=True,
            reason="nem by-sor",
            langdetect_hu=None,
            phunspell_ratio=None,
            lingua_hu=None,
            word_count=0,
            normalized_text=line.strip(),
        )
        return True, dummy

    _, payload, _ = match.groups()
    stripped_payload = payload.strip()
    if not stripped_payload:
        dummy = AnalysisResult(
            score=1.0,
            keep=True,
            reason="üres cím, megtartva",
            langdetect_hu=None,
            phunspell_ratio=None,
            lingua_hu=None,
            word_count=0,
            normalized_text="",
        )
        return True, dummy

    result = analyze_text(stripped_payload, detectors, threshold)
    return result.keep, result



def process_file(path: Path, out_path: Path, detectors: HungarianDetectors, threshold: float, show_kept: bool) -> tuple[int, int]:
    text = read_text(path)
    blocks = parse_blocks(text)

    kept_blocks: list[Block] = []
    removed_count = 0
    checked_count = 0

    for idx, block in enumerate(blocks):
        if idx == 0:
            kept_blocks.append(block)
            continue

        if block.kind == "raw" and block.header:
            line = block.header[0]
            if BY_LINE_RE.match(line):
                checked_count += 1
                keep, result = analyze_by_line(line, detectors, threshold)
                if keep:
                    kept_blocks.append(block)
                    if show_kept:
                        log_decision("MEGTARTVA", path, "post_title", line, result)
                else:
                    removed_count += 1
                    log_decision("TÖRÖLVE", path, "post_title", line, result)
                continue

            kept_blocks.append(block)
            continue

        if block.kind not in {"body", "comment"}:
            kept_blocks.append(block)
            continue

        checked_count += 1
        cleaned_block, removed_lines = clean_block_lines(block)
        for line_text, line_reason in removed_lines:
            log_line_cleanup(path, block.kind, line_text, line_reason)

        result = analyze_text(extract_text_from_block(cleaned_block), detectors, threshold)

        if result.keep:
            kept_blocks.append(cleaned_block)
            if show_kept:
                log_decision("MEGTARTVA", path, cleaned_block.kind, extract_text_from_block(cleaned_block), result)
        else:
            removed_count += 1
            log_decision("TÖRÖLVE", path, cleaned_block.kind, extract_text_from_block(cleaned_block), result)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rebuild_text(kept_blocks), encoding="utf-8")
    return checked_count, removed_count



def main() -> int:
    args = parse_args()

    input_dir = Path(args.inputmappa)
    output_dir = Path(args.out)
    threshold = args.threshold

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Hiba: az input mappa nem létezik vagy nem mappa: {input_dir}", file=sys.stderr)
        return 2

    if not 0.0 <= threshold <= 1.0:
        print("Hiba: a --threshold értéke 0.0 és 1.0 között legyen.", file=sys.stderr)
        return 2

    detectors = HungarianDetectors()
    print(f"Input mappa: {input_dir}")
    print(f"Output mappa: {output_dir}")
    print(f"Threshold: {threshold:.3f}")
    print(f"Lingua aktív: {'igen' if detectors.lingua_detector is not None else 'nem'}")
    print(f"phunspell aktív: {'igen' if detectors.spell is not None else 'nem'}")
    if detectors.phunspell_error:
        print(f"Figyelem: {detectors.phunspell_error}")

    txt_files = sorted(input_dir.rglob("*.txt"))
    if not txt_files:
        print("Nem találtam .txt fájlokat az input mappában.")
        return 0

    total_checked = 0
    total_removed = 0

    for in_file in txt_files:
        rel = in_file.relative_to(input_dir)
        out_file = output_dir / rel
        checked, removed = process_file(
            path=in_file,
            out_path=out_file,
            detectors=detectors,
            threshold=threshold,
            show_kept=args.showkomment,
        )
        total_checked += checked
        total_removed += removed

    print("-" * 80)
    print(f"Feldolgozott fájlok: {len(txt_files)}")
    print(f"Ellenőrzött body/comment/post_title blokkok: {total_checked}")
    print(f"Törölt blokkok: {total_removed}")
    print(f"Megtartott blokkok: {total_checked - total_removed}")
    print("Kész.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
