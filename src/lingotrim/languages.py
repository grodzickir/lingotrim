"""Language codes -> corpus source and Unicode script. ISO 639-1 keys; FineWeb-2 uses ISO 639-3 + script."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    fineweb2: str | None
    script: str


# FineWeb-2 config names are `{iso639-3}_{ISO 15924 script}`.
_TABLE = [
    ("en", "English", None, "Latin"),
    ("pl", "Polish", "pol_Latn", "Latin"),
    ("de", "German", "deu_Latn", "Latin"),
    ("fr", "French", "fra_Latn", "Latin"),
    ("es", "Spanish", "spa_Latn", "Latin"),
    ("it", "Italian", "ita_Latn", "Latin"),
    ("pt", "Portuguese", "por_Latn", "Latin"),
    ("nl", "Dutch", "nld_Latn", "Latin"),
    ("sv", "Swedish", "swe_Latn", "Latin"),
    ("da", "Danish", "dan_Latn", "Latin"),
    ("no", "Norwegian", "nob_Latn", "Latin"),
    ("fi", "Finnish", "fin_Latn", "Latin"),
    ("cs", "Czech", "ces_Latn", "Latin"),
    ("sk", "Slovak", "slk_Latn", "Latin"),
    ("hu", "Hungarian", "hun_Latn", "Latin"),
    ("ro", "Romanian", "ron_Latn", "Latin"),
    ("hr", "Croatian", "hrv_Latn", "Latin"),
    ("sl", "Slovenian", "slv_Latn", "Latin"),
    ("lt", "Lithuanian", "lit_Latn", "Latin"),
    ("lv", "Latvian", "lvs_Latn", "Latin"),
    ("et", "Estonian", "ekk_Latn", "Latin"),
    ("tr", "Turkish", "tur_Latn", "Latin"),
    ("vi", "Vietnamese", "vie_Latn", "Latin"),
    ("id", "Indonesian", "ind_Latn", "Latin"),
    ("ms", "Malay", "zsm_Latn", "Latin"),
    ("ca", "Catalan", "cat_Latn", "Latin"),
    ("uk", "Ukrainian", "ukr_Cyrl", "Cyrillic"),
    ("ru", "Russian", "rus_Cyrl", "Cyrillic"),
    ("bg", "Bulgarian", "bul_Cyrl", "Cyrillic"),
    ("sr", "Serbian", "srp_Cyrl", "Cyrillic"),
    ("el", "Greek", "ell_Grek", "Greek"),
    ("he", "Hebrew", "heb_Hebr", "Hebrew"),
    ("ar", "Arabic", "arb_Arab", "Arabic"),
    ("fa", "Persian", "pes_Arab", "Arabic"),
    ("hi", "Hindi", "hin_Deva", "Devanagari"),
    ("bn", "Bengali", "ben_Beng", "Bengali"),
    ("ta", "Tamil", "tam_Taml", "Tamil"),
    ("th", "Thai", "tha_Thai", "Thai"),
    ("zh", "Chinese", "cmn_Hani", "Han"),
    ("ja", "Japanese", "jpn_Jpan", "Han"),
    ("ko", "Korean", "kor_Hang", "Hangul"),
]
LANGUAGES: dict[str, Language] = {c: Language(c, n, f, s) for c, n, f, s in _TABLE}


def resolve(code: str) -> Language:
    """Accept an ISO 639-1 code from the table or a raw FineWeb-2 config like `pol_Latn`."""
    code = code.strip().lower()
    if code in LANGUAGES:
        return LANGUAGES[code]
    if "_" in code and len(code.split("_")[0]) == 3:
        iso3, script = code.split("_", 1)
        return Language(iso3, iso3, f"{iso3}_{script.capitalize()}", script.capitalize())
    known = ", ".join(sorted(LANGUAGES))
    raise ValueError(f"unknown language {code!r}; use one of {known} or a FineWeb-2 config like pol_Latn")
