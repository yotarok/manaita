from typing import Any

import datasets
import jinja2

from manaita.preprocessor import register_preprocessor
from manaita.preprocessors.common import DatasetRow, Tokenizer, tokenize_columns
from manaita.task_types import PreProcessorArgs

DEFAULT_S2TT_INSTRUCTIONS = [
    "translate the contents of this audio into {{ tgt_lang }}",
    "please provide a translation of this audio in {{ tgt_lang }}",
    "Convert this speech in {{ src_lang }} into text in {{ tgt_lang }}",
    "Generate a {{ tgt_lang }} text transcript of the recording",
    "Produce a {{ tgt_lang }} translation of the audio clip in {{ src_lang }}.",
    "Translate the {{ src_lang }} speech contained in this file into {{ tgt_lang }}.",
]

DEFAULT_S2TT_COT_THOUGHTS = [
    "The provided audio contains a {{ src_lang }} speech, and it can be transcribed as: {{ transcription }}",
    "The audio is a speech in {{ src_lang }}, it can be transcribed as: {{ transcription }}. I am instructed to translate this to {{ tgt_lang }}.",
    "Okay the provided audio contains voice in {{ src_lang }}. The voice seems to be saying: {{ transcription }}. The user want me to translate it to {{ tgt_lang }}.",
]

LOCALE_CODE_TO_LANGUAGE_NAME_EN_COVOST2 = {
    "en": "English",
    "ar": "Arabic",
    "ca": "Catalan",
    "cy": "Welsh",
    "de": "German",
    "et": "Estonian",
    "fa": "Persian",
    "id": "Indonesian",
    "ja": "Japanese",
    "lv": "Latvian",
    "mn": "Mongolian",
    "sl": "Slovenian",
    "sv-SE": "Swedish",
    "ta": "Tamil",
    "tr": "Turkish",
    "zh-CN": "Chinese",
}

LOCALE_CODE_TO_LANGUAGE_NAME_EN_FLEURS: dict[str, str] = {
    "af_za": "Afrikaans",
    "am_et": "Amharic",
    "ar_eg": "Arabic",
    "as_in": "Assamese",
    "ast_es": "Asturian",
    "az_az": "Azerbaijani",
    "be_by": "Belarusian",
    "bg_bg": "Bulgarian",
    "bn_in": "Bengali",
    "bs_ba": "Bosnian",
    "ca_es": "Catalan",
    "ceb_ph": "Cebuano",
    "ckb_iq": "Central Kurdish",
    "cmn_hans_cn": "Mandarin Chinese (Simplified)",
    "cs_cz": "Czech",
    "cy_gb": "Welsh",
    "da_dk": "Danish",
    "de_de": "German",
    "el_gr": "Greek",
    "en_us": "English",
    "es_419": "Spanish (Latin American)",
    "et_ee": "Estonian",
    "fa_ir": "Persian",
    "ff_sn": "Fulah",
    "fi_fi": "Finnish",
    "fil_ph": "Filipino",
    "fr_fr": "French",
    "ga_ie": "Irish",
    "gl_es": "Galician",
    "gu_in": "Gujarati",
    "ha_ng": "Hausa",
    "he_il": "Hebrew",
    "hi_in": "Hindi",
    "hr_hr": "Croatian",
    "hu_hu": "Hungarian",
    "hy_am": "Armenian",
    "id_id": "Indonesian",
    "ig_ng": "Igbo",
    "is_is": "Icelandic",
    "it_it": "Italian",
    "ja_jp": "Japanese",
    "jv_id": "Javanese",
    "ka_ge": "Georgian",
    "kam_ke": "Kamba",
    "kea_cv": "Kabuverdianu",
    "kk_kz": "Kazakh",
    "km_kh": "Khmer",
    "kn_in": "Kannada",
    "ko_kr": "Korean",
    "ky_kg": "Kyrgyz",
    "lb_lu": "Luxembourgish",
    "lg_ug": "Ganda",
    "ln_cd": "Lingala",
    "lo_la": "Lao",
    "lt_lt": "Lithuanian",
    "luo_ke": "Luo",
    "lv_lv": "Latvian",
    "mi_nz": "Maori",
    "mk_mk": "Macedonian",
    "ml_in": "Malayalam",
    "mn_mn": "Mongolian",
    "mr_in": "Marathi",
    "ms_my": "Malay",
    "mt_mt": "Maltese",
    "my_mm": "Burmese",
    "nb_no": "Norwegian Bokmal",
    "ne_np": "Nepali",
    "nl_nl": "Dutch",
    "nso_za": "Northern Sotho",
    "ny_mw": "Chichewa",
    "oc_fr": "Occitan",
    "om_et": "Oromo",
    "or_in": "Odia",
    "pa_in": "Punjabi",
    "pl_pl": "Polish",
    "ps_af": "Pashto",
    "pt_br": "Portuguese (Brazil)",
    "ro_ro": "Romanian",
    "ru_ru": "Russian",
    "sd_in": "Sindhi",
    "sk_sk": "Slovak",
    "sl_si": "Slovenian",
    "sn_zw": "Shona",
    "so_so": "Somali",
    "sr_rs": "Serbian",
    "sv_se": "Swedish",
    "sw_ke": "Swahili",
    "ta_in": "Tamil",
    "te_in": "Telugu",
    "tg_tj": "Tajik",
    "th_th": "Thai",
    "tr_tr": "Turkish",
    "uk_ua": "Ukrainian",
    "umb_ao": "Umbundu",
    "ur_pk": "Urdu",
    "uz_uz": "Uzbek",
    "vi_vn": "Vietnamese",
    "wo_sn": "Wolof",
    "xh_za": "Xhosa",
    "yo_ng": "Yoruba",
    "yue_hant_hk": "Cantonese",
    "zu_za": "Zulu",
}

LOCALE_CODE_TO_LANGUAGE_NAME_EN = {
    **LOCALE_CODE_TO_LANGUAGE_NAME_EN_COVOST2,
    **LOCALE_CODE_TO_LANGUAGE_NAME_EN_FLEURS,
}


def _unbatch(rows: DatasetRow) -> list[DatasetRow]:
    keys = rows.keys()
    return [dict(zip(keys, values)) for values in zip(*rows.values())]


@register_preprocessor("s2tt_random_target_lang", version="20250115_v0")
def s2tt_random_target_lang(
    ds: datasets.Dataset, args: PreProcessorArgs, **kwargs: Any
) -> datasets.Dataset:
    del kwargs

    def _choose_random_target_lang(
        row: DatasetRow,
        *,
        column_to_lang: dict[str, str],
        translation_column: str = "translation",
        tgt_lang_column: str = "tgt_lang",
        src_lang_column: str = "src_lang",
        src_lang_constant: str = "en",
        hash_src_column: str = "transcription",
    ) -> DatasetRow:
        columns = list(column_to_lang.keys())
        column_chosen = columns[hash(row[hash_src_column]) % len(columns)]

        row[translation_column] = row[column_chosen]
        row[tgt_lang_column] = column_to_lang[column_chosen]
        row[src_lang_column] = src_lang_constant
        return row

    columns = [col for col in ds.column_names if col.startswith("translation_")]
    column_to_lang = {}

    for col in columns:
        _, langcode = col.split("_", maxsplit=1)
        column_to_lang[col] = langcode

    return ds.map(
        _choose_random_target_lang,
        fn_kwargs={"column_to_lang": column_to_lang},
        num_proc=args.num_proc,
        desc=f"Selecting translation targets [{ds.info.dataset_name}]",
    )


@register_preprocessor("s2tt_expand_target_langs", version="20250115_v0")
def s2tt_expand_target_langs(
    ds: datasets.Dataset,
    args: PreProcessorArgs,
    column_to_lang: dict[str, str] | None = None,
    **kwargs: Any,
) -> datasets.Dataset:
    del kwargs

    def _expand_target_lang(
        batched_rows: DatasetRow,
        *,
        column_to_lang: dict[str, str],
        translation_column: str = "translation",
        tgt_lang_column: str = "tgt_lang",
        src_lang_column: str = "src_lang",
        src_lang_constant: str = "en",
    ) -> DatasetRow:
        tr_columns = list(column_to_lang.keys())

        rows = _unbatch(batched_rows)
        new_rows = []
        for row in rows:
            for column in tr_columns:
                newrow = row.copy()
                newrow[translation_column] = row[column]
                newrow[tgt_lang_column] = column_to_lang[column]
                newrow[src_lang_column] = src_lang_constant
                newrow.update({k: v for k, v in row.items() if k not in tr_columns})
                new_rows.append(newrow)
        return {k: [r[k] for r in new_rows] for k in new_rows[0].keys()}

    tr_columns = [col for col in ds.column_names if col.startswith("translation_")]

    if column_to_lang is None:
        column_to_lang = {}
        for col in tr_columns:
            _, langcode = col.split("_", maxsplit=1)
            column_to_lang[col] = langcode

    return ds.map(
        _expand_target_lang,
        batched=True,
        batch_size=10,
        fn_kwargs={"column_to_lang": column_to_lang},
        remove_columns=tr_columns,
        num_proc=args.num_proc,
        desc=f"Expanding translation targets [{ds.info.dataset_name}]",
    )


@register_preprocessor("s2tt_nothink", version="20250115_v0")
def preprocess_for_s2tt_nothink(
    ds: datasets.Dataset,
    args: PreProcessorArgs,
    audio_placeholder: str = "___AUDIO___",
    target_text_placeholder: str = "___TARGET_TEXT___",
    **kwargs: Any,
) -> datasets.Dataset:
    del kwargs

    def _format_and_tokenize_s2tt_nothink(
        row: DatasetRow,
        *,
        tokenizer: Tokenizer,
        instructions: list[str],
        training: bool,
        translation_column: str = "translation",
        tgt_lang_column: str = "tgt_lang",
        src_lang_column: str = "src_lang",
    ) -> DatasetRow:
        translation = str(row[translation_column])
        row_src_lang = str(row[src_lang_column])
        row_tgt_lang = str(row[tgt_lang_column])
        src_lang = LOCALE_CODE_TO_LANGUAGE_NAME_EN.get(row_src_lang, row_src_lang)
        tgt_lang = LOCALE_CODE_TO_LANGUAGE_NAME_EN.get(row_tgt_lang, row_tgt_lang)

        if training:
            instruction = instructions[hash(translation) % len(instructions)]
        else:
            instruction = instructions[0]

        instruction = jinja2.Template(instruction).render(
            src_lang=src_lang, tgt_lang=tgt_lang, translation=translation
        )
        chat_text = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": f"<|audio_start|>{audio_placeholder}<|audio_end|>\n{instruction}",
                },
                {"role": "assistant", "content": target_text_placeholder},
            ],
            tokenize=False,
        )
        if not isinstance(chat_text, str):
            raise TypeError("tokenizer.apply_chat_template must return str")

        instruction_text, label_suffix = chat_text.split(
            "___TARGET_TEXT___", maxsplit=1
        )
        target_labels = translation + label_suffix

        instruction_prefix, instruction_suffix = instruction_text.split(
            "___AUDIO___", maxsplit=1
        )

        row["prefix_ids"] = tokenizer.encode(
            instruction_prefix, add_special_tokens=False
        )
        row["suffix_ids"] = tokenizer.encode(
            instruction_suffix, add_special_tokens=False
        )
        row["label_ids"] = tokenizer.encode(
            target_labels.rstrip(), add_special_tokens=False
        )
        return row

    return ds.map(
        _format_and_tokenize_s2tt_nothink,
        fn_kwargs={
            "tokenizer": args.tokenizer,
            "instructions": DEFAULT_S2TT_INSTRUCTIONS,
            "training": args.training,
        },
        num_proc=args.num_proc,
        desc=f"Formatting and tokenizing (S2TT) [{ds.info.dataset_name}]",
    )


@register_preprocessor("s2tt_cot", version="20250115_v0")
def preprocess_for_s2tt_cot(
    ds: datasets.Dataset,
    args: PreProcessorArgs,
    transcription_column: str = "transcription",
    translation_column: str = "translation",
    tgt_lang_column: str = "tgt_lang",
    src_lang_column: str = "src_lang",
    seed: int = 0,
    audio_placeholder: str = "___AUDIO___",
    target_text_placeholder: str = "___TARGET_TEXT___",
    **kwargs: Any,
) -> datasets.Dataset:
    del kwargs

    def _format(
        translation: str,
        transcription: str,
        tgt_lang: str,
        src_lang: str,
        *,
        tokenizer: Tokenizer,
        instructions: list[str],
        thoughts: list[str],
        training: bool,
        seed: int,
    ) -> DatasetRow:
        src_lang = LOCALE_CODE_TO_LANGUAGE_NAME_EN.get(src_lang, src_lang)
        tgt_lang = LOCALE_CODE_TO_LANGUAGE_NAME_EN.get(tgt_lang, tgt_lang)

        if training:
            randkey = translation + "THINK" + "" if seed == 0 else f"{seed}"
            instruction = instructions[hash(randkey) % len(instructions)]
            randkey = translation + "THINK_" + "" if seed == 0 else f"{seed}"
            thought = thoughts[hash(randkey) % len(thoughts)]
        else:
            instruction = instructions[0]
            thought = thoughts[0]

        instruction = jinja2.Template(instruction).render(
            src_lang=src_lang, tgt_lang=tgt_lang, translation=translation
        )
        thought = jinja2.Template(thought).render(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            translation=translation,
            transcription=transcription,
        )
        chat_text = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": f"<|audio_start|>{audio_placeholder}<|audio_end|>\n{instruction}",
                },
                {"role": "assistant", "content": target_text_placeholder},
            ],
            tokenize=False,
        )
        if not isinstance(chat_text, str):
            raise TypeError("tokenizer.apply_chat_template must return str")

        instruction_text, label_suffix = chat_text.split("<think>", maxsplit=1)
        _, label_suffix = label_suffix.split("___TARGET_TEXT___", maxsplit=1)
        target_labels = (
            "<think>\n" + thought + "\n</think>\n\n" + translation + label_suffix
        )

        instruction_prefix, instruction_suffix = instruction_text.split(
            "___AUDIO___", maxsplit=1
        )
        return {
            "_prefix_text": instruction_prefix,
            "_suffix_text": instruction_suffix,
            "_label_text": target_labels.rstrip(),
        }

    ds = ds.map(
        _format,
        fn_kwargs={
            "tokenizer": args.tokenizer,
            "instructions": DEFAULT_S2TT_INSTRUCTIONS,
            "thoughts": DEFAULT_S2TT_COT_THOUGHTS,
            "training": args.training,
            "seed": seed,
        },
        input_columns=[
            translation_column,
            transcription_column,
            tgt_lang_column,
            src_lang_column,
        ],
        num_proc=args.num_proc,
        desc=f"Formatting for S2TT-CoT [{ds.info.dataset_name}]",
    )
    return tokenize_columns(
        ds,
        {
            "_prefix_text": "prefix_ids",
            "_suffix_text": "suffix_ids",
            "_label_text": "label_ids",
        },
        args.tokenizer,
        remove_text_columns=True,
        num_proc=args.num_proc,
        tokenizer_kwargs={"add_special_tokens": False},
    )
