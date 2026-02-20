import re
from collections.abc import Sequence
from typing import Any

import datasets

from manaita.preprocessor import register_preprocessor
from manaita.preprocessors.common import DatasetRow, Tokenizer, tokenize_columns
from manaita.task_types import PreProcessorArgs

DEFAULT_ASR_INSTRUCTIONS = [
    "transcribe this audio",
    "please provide a transcription of this audio",
    "Convert this audio into text",
    "Generate a text transcript of the recording",
    "Produce a written transcription of the audio clip.",
    "Transcribe the speech contained in this file.",
    "capture the spoken content here as text.",
    "Output a transcription of the spoken audio.",
    "write out the contents of this audio.",
    "Decode the audio into written language",
]

RE_TAGS_SURROUNDED_BY_ANGLES = re.compile(r"<[^>]+>")


@register_preprocessor("remove_gigaspeech_text_tag", version="20250115_v0")
def remove_giga_speech_text_tag(
    ds: datasets.Dataset,
    args: PreProcessorArgs,
    *,
    text_column: str = "text",
    **kwargs: Any,
) -> datasets.Dataset:
    del kwargs

    def _remove_tag(text: str) -> DatasetRow:
        return {
            text_column: RE_TAGS_SURROUNDED_BY_ANGLES.sub("", text)
            .strip()
            .replace("  ", " ")
        }

    return ds.map(
        _remove_tag,
        input_columns=[text_column],
        num_proc=args.num_proc,
        desc=f"Removing Gigaspeech tags [{ds.info.dataset_name}]",
    )


@register_preprocessor("asr", version="20260602_v0")
def preprocess_for_asr(
    ds: datasets.Dataset,
    args: PreProcessorArgs,
    *,
    text_column: str = "text",
    instructions: list[str] = DEFAULT_ASR_INSTRUCTIONS,
    seed: int = 0,
    target_text_placeholder: str = "___TARGET_TEXT___",
    audio_placeholder: str = "___AUDIO___",
    **kwargs: Any,
) -> datasets.Dataset:
    del kwargs

    def _format(
        target_text: str,
        *,
        tokenizer: Tokenizer,
        instructions: Sequence[str],
        training: bool,
        seed: int,
    ) -> DatasetRow:
        if training:
            instruction = instructions[(hash(target_text) + seed) % len(instructions)]
        else:
            instruction = instructions[0]

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

        instruction, label_suffix = chat_text.split(target_text_placeholder, maxsplit=1)
        target_labels = target_text + label_suffix

        instruction_prefix, instruction_suffix = instruction.split(
            audio_placeholder, maxsplit=1
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
            "instructions": instructions,
            "training": args.training,
            "seed": seed,
        },
        input_columns=[text_column],
        num_proc=args.num_proc,
        desc=f"Formatting for ASR [{ds.info.dataset_name}]",
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
