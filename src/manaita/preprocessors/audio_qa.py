import random
from typing import Any

import datasets

from manaita.preprocessor import register_preprocessor
from manaita.preprocessors.common import DatasetRow, Tokenizer, tokenize_columns
from manaita.task_types import PreProcessorArgs

SCENE_CLASSIFICATION_CHOICE_INSTRUCTIONS = [
    "Listen to the audio, what is the possible environment? choose from the below list\n",
    "predict a single label describing the recording scene from the list below\n",
    "determine the recording scene of the scenario. choose from the list\n",
]

COCHLSCENE_ALL_CLASSES = [
    "Bus",
    "Cafe",
    "Car",
    "CrowdedIndoor",
    "Elevator",
    "Kitchen",
    "Park",
    "ResidentialArea",
    "Restaurant",
    "Restroom",
    "Street",
    "Subway",
    "SubwayStation",
]


@register_preprocessor("audio_qa_choice", version="20250115_v0")
def preprocess_for_audio_qa_choice(
    ds: datasets.Dataset,
    args: PreProcessorArgs,
    target_text_placeholder: str = "___TARGET_TEXT___",
    audio_placeholder: str = "___AUDIO___",
    **kwargs: Any,
) -> datasets.Dataset:
    del kwargs

    def _format(
        question: str, answer: str, choice_list: list[str], *, tokenizer: Tokenizer
    ) -> DatasetRow:
        choices = "\n".join(choice_list)
        instruction = f"{question}\n{choices}"
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

        prompt, label_suffix = chat_text.split(target_text_placeholder, maxsplit=1)
        target_labels = answer + label_suffix

        prompt_prefix, prompt_suffix = prompt.split(audio_placeholder, maxsplit=1)

        return {
            "_prefix_text": prompt_prefix,
            "_suffix_text": prompt_suffix,
            "_label_text": target_labels.rstrip(),
        }

    ds = ds.map(
        _format,
        fn_kwargs={"tokenizer": args.tokenizer},
        input_columns=["question", "answer", "choice"],
        num_proc=args.num_proc,
        desc=f"Formatting for AQA-MCQ [{ds.info.dataset_name}]",
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


@register_preprocessor("audio_qa_freeform", version="20250115_v0")
def preprocess_for_audio_qa_freeform(
    ds: datasets.Dataset,
    args: PreProcessorArgs,
    col_question: str = "question",
    col_answer: str = "answer",
    additional_instruction: str = "",
    target_text_placeholder: str = "___TARGET_TEXT___",
    audio_placeholder: str = "___AUDIO___",
    **kwargs: Any,
) -> datasets.Dataset:
    del kwargs

    def _format(
        question: str,
        answer: str,
        *,
        tokenizer: Tokenizer,
        additional_instruction: str,
    ) -> DatasetRow:
        instruction = question + additional_instruction
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

        prompt, label_suffix = chat_text.split(target_text_placeholder, maxsplit=1)
        target_labels = answer + label_suffix

        prompt_prefix, prompt_suffix = prompt.split(audio_placeholder, maxsplit=1)

        return {
            "_prefix_text": prompt_prefix,
            "_suffix_text": prompt_suffix,
            "_label_text": target_labels.rstrip(),
        }

    ds = ds.map(
        _format,
        fn_kwargs={
            "tokenizer": args.tokenizer,
            "additional_instruction": additional_instruction,
        },
        input_columns=[col_question, col_answer],
        num_proc=args.num_proc,
        desc=f"Formatting for AQA-FF [{ds.info.dataset_name}]",
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


@register_preprocessor("cochlscene_choice", version="20250115_v0")
def preprocess_for_cochlscene_choice(
    ds: datasets.Dataset,
    args: PreProcessorArgs,
    randomize: bool = False,
    audio_placeholder: str = "___AUDIO___",
    target_text_placeholder: str = "___TARGET_TEXT___",
    **kwargs: Any,
) -> datasets.Dataset:
    del kwargs

    def _format(audio_id: str, target_tag: str, *, tokenizer: Tokenizer) -> DatasetRow:
        instruction_id = hash(audio_id) % len(SCENE_CLASSIFICATION_CHOICE_INSTRUCTIONS)
        instruction = SCENE_CLASSIFICATION_CHOICE_INSTRUCTIONS[instruction_id]
        if randomize:
            choices = [
                tag
                for tag in COCHLSCENE_ALL_CLASSES
                if hash(audio_id + ":" + tag) % 2 == 0
            ]
            if target_tag not in choices:
                choices.append(target_tag)

            random.seed(hash(audio_id))
            random.shuffle(choices)
        else:
            choices = COCHLSCENE_ALL_CLASSES

        marks = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]
        if len(choices) > len(marks):
            raise ValueError(f"too many choices: {choices}")

        target_idx = choices.index(target_tag)
        marked_choices = [f"{mark}. {tag}" for mark, tag in zip(marks, choices)]
        choices_str = "\n".join(marked_choices)

        instruction = f"{instruction}\n{choices_str}"
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

        prompt, label_suffix = chat_text.split("___TARGET_TEXT___", maxsplit=1)
        target_labels = marked_choices[target_idx] + label_suffix

        prompt_prefix, prompt_suffix = prompt.split("___AUDIO___", maxsplit=1)

        return {
            "_prefix_text": prompt_prefix,
            "_suffix_text": prompt_suffix,
            "_label_text": target_labels.rstrip(),
        }

    ds = ds.map(
        _format,
        fn_kwargs={"tokenizer": args.tokenizer},
        input_columns=["audio_id", "class"],
        num_proc=args.num_proc,
        desc=f"Formatting for CochlScene [{ds.info.dataset_name}]",
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
