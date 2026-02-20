from typing import Any

import datasets
import numpy as np

from manaita.preprocessor import get_preprocessor
from manaita.task_types import PreProcessorArgs


class FakeTokenizer:
    def apply_chat_template(
        self, messages: list[dict[str, str]], *, tokenize: bool = False
    ) -> str:
        assert not tokenize
        user = messages[0]["content"]
        assistant = messages[1]["content"]
        return f"<user>{user}</user><assistant><think>{assistant}</assistant>"

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(ch) for ch in text]

    def __call__(self, xs: list[str], **kwargs: Any) -> dict[str, list[list[int]]]:
        del kwargs
        return {"input_ids": [self.encode(x) for x in xs]}


class FakeAudioPreProcessor:
    sampling_rate = 16_000

    def __call__(
        self, array: list[float], *, sampling_rate: int
    ) -> dict[str, list[Any]]:
        del sampling_rate
        return {"input_values": [np.asarray(array)]}


def _args() -> PreProcessorArgs:
    return PreProcessorArgs(tokenizer=FakeTokenizer(), num_proc=None, training=False)


def test_common_preprocessors():
    ds = datasets.Dataset.from_dict(
        {
            "text": ["Hello", "WORLD"],
            "keep": [True, False],
            "values": [[1, 2], [1, 2, 3, 4]],
        }
    )

    ds = get_preprocessor("casing")(ds, PreProcessorArgs(num_proc=None))
    ds = get_preprocessor({"op": "filter_by_column", "column": "keep"})(ds, _args())
    ds = get_preprocessor(
        {"op": "filter_by_shape", "column": "values", "axis": 0, "max_length": 2}
    )(ds, _args())

    assert ds["text"] == ["hello"]
    assert ds["values"] == [[1, 2]]


def test_audio_preprocessor():
    ds = datasets.Dataset.from_list(
        [{"audio": {"array": [0.1, 0.2, 0.3], "sampling_rate": 16_000}}]
    )
    args = PreProcessorArgs(
        audio_preprocessor=FakeAudioPreProcessor(), num_proc=None, debug=False
    )

    out = get_preprocessor("audio")(ds, args)

    assert out["input_value_length"] == [3]
    assert "audio" not in out.column_names


def test_asr_preprocessors():
    ds = datasets.Dataset.from_dict({"text": ["<S> Hello </S>"]})

    cleaned = get_preprocessor("remove_gigaspeech_text_tag")(ds, _args())
    out = get_preprocessor("asr")(cleaned, _args())

    assert cleaned["text"] == ["Hello"]
    assert {"prefix_ids", "suffix_ids", "label_ids"}.issubset(out.column_names)
    assert out["label_ids"][0][:5] == FakeTokenizer().encode("Hello")[:5]


def test_s2tt_preprocessors():
    ds = datasets.Dataset.from_dict(
        {
            "transcription": ["hello"],
            "translation_ja": ["konnichiwa"],
            "translation_de": ["hallo"],
        }
    )

    selected = get_preprocessor("s2tt_random_target_lang")(ds, _args())
    assert selected["translation"][0] in {"konnichiwa", "hallo"}
    assert selected["tgt_lang"][0] in {"ja", "de"}
    assert selected["src_lang"][0] == "en"

    expanded = get_preprocessor("s2tt_expand_target_langs")(ds, _args())
    assert len(expanded) == 2
    assert set(expanded["tgt_lang"]) == {"ja", "de"}

    nothink = get_preprocessor("s2tt_nothink")(selected, _args())
    cot = get_preprocessor("s2tt_cot")(selected, _args())

    assert {"prefix_ids", "suffix_ids", "label_ids"}.issubset(nothink.column_names)
    assert {"prefix_ids", "suffix_ids", "label_ids"}.issubset(cot.column_names)


def test_audio_qa_preprocessors():
    args = _args()
    choice_ds = datasets.Dataset.from_dict(
        {
            "question": ["Which sound?"],
            "answer": ["A"],
            "choice": [["A", "B"]],
        }
    )
    freeform_ds = datasets.Dataset.from_dict(
        {"question": ["What is heard?"], "answer": ["rain"]}
    )
    scene_ds = datasets.Dataset.from_dict({"audio_id": ["a0"], "class": ["Bus"]})

    choice = get_preprocessor("audio_qa_choice")(choice_ds, args)
    freeform = get_preprocessor("audio_qa_freeform")(freeform_ds, args)
    scene = get_preprocessor("cochlscene_choice")(scene_ds, args)

    assert {"prefix_ids", "suffix_ids", "label_ids"}.issubset(choice.column_names)
    assert {"prefix_ids", "suffix_ids", "label_ids"}.issubset(freeform.column_names)
    assert {"prefix_ids", "suffix_ids", "label_ids"}.issubset(scene.column_names)
