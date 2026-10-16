"""Offline DROP target and prompt tests using real task YAML and Datasets."""

from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

import lm_eval
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks import TaskManager
from lm_eval.tasks._yaml_loader import load_yaml


EXPECTED_TARGETS = ["cedar", "7", "3 May 2001", "red, blue"]
LEGACY_TARGET = "number,date,spans,worker_id,hit_id"
EVAL_TEXT = "The unlabelled evaluation passage. What is the held-out answer?"


def answer(*, number="", spans=(), day="", month="", year=""):
    return {
        "number": number,
        "date": {"day": day, "month": month, "year": year},
        "spans": list(spans),
        "worker_id": "",
        "hit_id": "",
    }


def document(query_id, primary, alternate):
    return {
        "query_id": query_id,
        "passage": f"Passage for {query_id}.",
        "question": f"What is the answer to {query_id}?",
        "answer": primary,
        "validated_answers": {key: [value] for key, value in alternate.items()},
    }


def local_dataset(**kwargs):
    rows = [
        document("single", answer(spans=["cedar"]), answer(spans=["oak"])),
        document("number", answer(number="7"), answer(number="7")),
        document(
            "date",
            answer(day="3", month="May", year="2001"),
            answer(day="3", month="May", year="2001"),
        ),
        document(
            "multi",
            answer(spans=["red", "blue"]),
            answer(spans=["green", "yellow"]),
        ),
    ]
    evaluation = document(
        "eval", answer(spans=["HELD_OUT_GOLD"]), answer(spans=["HELD_OUT_GOLD"])
    )
    evaluation.update(
        passage="The unlabelled evaluation passage.",
        question="What is the held-out answer?",
    )
    return DatasetDict(
        train=Dataset.from_list(rows), validation=Dataset.from_list([evaluation])
    )


def make_task(task_name):
    task_dir = Path(lm_eval.__file__).parent / "tasks" / "drop"
    manager = TaskManager(include_path=task_dir, include_defaults=False)
    config = load_yaml(manager.task_index[task_name].yaml_path)
    config["custom_dataset"] = local_dataset
    config["fewshot_split"] = "train"
    task = manager.load(config)["tasks"][task_name]
    assert isinstance(task, ConfigurableTask)
    assert isinstance(task.dataset["train"], Dataset)
    task.set_fewshot_seed(23)
    return task


def snapshot(task):
    docs = list(task.training_docs())
    evaluation = task.validation_docs()[0]
    task.set_fewshot_seed(23)
    return {
        "targets": [task.doc_to_target(doc) for doc in docs],
        "answers": [doc["answers"] for doc in docs],
        "fewshot": task.fewshot_context(evaluation, num_fewshot=4),
        "zero_shot": task.fewshot_context(evaluation, num_fewshot=0),
        "eval_target": task.doc_to_target(evaluation),
    }


def assert_fewshot_contract(result, targets):
    # Match answer positions, not substrings that might occur in a passage.
    parts = result["fewshot"].split("\n\n")
    ids = ["single", "number", "date", "multi"]
    expected = [
        f"Passage for {query_id}. What is the answer to {query_id}?{target}"
        for query_id, target in zip(ids, targets, strict=True)
    ]
    assert sorted(parts[:-1]) == sorted(expected)
    assert parts[-1] == EVAL_TEXT
    assert "HELD_OUT_GOLD" not in result["fewshot"]
    assert result["zero_shot"] == EVAL_TEXT


@pytest.fixture(scope="module")
def fixed():
    return make_task("drop_fixed")


@pytest.fixture(scope="module")
def legacy():
    return make_task("drop")


def test_builtin_registration():
    manager = TaskManager()
    assert {"drop", "drop_fixed"} <= set(manager.all_subtasks)


@pytest.mark.parametrize("row, expected", list(enumerate(EXPECTED_TARGETS)))
def test_fixed_targets(fixed, row, expected):
    assert fixed.doc_to_target(fixed.training_docs()[row]) == expected


def test_fixed_fewshot_and_recorded_target(fixed):
    result = snapshot(fixed)
    assert_fewshot_contract(result, EXPECTED_TARGETS)
    assert result["eval_target"] == "HELD_OUT_GOLD"
    assert LEGACY_TARGET not in result["fewshot"]


def test_legacy_prompt_and_target_are_preserved(legacy):
    result = snapshot(legacy)
    assert result["targets"] == [LEGACY_TARGET] * 4
    assert result["eval_target"] == LEGACY_TARGET
    assert_fewshot_contract(result, [LEGACY_TARGET] * 4)


def test_zero_shot_context_is_unchanged(fixed, legacy):
    assert snapshot(fixed)["zero_shot"] == snapshot(legacy)["zero_shot"]


def test_complete_answer_groups_are_preserved(fixed, legacy):
    expected = [
        [["cedar"], ["oak"]],
        [["7"]],
        [["3 May 2001"]],
        [["red", "blue"], ["green", "yellow"]],
    ]
    assert snapshot(fixed)["answers"] == expected
    assert snapshot(legacy)["answers"] == expected


@pytest.mark.parametrize(
    "prediction, expected",
    [
        ("cedar", {"em": 1.0, "f1": 1.0}),
        ("oak", {"em": 1.0, "f1": 1.0}),
        ("pine", {"em": 0.0, "f1": 0.0}),
    ],
)
def test_existing_scoring_accepts_alternatives(fixed, legacy, prediction, expected):
    for task in (fixed, legacy):
        assert task.process_results(task.training_docs()[0], [prediction]) == expected


def test_partial_multispan_answer_is_not_a_complete_gold(fixed, legacy):
    for task in (fixed, legacy):
        result = task.process_results(task.training_docs()[3], ["red"])
        assert result["em"] == 0.0
        assert result["f1"] == 0.5


def test_only_target_configuration_changes(fixed, legacy):
    assert fixed.config.metadata["version"] == 1.0
    assert legacy.config.metadata["version"] == 3.0
    fixed_config = fixed.dump_config()
    legacy_config = legacy.dump_config()
    for config in (fixed_config, legacy_config):
        config.pop("task")
        config.pop("doc_to_target")
        config["fewshot_config"].pop("doc_to_target")
        config.pop("metadata")
    assert fixed_config == legacy_config
    assert fixed.config.process_docs is legacy.config.process_docs
    assert fixed.config.process_results is legacy.config.process_results
    assert fixed.config.generation_kwargs["until"] == ["."]
