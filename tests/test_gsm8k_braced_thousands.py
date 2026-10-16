"""Offline public task/filter/metric regression for braced thousands separators."""

from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

import lm_eval
from lm_eval.tasks import TaskManager
from lm_eval.tasks._yaml_loader import load_yaml


TASKS = ("gsm8k", "gsm8k_cot", "gsm8k_cot_zeroshot", "gsm8k_cot_llama")
# response, gold, expected flexible extraction, expected score
BRACED = [
    (r"\boxed{\$10{,}000}", "10000", "$10,000", 1),
    (r"\boxed{\$31{,}800}", "31800", "$31,800", 1),
    (r"\boxed{\$10{,}000}", "10001", "$10,000", 0),
    ("1{,}234{,}567", "1234567", "1,234,567", 1),
    ("1,234{,}567", "1234567", "1,234,567", 1),
    ("1{,}234,567", "1234567", "1,234,567", 1),
    ("-1{,}234.50", "-1234.50", "-1,234.50", 1),
    ("-$1{,}234.50", "-1234.50", "-$1,234.50", 1),
    ("+1{,}234.50", "1234.50", "1,234.50", 1),
    ("1{,}234.50", "1234.5", "1,234.50", 0),
    ("The total is 12{,}345.", "12345", "12,345.", 1),
    ("The total is 12{,}345...", "12345", "12,345...", 0),
    ("Earlier 99, then 1{,}234", "1234", "1,234", 1),
    (r"\boxed{1{,}234} then 7", "7", "7", 1),
    (r"\boxed{1{,}234} then 7", "1234", "7", 0),
    # Preserve the separate digitless-tail bug; do not fold in PR #4379.
    (r"$\boxed{1{,}234}$.", "1234", "$.", 0),
]
CONTROLS = [
    (r"\boxed{\$10,000}", "10000", "$10,000", 1),
    (r"\boxed{\$31,800}", "31800", "$31,800", 1),
    ("1234567", "1234567", "1234567", 1),
    ("1,234,567", "1234567", "1,234,567", 1),
    ("-1,234.50", "-1234.50", "-1,234.50", 1),
    ("-$1,234.50", "-1234.50", "-$1,234.50", 1),
    ("+1,234.50", "1234.50", "1,234.50", 1),
    ("1,234.50", "1234.5", "1,234.50", 0),
    ("12,345.", "12345", "12,345.", 1),
    ("12,345...", "12345", "12,345...", 0),
    ("99 then 1,234", "1234", "1,234", 1),
    ("1,234 then 7", "7", "7", 1),
    ("no answer", "7", "[invalid]", 0),
    ("no number...", "7", "...", 0),
    ("$29.00", "29", "$29.00", 0),
    ("12{,}34", "1234", "34", 0),
    ("1{,}2345", "12345", "2345", 0),
    ("1234{,}567{,}890", "1234567890", "890", 0),
    ("1{,}23{,}456", "123456", "456", 0),
    ("1.2{,}345", "12345", "345", 0),
    (".1{,}234", "1234", "234", 0),
    ("{,}1{,}234", "1234", "234", 0),
    (",1{,}234", "1234", "234", 0),
    ("{,}", "7", "[invalid]", 0),
    ("{42}", "42", "42", 1),
]
CASES = BRACED + CONTROLS


def local_dataset(**kwargs):
    return DatasetDict(
        train=Dataset.from_list(
            [{"question": "Training example?", "answer": "Training rationale. #### 6"}]
        ),
        test=Dataset.from_list(
            [
                {"question": f"Evaluation {i}?", "answer": f"Rationale. #### {gold}"}
                for i, (_, gold, _, _) in enumerate(CASES)
            ]
        ),
    )


def evaluate(task_name):
    task_dir = Path(lm_eval.__file__).parent / "tasks" / "gsm8k"
    manager = TaskManager(include_path=task_dir, include_defaults=False)
    config = load_yaml(manager.task_index[task_name].yaml_path)
    config.update(custom_dataset=local_dataset, num_fewshot=0)
    task = manager.load(config)["tasks"][task_name]
    task.set_fewshot_seed(23)
    task.build_all_requests(rank=0, world_size=1, cache_requests=False)
    assert len(task.instances) == len(CASES)
    for instance, (response, _, _, _) in zip(task.instances, CASES, strict=True):
        instance.resps = [response]
    task.apply_filters()
    rows = []
    for instance in task.instances:
        rows.append(
            {
                "raw": instance.resps,
                "filtered": dict(instance.filtered_resps),
                "scores": {
                    name: task.process_results(instance.doc, [response])["exact_match"]
                    for name, response in instance.filtered_resps.items()
                },
                "prompt": instance.args[0],
            }
        )
    # A real one-shot prompt, with the existing sampler and task-specific examples.
    fewshot = task.fewshot_context(task.test_docs()[0], num_fewshot=1)
    strict = []
    for number in ("10000", "10{,}000"):
        response = (
            f"#### {number}\nThe answer is {number}.\nThe final answer is {number}"
        )
        task.instances[0].resps = [response]
        task.apply_filters()
        instance = task.instances[0]
        value = instance.filtered_resps["strict-match"]
        strict.append(
            (value, task.process_results(instance.doc, [value])["exact_match"])
        )
        assert instance.resps == [response]
    return {"task": task_name, "rows": rows, "fewshot": fewshot, "strict": strict}


@pytest.fixture(scope="module", params=TASKS)
def evaluation(request):
    return evaluate(request.param)


@pytest.mark.parametrize("index", range(len(CASES)))
def test_flexible_extraction_and_scoring(evaluation, index):
    response, _, extracted, score = CASES[index]
    row = evaluation["rows"][index]
    assert row["raw"] == [response]
    assert row["filtered"]["flexible-extract"] == extracted
    assert row["scores"]["flexible-extract"] == score


def test_strict_and_fewshot_controls(evaluation):
    # These responses have no strict answer prefix; normalizing flexible extraction
    # must not alter strict results or the original model response.
    for row, (response, _, _, _) in zip(evaluation["rows"], CASES, strict=True):
        assert row["filtered"]["strict-match"] == "[invalid]"
        assert row["scores"]["strict-match"] == 0
        assert row["raw"] == [response]
    assert evaluation["fewshot"].endswith(evaluation["rows"][0]["prompt"])
    assert len(evaluation["fewshot"]) > len(evaluation["rows"][0]["prompt"])
    assert evaluation["strict"] == [("10000", 1), ("10", 0)]
    assert "10000" not in evaluation["fewshot"]
