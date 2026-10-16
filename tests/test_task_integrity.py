"""Offline regression coverage for integrity checks on resolved task objects."""

import builtins
import json

import pytest

from lm_eval import evaluator
from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask, Task
from lm_eval.evaluator_utils import run_task_tests
from lm_eval.models.dummy import DummyLM
from lm_eval.tasks import TaskManager


class RecordingLM(DummyLM):
    """Record the real evaluator's requests without loading a model or tokenizer."""

    def __init__(self):
        super().__init__()
        self.requests = []

    def loglikelihood(self, requests, disable_tqdm=False):
        requests = list(requests)
        self.requests.extend(request.args for request in requests)
        return [(-1.0, False) for _ in requests]


@pytest.fixture
def task_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    tasks = tmp_path / "tasks"
    tasks.mkdir()

    def write_json(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def rows(prefix, count, *, bad=False):
        return [
            {
                "question": f"{prefix}-{index}?" + (" " if bad else ""),
                "choices": ["alpha", "beta"],
                "answer": index % 2,
            }
            for index in range(count)
        ]

    train_path = tmp_path / "train.json"
    write_json(train_path, rows("training", 8))
    for name, count, fewshot, bad in [
        ("integrity_one", 1, 0, False),
        ("integrity_five", 5, 0, False),
        ("integrity_one_bad", 5, 2, True),
        ("integrity_fewshot", 5, 2, False),
    ]:
        data_path = tmp_path / f"{name}.json"
        write_json(data_path, rows(name, count, bad=bad))
        # JSON is valid YAML; every dataset file stays under this fixture's tmp_path.
        write_json(
            tasks / f"{name}.yaml",
            {
                "task": name,
                "dataset_path": "json",
                "dataset_kwargs": {
                    "data_files": {"test": str(data_path), "train": str(train_path)},
                    "cache_dir": str(tmp_path / "cache"),
                },
                "test_split": "test",
                "training_split": "train",
                "fewshot_split": "train",
                "output_type": "multiple_choice",
                "doc_to_text": "question",
                "doc_to_choice": "choices",
                "doc_to_target": "answer",
                "target_delimiter": " ",
                "num_fewshot": fewshot,
                "metric_list": [
                    {"metric": "acc", "aggregation": "mean", "higher_is_better": True}
                ],
                "metadata": {"version": 1},
            },
        )
    write_json(
        tasks / "nested_group.yaml",
        {"group": "integrity_nested", "task": ["integrity_five"]},
    )
    write_json(
        tasks / "group.yaml",
        {
            "group": "integrity_group",
            "task": [
                {"task": "integrity_one", "description": "group-only prefix"},
                "integrity_nested",
            ],
        },
    )
    write_json(
        tasks / "bad_group.yaml",
        {
            "group": "integrity_bad_group",
            "task": ["integrity_one", "integrity_one_bad"],
        },
    )
    return tasks


def make_manager(task_dir):
    return TaskManager(include_path=str(task_dir), include_defaults=False)


def evaluate_selection(manager, selector, model, *, check_integrity=True):
    return evaluator.simple_evaluate(
        model=model,
        tasks=manager.match_tasks([selector]),
        task_manager=manager,
        check_integrity=check_integrity,
        limit=1,
        bootstrap_iters=0,
        log_samples=False,
        random_seed=0,
        numpy_random_seed=0,
        torch_random_seed=None,
        fewshot_random_seed=17,
    )


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ("integrity_one", {"integrity_one"}),
        ("integrity_group", {"integrity_one", "integrity_five"}),
        ("integrity_*e", {"integrity_one", "integrity_five"}),
    ],
)
def test_evaluator_checks_exact_resolved_tasks(
    task_dir, monkeypatch, selector, expected
):
    manager = make_manager(task_dir)
    loaded = []
    checked = []
    load = manager.load

    def record_load(specs):
        result = load(specs)
        loaded.append(result)
        return result

    def record_check(*, task_dict, **kwargs):
        assert task_dict is loaded[-1]["tasks"]
        checked.append(task_dict)
        # Execute the actual pytest suite, including its structural assertions.
        run_task_tests(task_dict=task_dict, **kwargs)

    monkeypatch.setattr(manager, "load", record_load)
    monkeypatch.setattr(evaluator, "run_task_tests", record_check)
    model = RecordingLM()
    result = evaluate_selection(manager, selector, model)

    assert len(checked) == 1
    assert set(checked[0]) == expected
    assert set(result["results"]) == expected | set(loaded[-1]["groups"])
    assert len(model.requests) == 2 * len(expected)
    for name, task in checked[0].items():
        assert isinstance(task, ConfigurableTask)
        assert task.config.dataset_kwargs["data_files"]["test"] == str(
            task_dir.parent / f"{name}.json"
        )
        assert task.config.metadata["config_source"] == str(task_dir / f"{name}.yaml")
    if selector == "integrity_group":
        assert checked[0]["integrity_one"].config.description == "group-only prefix"
        assert any("group-only prefix" in context for context, _ in model.requests)
    # The similarly named invalid task is registered but must never be checked.
    assert "integrity_one_bad" in manager.all_tasks
    assert "integrity_one_bad" not in checked[0]


@pytest.mark.parametrize("selector", ["integrity_one_bad", "integrity_bad_group"])
def test_bad_selected_task_fails_before_model_requests(task_dir, selector):
    model = RecordingLM()
    with pytest.raises(ValueError, match="integrity_one_bad"):
        evaluate_selection(make_manager(task_dir), selector, model)
    assert model.requests == []


def test_repeated_checks_restore_state_and_ignore_outer_pytest_filters(
    task_dir, monkeypatch
):
    manager = make_manager(task_dir)
    # If inherited by nested pytest, this would deselect every integrity check.
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k no_integrity_test_can_match_this")
    for name in ("integrity_one", "integrity_one_bad", "integrity_five"):
        task_dict = manager.load([name])["tasks"]
        task = task_dict[name]
        task.set_fewshot_seed(17)
        if name == "integrity_five":
            task.build_all_requests(limit=1)
        had_instances = hasattr(task, "_instances")
        instances = getattr(task, "_instances", None)
        fewshot_state = task.fewshot_rnd.getstate()
        sampler_state = task.sampler.rnd.getstate()
        sampler_docs = task.sampler.df
        sampler_loaded = task.sampler._loaded

        if name == "integrity_one_bad":
            with pytest.raises(ValueError, match=name):
                run_task_tests(task_dict=task_dict)
        else:
            run_task_tests(task_dict=task_dict)

        assert hasattr(task, "_instances") == had_instances
        assert getattr(task, "_instances", None) is instances
        assert task.fewshot_rnd.getstate() == fewshot_state
        assert task.sampler.rnd.getstate() == sampler_state
        assert task.sampler.df is sampler_docs
        assert task.sampler._loaded == sampler_loaded


def test_integrity_does_not_change_nonzero_fewshot_requests(task_dir):
    unchecked = RecordingLM()
    checked = RecordingLM()
    evaluate_selection(
        make_manager(task_dir), "integrity_fewshot", unchecked, check_integrity=False
    )
    evaluate_selection(make_manager(task_dir), "integrity_fewshot", checked)

    assert len(unchecked.requests) == 2
    assert unchecked.requests[0][0].count("training-") == 2
    assert checked.requests == unchecked.requests


@pytest.mark.parametrize("as_object", [False, True], ids=["config", "task-object"])
def test_python_task_specs_keep_local_configuration(task_dir, as_object):
    manager = make_manager(task_dir)
    config = json.loads((task_dir / "integrity_one.yaml").read_text(encoding="utf-8"))
    task_spec = ConfigurableTask(config=config) if as_object else config
    model = RecordingLM()
    result = evaluator.simple_evaluate(
        model=model,
        tasks=[task_spec],
        task_manager=manager,
        check_integrity=True,
        limit=1,
        bootstrap_iters=0,
        log_samples=False,
        torch_random_seed=None,
    )
    assert set(result["results"]) == {"integrity_one"}
    assert len(model.requests) == 2


def test_empty_task_mapping_is_rejected():
    with pytest.raises(ValueError, match="No resolved tasks"):
        run_task_tests(task_dict={})


def test_missing_optional_pytest_has_installation_hint(task_dir, monkeypatch):
    task_dict = make_manager(task_dir).load(["integrity_one"])["tasks"]
    original_import = builtins.__import__

    def without_pytest(name, *args, **kwargs):
        if name == "pytest":
            raise ModuleNotFoundError("No module named 'pytest'", name="pytest")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pytest)
    with pytest.raises(ImportError, match=r"pip install.*lm_eval\[dev\]"):
        run_task_tests(task_dict=task_dict)


def test_handwritten_task_uses_public_docs_and_decontamination_methods():
    class PublicMethodsTask(Task):
        OUTPUT_TYPE = "loglikelihood"

        def __init__(self):
            super().__init__()
            self.set_config("task", "integrity_public_methods")
            self.set_config("output_type", self.OUTPUT_TYPE)
            self.set_fewshot_seed(17)
            self.docs_read = set()
            self.queries = []

        def download(self, data_dir=None, cache_dir=None, download_mode=None):
            self.dataset = {
                split: [{"question": f"{split} question?", "answer": " yes"}]
                for split in ("training", "validation", "test")
            }

        def has_training_docs(self):
            return True

        def has_validation_docs(self):
            return True

        def has_test_docs(self):
            return True

        def training_docs(self):
            self.docs_read.add("training")
            return self.dataset["training"]

        def validation_docs(self):
            self.docs_read.add("validation")
            return self.dataset["validation"]

        def test_docs(self):
            self.docs_read.add("test")
            return self.dataset["test"]

        def should_decontaminate(self):
            return True

        def doc_to_decontamination_query(self, doc):
            self.queries.append(doc)
            return doc["question"]

        def doc_to_text(self, doc):
            return doc["question"]

        def doc_to_target(self, doc):
            return doc["answer"]

        def construct_requests(self, doc, ctx, **kwargs):
            return Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(ctx, self.doc_to_target(doc)),
                idx=0,
                metadata=kwargs.get("metadata", (None, None, None)),
            )

        def process_results(self, doc, results):
            return {"acc": int(results[0][1])}

        def aggregation(self):
            return {"acc": lambda values: sum(values) / len(values)}

        def higher_is_better(self):
            return {"acc": True}

    task = PublicMethodsTask()
    assert not hasattr(task, "multiple_input")
    assert all(
        task.get_config(key) is None
        for key in (
            "training_split",
            "validation_split",
            "test_split",
            "doc_to_decontamination_query",
        )
    )

    run_task_tests(task_dict={task.task_name: task})

    assert task.docs_read == {"training", "validation", "test"}
    assert task.queries == task.dataset["test"]


def test_configurable_task_decontamination_uses_text_fallback(task_dir):
    task_dict = make_manager(task_dir).load(["integrity_one"])["tasks"]
    task = task_dict["integrity_one"]
    task.set_config("should_decontaminate", True)
    task.set_fewshot_seed(17)
    assert task.config.doc_to_decontamination_query is None
    doc = task.test_docs()[0]
    assert task.doc_to_decontamination_query(doc) == task.doc_to_text(doc)

    run_task_tests(task_dict=task_dict)


def test_message_task_checks_use_evaluation_chat_formatting(task_dir):
    class MessageTask(ConfigurableTask):
        def doc_to_text(self, doc, *args):
            return [{"role": "user", "content": doc["question"]}]

        def fewshot_context(self, doc, **kwargs):
            # This is the public contract used by Unitxt's message-based tasks.
            assert kwargs["apply_chat_template"]
            return kwargs["chat_template"](self.doc_to_text(doc))

    class ChatRecordingLM(RecordingLM):
        tokenizer_name = "offline-test"

        def apply_chat_template(self, messages, **kwargs):
            return "chat: " + messages[0]["content"] + "\n"

    config = json.loads((task_dir / "integrity_one.yaml").read_text(encoding="utf-8"))
    task = MessageTask(config=config)
    model = ChatRecordingLM()
    evaluator.simple_evaluate(
        model=model,
        tasks=[task],
        task_manager=make_manager(task_dir),
        check_integrity=True,
        apply_chat_template=True,
        limit=1,
        bootstrap_iters=0,
        log_samples=False,
        torch_random_seed=None,
    )
    assert len(model.requests) == 2
    assert all(context.startswith("chat: ") for context, _ in model.requests)


def test_multiple_input_task_checks_construct_choice_contexts(task_dir):
    config = json.loads((task_dir / "integrity_one.yaml").read_text(encoding="utf-8"))
    # As in Winogrande, the prompt selects an index into the choice contexts.
    config["doc_to_text"] = 0
    task = ConfigurableTask(config=config)
    assert task.multiple_input == 2
    model = RecordingLM()
    evaluator.simple_evaluate(
        model=model,
        tasks=[task],
        task_manager=make_manager(task_dir),
        check_integrity=True,
        limit=1,
        bootstrap_iters=0,
        log_samples=False,
        torch_random_seed=None,
    )
    assert len(model.requests) == 2
    assert {context for context, _ in model.requests} == {"alpha", "beta"}
