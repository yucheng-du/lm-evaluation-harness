"""Task integrity checks shared by the runtime entry point and repository CI."""

from __future__ import annotations

from itertools import islice
from typing import TYPE_CHECKING

import pytest

from lm_eval.api.instance import Instance


if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


@pytest.fixture()
def limit() -> int:
    return 10


@pytest.fixture()
def request_kwargs() -> dict:
    """CI defaults; the runtime runner supplies evaluation formatting options."""
    return {}


class BaseTasks:
    """
    Base class for testing tasks
    """

    def test_download(self, task_class: ConfigurableTask):
        # Initialization already loads the configured dataset. Reloading it here
        # would leave the few-shot sampler and cached docs tied to the old data.
        assert task_class.dataset is not None

    def test_has_training_docs(self, task_class: ConfigurableTask):
        assert task_class.has_training_docs() in [True, False]

    def test_check_training_docs(self, task_class: ConfigurableTask):
        if task_class.has_training_docs():
            assert task_class.training_docs() is not None

    def test_has_validation_docs(self, task_class):
        assert task_class.has_validation_docs() in [True, False]

    def test_check_validation_docs(self, task_class):
        if task_class.has_validation_docs():
            assert task_class.validation_docs() is not None

    def test_has_test_docs(self, task_class):
        assert task_class.has_test_docs() in [True, False]

    def test_check_test_docs(self, task_class):
        task = task_class
        if task.has_test_docs():
            assert task.test_docs() is not None

    def test_should_decontaminate(self, task_class, limit):
        task = task_class
        assert task.should_decontaminate() in [True, False]
        if task.should_decontaminate():
            for doc in islice(task.eval_docs, limit):
                assert isinstance(task.doc_to_decontamination_query(doc), str)

    def test_doc_to_text(self, task_class, limit, request_kwargs):
        task = task_class
        arr = (
            list(islice(task.test_docs(), limit))
            if task.has_test_docs()
            else list(islice(task.validation_docs(), limit))
        )
        _array = [task.doc_to_text(doc) for doc in arr]
        # space convention; allow txt to have length 0 for perplexity-like tasks since the model tacks an <|endoftext|> on
        target_delimiter: str = task.config.target_delimiter
        if not getattr(task, "multiple_input", False):
            for x in _array:
                # Some Python tasks provide messages directly (e.g. Unitxt).
                if isinstance(x, list) and request_kwargs.get("apply_chat_template"):
                    assert isinstance(request_kwargs["chat_template"](x), str)
                    continue
                assert isinstance(x, str)
                assert (
                    (x[-1].isspace() is False if len(x) > 0 else True)
                    if target_delimiter.isspace()
                    else True
                ), (
                    "doc_to_text ends in a whitespace and target delimiter also a whitespace"
                )
        else:
            pass

    def test_create_choices(self, task_class, limit):
        task = task_class
        arr = (
            list(islice(task.test_docs(), limit))
            if task.has_test_docs()
            else list(islice(task.validation_docs(), limit))
        )
        if "multiple_choice" in task._config.output_type:
            _array = [task.doc_to_choice(doc) for doc in arr]
            assert all(isinstance(x, list) for x in _array)
            assert all(isinstance(x[0], str) for x in _array)

    def test_doc_to_target(self, task_class, limit):
        task = task_class
        arr = (
            list(islice(task.test_docs(), limit))
            if task.has_test_docs()
            else list(islice(task.validation_docs(), limit))
        )
        _array_target = [task.doc_to_target(doc) for doc in arr]
        if task._config.output_type == "multiple_choice":
            # TODO<baber>: label can be string or int; add better test conditions
            for label in _array_target:
                if isinstance(label, list):
                    # tasks with multiple gold answers (e.g. webqs) return a list of indices
                    assert all(isinstance(x, int) for x in label)
                else:
                    assert isinstance(label, (int, str))

    def test_build_all_requests(self, task_class, limit, request_kwargs):
        task_class.build_all_requests(
            rank=0, limit=limit, world_size=1, **request_kwargs
        )
        assert task_class.instances is not None

    def test_construct_requests(self, task_class, limit, request_kwargs):
        task = task_class
        arr = (
            list(islice(task.test_docs(), limit))
            if task.has_test_docs()
            else list(islice(task.validation_docs(), limit))
        )
        for doc in arr:
            # Use the task's formatter: multiple-input tasks need one context
            # per choice, while message-based tasks need the chat template.
            ctx = task.fewshot_context(
                doc,
                num_fewshot=0,
                gen_prefix=task.doc_to_prefix(doc),
                **{
                    key: value
                    for key, value in request_kwargs.items()
                    if key != "tokenizer_name"
                },
            )
            requests = task.construct_requests(
                doc=doc,
                ctx=ctx,
                **{
                    key: request_kwargs[key]
                    for key in ("apply_chat_template", "chat_template")
                    if key in request_kwargs
                },
            )
            if not isinstance(requests, list):
                requests = [requests]
            assert requests
            assert all(isinstance(request, Instance) for request in requests)


class TestTaskIntegrity(BaseTasks):
    """Collect integrity checks for the tasks supplied by the runtime runner."""
