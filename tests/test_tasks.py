import os
from typing import cast

import datasets
import pytest

from lm_eval._task_testing import (  # noqa: F401 -- shared pytest fixtures
    BaseTasks,
    limit,
    request_kwargs,
)
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks import TaskManager

from .utils import new_tasks


datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Default Task
TASKS = ["arc_easy"]


def get_new_tasks_else_default():
    """
    Check if any modifications have been made to built-in tasks and return
    the list, otherwise return the default task list
    """
    # CI: new_tasks checks if any modifications have been made
    task_list = new_tasks()
    # Check if task_classes is empty
    return task_list or TASKS


def task_class(
    task_names: list[str], task_manager: TaskManager | None = None
) -> list[ConfigurableTask]:
    """
    Convert a list of task names to a list of ConfigurableTask instances
    """
    if task_manager is None:
        from lm_eval.tasks import TaskManager

        task_manager = TaskManager()

    res = task_manager.load(task_names)
    res = res["tasks"]

    return cast("list[ConfigurableTask]", res.values())


@pytest.mark.parametrize(
    "task_class",
    task_class(get_new_tasks_else_default()),
    ids=lambda x: f"{x.config.task}",
)
class TestNewTasksElseDefault(BaseTasks):
    """
    Test class parameterized with a list of new/modified tasks
    (or a set of default tasks if none have been modified)
    """


@pytest.mark.parametrize(
    "task_cls_path",
    [
        "lm_eval.tasks.fda.task:FDA",
        "lm_eval.tasks.swde.task:SWDE",
        "lm_eval.tasks.squad_completion.task:SQUADCompletion",
    ],
    ids=["fda", "swde", "squad_completion"],
)
def test_strip_whitespace_in_target_and_metric(task_cls_path):
    """Regression test for the FDA/SWDE/SQuAD_completion whitespace strip.

    doc_to_text and doc_to_target strip surrounding whitespace, and
    process_results must route the gold answer through doc_to_target so the
    `contains` metric compares against the stripped target. Otherwise an
    otherwise-correct prediction misses on the surrounding whitespace.
    """
    import importlib

    module_path, cls_name = task_cls_path.split(":")
    cls = getattr(importlib.import_module(module_path), cls_name)
    task = cls()

    doc = {"text": "  prompt  ", "value": "  answer  "}

    assert task.doc_to_text(doc) == "prompt"
    assert task.doc_to_target(doc) == "answer"
    # The prediction has no surrounding whitespace; with the unstripped target
    # this would be {"contains": 0}.
    assert task.process_results(doc, ["answer"]) == {"contains": 1}
