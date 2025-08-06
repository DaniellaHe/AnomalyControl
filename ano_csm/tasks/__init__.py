"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from ano_csm.common.registry import registry
from ano_csm.tasks.base_task import BaseTask
from ano_csm.tasks.captioning import CaptionTask
from ano_csm.tasks.image_text_pretrain import ImageTextPretrainTask
from ano_csm.tasks.multimodal_classification import (
    MultimodalClassificationTask,
)
from ano_csm.tasks.retrieval import RetrievalTask
from ano_csm.tasks.vqa import VQATask, GQATask, AOKVQATask, DisCRNTask
from ano_csm.tasks.vqa_reading_comprehension import VQARCTask, GQARCTask
from ano_csm.tasks.dialogue import DialogueTask
from ano_csm.tasks.text_to_image_generation import TextToImageGenerationTask


def setup_task(cfg):
    assert "task" in cfg.run_cfg, "Task name must be provided."

    task_name = cfg.run_cfg.task
    task = registry.get_task_class(task_name).setup_task(cfg=cfg)
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task


__all__ = [
    "BaseTask",
    "AOKVQATask",
    "RetrievalTask",
    "CaptionTask",
    "VQATask",
    "GQATask",
    "VQARCTask",
    "GQARCTask",
    "MultimodalClassificationTask",
    # "VideoQATask",
    # "VisualEntailmentTask",
    "ImageTextPretrainTask",
    "DialogueTask",
    "TextToImageGenerationTask",
    "DisCRNTask"
]
