"""Лабораторная работа № 3: методы распознавания."""

from .data import Task, default_task
from .methods import (build_min_distance, complexity_note, recognize, scale_task,
                      train_perceptron, training_accuracy)

__all__ = [
    "Task",
    "default_task",
    "scale_task",
    "build_min_distance",
    "train_perceptron",
    "recognize",
    "training_accuracy",
    "complexity_note",
]
