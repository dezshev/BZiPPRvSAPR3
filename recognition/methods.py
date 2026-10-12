"""Методы распознавания: решающие функции на основе минимального расстояния
и построение разделяющей функции алгоритмом восприятия."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .data import Sample, Task

EPS = 1e-12


@dataclass
class Scaled:
    """Признаки, приведённые к безразмерному виду делением на наибольшее значение."""

    task: Task
    maximums: list[float]
    values: list[list[float]]       # строка — объект обучающего множества
    unknown: list[list[float]]      # строка — объект, подлежащий распознаванию

    def scale(self, values: list[float]) -> list[float]:
        """Приведение произвольного объекта к безразмерному виду."""
        return [value / top if top else 0.0 for value, top in zip(values, self.maximums)]


def scale_task(task: Task) -> Scaled:
    """Нормирование на максимумы обучающего множества.

    Признаки измеряются в разных единицах и отличаются по порядку величин,
    поэтому каждый столбец делится на своё наибольшее значение — так делается
    в примере методических указаний. Неотрицательные обучающие значения лежат
    в [0, 1]; значения новых объектов могут превышать единицу.
    """
    columns = list(zip(*task.matrix()))
    maximums = [max(abs(value) for value in column) for column in columns]
    values = [[value / top if top else 0.0 for value, top in zip(row, maximums)]
              for row in task.matrix()]
    unknown = [[value / top if top else 0.0 for value, top in zip(sample.values, maximums)]
               for sample in task.unknown]
    return Scaled(task=task, maximums=maximums, values=values, unknown=unknown)


def distance(first: list[float], second: list[float]) -> float:
    """Евклидово расстояние между объектами."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


@dataclass
class MinDistanceModel:
    """Решающие функции на основе минимального расстояния."""

    prototypes: list[list[float]]        # прототип каждого класса
    constants: list[float]               # свободный член каждой решающей функции
    class_names: list[str]

    def values(self, point: list[float]) -> list[float]:
        """Значения решающих функций Di для объекта."""
        return [2 * sum(x * p for x, p in zip(point, prototype)) - constant
                for prototype, constant in zip(self.prototypes, self.constants)]

    def classify(self, point: list[float]) -> int:
        """Номер класса: объект относится к классу с наибольшим значением функции."""
        scores = self.values(point)
        return scores.index(max(scores))

    def distances(self, point: list[float]) -> list[float]:
        """Расстояния от объекта до прототипов классов."""
        return [distance(point, prototype) for prototype in self.prototypes]

    def formula(self, index: int, features) -> str:
        """Решающая функция в виде строки, как записывают в отчёте."""
        parts = []
        for value, feature in zip(self.prototypes[index], features):
            parts.append(f"{2 * value:.3f}·{feature.short}".replace(".", ","))
        constant = f"{self.constants[index]:.3f}".replace(".", ",")
        return f"D{index + 1} = " + " + ".join(parts) + f" − {constant}"


def build_min_distance(scaled: Scaled) -> MinDistanceModel:
    """Построение прототипов классов и решающих функций.

    Прототип класса — «средний» объект: по каждому признаку берётся среднее
    значение признака у объектов этого класса.
    """
    task = scaled.task
    prototypes, constants = [], []
    for class_index in range(len(task.class_names)):
        rows = [row for row, sample in zip(scaled.values, task.samples)
                if sample.class_index == class_index]
        prototype = [sum(column) / len(column) for column in zip(*rows)]
        prototypes.append(prototype)
        constants.append(sum(value * value for value in prototype))
    return MinDistanceModel(prototypes=prototypes, constants=constants, class_names=task.class_names)


@dataclass
class PerceptronStep:
    """Один шаг обучения алгоритмом восприятия."""

    number: int
    sample_name: str
    class_index: int
    value: float
    correct: bool
    counter: int
    weights: list[float]
    bias: float


@dataclass
class PerceptronModel:
    """Разделяющая решающая функция, построенная алгоритмом восприятия."""

    weights: list[float]
    bias: float
    steps: list[PerceptronStep] = field(default_factory=list)
    corrections: int = 0
    converged: bool = True

    def value(self, point: list[float]) -> float:
        """Значение решающей функции D12 для объекта."""
        return sum(w * x for w, x in zip(self.weights, point)) + self.bias

    def classify(self, point: list[float]) -> int:
        """Положительное значение — первый класс, отрицательное — второй."""
        return 0 if self.value(point) > 0 else 1

    def formula(self, features) -> str:
        """Решающая функция в виде строки."""
        text = "D12 = "
        for position, (weight, feature) in enumerate(zip(self.weights, features)):
            number = f"{abs(weight):.3f}".replace(".", ",")
            if position == 0:
                text += ("−" if weight < 0 else "") + f"{number}·{feature.short}"
            else:
                text += (" − " if weight < 0 else " + ") + f"{number}·{feature.short}"
        number = f"{abs(self.bias):.3f}".replace(".", ",")
        text += (" − " if self.bias < 0 else " + ") + number
        return text


def train_perceptron(scaled: Scaled, limit: int = 10_000) -> PerceptronModel:
    """Алгоритм восприятия из методических указаний.

    Коэффициенты начинаются с нуля. Объекты перебираются по кругу: если объект
    первого класса даёт неположительное значение функции, признаки прибавляются
    к коэффициентам; если объект второго класса даёт неотрицательное значение —
    вычитаются. Счётчик правильных ответов сбрасывается после каждой поправки.
    Обучение заканчивается, когда функция правильно распознаёт все объекты подряд.
    """
    task = scaled.task
    count = len(scaled.values)
    weights = [0.0] * len(task.features)
    bias = 0.0
    steps: list[PerceptronStep] = []
    counter = 0
    corrections = 0
    index = 0
    number = 0

    while counter < count and number < limit:
        row = scaled.values[index]
        sample = task.samples[index]
        value = sum(w * x for w, x in zip(weights, row)) + bias
        number += 1
        correct = (value > 0) if sample.class_index == 0 else (value < 0)
        if correct:
            counter += 1
        else:
            corrections += 1
            counter = 0
            sign = 1.0 if sample.class_index == 0 else -1.0
            weights = [w + sign * x for w, x in zip(weights, row)]
            bias += sign
        steps.append(PerceptronStep(number, sample.name, sample.class_index, value,
                                    correct, counter, list(weights), bias))
        index = (index + 1) % count

    return PerceptronModel(weights=weights, bias=bias, steps=steps,
                           corrections=corrections, converged=counter >= count)


@dataclass
class Recognition:
    """Результат распознавания одного объекта двумя методами."""

    name: str
    raw: list[float]
    point: list[float]
    distance_scores: list[float]
    distances: list[float]
    distance_class: int
    perceptron_value: float
    perceptron_class: int

    @property
    def agree(self) -> bool:
        return self.distance_class == self.perceptron_class


def recognize(scaled: Scaled, distance_model: MinDistanceModel, perceptron: PerceptronModel,
              sample: Sample) -> Recognition:
    """Распознавание объекта обоими методами."""
    point = scaled.scale(sample.values)
    return Recognition(
        name=sample.name,
        raw=list(sample.values),
        point=point,
        distance_scores=distance_model.values(point),
        distances=distance_model.distances(point),
        distance_class=distance_model.classify(point),
        perceptron_value=perceptron.value(point),
        perceptron_class=perceptron.classify(point),
    )


def training_accuracy(scaled: Scaled, distance_model: MinDistanceModel,
                      perceptron: PerceptronModel) -> tuple[int, int]:
    """Сколько объектов обучающего множества каждый метод распознаёт правильно."""
    by_distance = sum(1 for row, sample in zip(scaled.values, scaled.task.samples)
                      if distance_model.classify(row) == sample.class_index)
    by_perceptron = sum(1 for row, sample in zip(scaled.values, scaled.task.samples)
                        if perceptron.classify(row) == sample.class_index)
    return by_distance, by_perceptron


def complexity_note(task: Task, perceptron: PerceptronModel) -> list[tuple[str, str, str]]:
    """Оценка вычислительной сложности алгоритмов: этап, порядок, пояснение."""
    n = len(task.samples)
    m = len(task.class_names)
    k = len(task.features)
    return [
        ("Приведение к безразмерному виду", "O(N·M)",
         f"поиск максимумов и нормирование {n * k} обучающих значений; не точное число операций"),
        ("Построение прототипов классов", "O(K·N + N·M)",
         "отбор объектов для каждого класса и усреднение признаков; при K=2 — O(N·M)"),
        ("Распознавание по минимальному расстоянию", "O(K·M)",
         f"{m} скалярных произведения по {k} признакам; N — объекты, M — признаки, K — классы"),
        ("Обучение алгоритмом восприятия", f"O(T·M), выполнено {len(perceptron.steps)} проверок",
         f"T — число просмотров объектов, потребовалось поправок: {perceptron.corrections}"),
        ("Распознавание по разделяющей функции", "O(M)",
         f"одно скалярное произведение: {k} умножений, суммирование и свободный член"),
    ]
