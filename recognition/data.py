"""Обучающее множество и объекты, подлежащие распознаванию.

Предметная область взята из методических указаний и расширена: экспертная система
выбирает типовой проект склада. Обучающее множество — двенадцать складов, которые
уже построены и успешно работают: шесть по проекту ТП1 и шесть по проекту ТП2.
Каждый склад описывается пятью признаками.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Feature:
    """Признак объекта."""

    name: str
    unit: str
    short: str

    @property
    def title(self) -> str:
        return f"{self.name}, {self.unit}" if self.unit else self.name


@dataclass
class Sample:
    """Объект обучающего множества: значения признаков и известный класс."""

    name: str
    values: list[float]
    class_index: int


@dataclass
class Task:
    """Условие задачи целиком."""

    title: str
    features: list[Feature]
    class_names: list[str]
    samples: list[Sample]
    unknown: list[Sample] = field(default_factory=list)

    def matrix(self) -> list[list[float]]:
        """Значения признаков обучающего множества: строка — объект."""
        return [sample.values for sample in self.samples]

    def by_class(self, class_index: int) -> list[Sample]:
        return [sample for sample in self.samples if sample.class_index == class_index]


def default_task() -> Task:
    """Задача выбора типового проекта склада: 12 объектов, 5 признаков, два класса."""
    features = [
        Feature("Грузооборот", "тыс. тонн в год", "X1"),
        Feature("Запас единовременного хранения", "тонн", "X2"),
        Feature("Площадь участка", "тыс. м²", "X3"),
        Feature("Отгрузок в сутки", "шт.", "X4"),
        Feature("Доля крупногабаритных грузов", "%", "X5"),
    ]
    samples = [
        Sample("Склад 1", [4, 400, 1.2, 6, 10], 0),
        Sample("Склад 2", [6, 500, 1.5, 8, 15], 0),
        Sample("Склад 3", [7, 300, 1.0, 7, 12], 0),
        Sample("Склад 4", [5, 450, 1.3, 5, 8], 0),
        Sample("Склад 5", [9, 550, 1.8, 9, 18], 0),
        Sample("Склад 6", [8, 380, 1.4, 6, 14], 0),
        Sample("Склад 7", [23, 620, 3.4, 16, 35], 1),
        Sample("Склад 8", [32, 570, 4.0, 20, 42], 1),
        Sample("Склад 9", [30, 800, 4.6, 18, 38], 1),
        Sample("Склад 10", [26, 700, 3.8, 15, 30], 1),
        Sample("Склад 11", [28, 640, 3.6, 17, 45], 1),
        Sample("Склад 12", [21, 750, 3.2, 14, 28], 1),
    ]
    unknown = [
        Sample("Проект А", [12, 200, 1.8, 9, 16], -1),
        Sample("Проект Б", [27, 690, 3.9, 16, 33], -1),
        Sample("Проект В", [17, 520, 2.6, 12, 25], -1),
    ]
    return Task(
        title="Выбор типового проекта склада по пяти признакам",
        features=features,
        class_names=["ТП1 — небольшой склад", "ТП2 — крупный склад"],
        samples=samples,
        unknown=unknown,
    )
