"""Проверки методов распознавания.

Запуск из папки проекта: python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recognition.data import Sample, default_task
from recognition.methods import (build_min_distance, distance, recognize, scale_task,
                                 train_perceptron, training_accuracy)


class TestScaling(unittest.TestCase):
    """Приведение признаков к безразмерному виду."""

    def setUp(self) -> None:
        self.task = default_task()
        self.scaled = scale_task(self.task)

    def test_values_between_zero_and_one(self):
        """Все безразмерные признаки лежат в диапазоне от 0 до 1."""
        for row in self.scaled.values:
            self.assertTrue(all(0.0 <= value <= 1.0 for value in row))

    def test_maximum_becomes_one(self):
        """По каждому признаку наибольшее значение превращается в единицу."""
        for column in zip(*self.scaled.values):
            self.assertAlmostEqual(max(column), 1.0, places=9)

    def test_divisors_are_maximums(self):
        """Делителями служат наибольшие значения признаков."""
        for index, column in enumerate(zip(*self.task.matrix())):
            self.assertAlmostEqual(self.scaled.maximums[index], max(column), places=9)


class TestMinDistance(unittest.TestCase):
    """Решающие функции на основе минимального расстояния."""

    def setUp(self) -> None:
        self.task = default_task()
        self.scaled = scale_task(self.task)
        self.model = build_min_distance(self.scaled)

    def test_prototype_is_class_average(self):
        """Прототип класса равен среднему значению признаков объектов этого класса."""
        for class_index in range(len(self.task.class_names)):
            rows = [row for row, sample in zip(self.scaled.values, self.task.samples)
                    if sample.class_index == class_index]
            manual = [sum(column) / len(column) for column in zip(*rows)]
            for expected, actual in zip(manual, self.model.prototypes[class_index]):
                self.assertAlmostEqual(expected, actual, places=9)

    def test_function_matches_formula(self):
        """Значение решающей функции считается по формуле из методических указаний."""
        point = self.scaled.values[0]
        for class_index, prototype in enumerate(self.model.prototypes):
            manual = 2 * sum(x * p for x, p in zip(point, prototype)) - sum(p * p for p in prototype)
            self.assertAlmostEqual(self.model.values(point)[class_index], manual, places=9)

    def test_largest_function_means_nearest_prototype(self):
        """Наибольшее значение функции соответствует наименьшему расстоянию до прототипа."""
        for row in self.scaled.values:
            scores = self.model.values(row)
            distances = [distance(row, prototype) for prototype in self.model.prototypes]
            self.assertEqual(scores.index(max(scores)), distances.index(min(distances)))

    def test_training_set_recognized(self):
        """Все объекты обучающего множества распознаются правильно."""
        for row, sample in zip(self.scaled.values, self.task.samples):
            self.assertEqual(self.model.classify(row), sample.class_index, sample.name)


class TestPerceptron(unittest.TestCase):
    """Алгоритм восприятия."""

    def setUp(self) -> None:
        self.task = default_task()
        self.scaled = scale_task(self.task)
        self.model = train_perceptron(self.scaled)

    def test_converged(self):
        """Обучение завершилось построением функции."""
        self.assertTrue(self.model.converged)
        self.assertGreater(self.model.corrections, 0)

    def test_all_objects_recognized(self):
        """Готовая функция правильно распознаёт всё обучающее множество."""
        for row, sample in zip(self.scaled.values, self.task.samples):
            value = self.model.value(row)
            if sample.class_index == 0:
                self.assertGreater(value, 0, sample.name)
            else:
                self.assertLess(value, 0, sample.name)

    def test_steps_follow_the_rule(self):
        """Поправка меняет коэффициенты ровно на значения признаков объекта."""
        previous = [0.0] * len(self.task.features)
        previous_bias = 0.0
        for step in self.model.steps:
            row = self.scaled.values[[s.name for s in self.task.samples].index(step.sample_name)]
            if step.correct:
                self.assertEqual(step.weights, previous)
                self.assertEqual(step.bias, previous_bias)
            else:
                sign = 1.0 if step.class_index == 0 else -1.0
                expected = [w + sign * x for w, x in zip(previous, row)]
                for a, b in zip(expected, step.weights):
                    self.assertAlmostEqual(a, b, places=9)
                self.assertAlmostEqual(step.bias, previous_bias + sign, places=9)
            previous, previous_bias = step.weights, step.bias

    def test_counter_resets_after_correction(self):
        """После поправки счётчик правильных ответов сбрасывается."""
        for step in self.model.steps:
            if not step.correct:
                self.assertEqual(step.counter, 0)

    def test_first_correction_equals_first_object(self):
        """Первая поправка превращает нулевые коэффициенты в признаки первого объекта."""
        first = self.model.steps[0]
        self.assertFalse(first.correct)
        for value, weight in zip(self.scaled.values[0], first.weights):
            self.assertAlmostEqual(value, weight, places=9)
        self.assertAlmostEqual(first.bias, 1.0, places=9)


class TestRecognition(unittest.TestCase):
    """Распознавание новых объектов."""

    def setUp(self) -> None:
        self.task = default_task()
        self.scaled = scale_task(self.task)
        self.model = build_min_distance(self.scaled)
        self.perceptron = train_perceptron(self.scaled)

    def test_both_methods_agree_on_examples(self):
        """На трёх примерах из задания оба метода дают одинаковый ответ."""
        for sample in self.task.unknown:
            result = recognize(self.scaled, self.model, self.perceptron, sample)
            self.assertTrue(result.agree, sample.name)

    def test_small_warehouse_is_first_class(self):
        """Небольшой склад относится к проекту ТП1."""
        result = recognize(self.scaled, self.model, self.perceptron,
                           Sample("маленький", [3, 250, 0.9, 4, 7], -1))
        self.assertEqual(result.distance_class, 0)
        self.assertEqual(result.perceptron_class, 0)

    def test_large_warehouse_is_second_class(self):
        """Крупный склад относится к проекту ТП2."""
        result = recognize(self.scaled, self.model, self.perceptron,
                           Sample("большой", [31, 780, 4.4, 19, 40], -1))
        self.assertEqual(result.distance_class, 1)
        self.assertEqual(result.perceptron_class, 1)

    def test_accuracy_on_training_set(self):
        """Оба метода распознают всё обучающее множество."""
        by_distance, by_perceptron = training_accuracy(self.scaled, self.model, self.perceptron)
        self.assertEqual(by_distance, len(self.task.samples))
        self.assertEqual(by_perceptron, len(self.task.samples))


if __name__ == "__main__":
    unittest.main()
