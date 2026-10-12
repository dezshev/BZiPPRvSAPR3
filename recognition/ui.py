"""Окно приложения: обучающее множество, два метода распознавания и разбор результатов."""

from __future__ import annotations

import os
import sys
import time

# PySide6 подключается раньше matplotlib, чтобы графики использовали именно его.
os.environ.setdefault("QT_API", "pyside6")
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QMainWindow, QPushButton, QTabWidget,
    QTableWidget, QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget,
)

from .data import Sample, Task, default_task
from .methods import (MinDistanceModel, PerceptronModel, Scaled, build_min_distance,
                      complexity_note, recognize, scale_task, train_perceptron, training_accuracy)

ACCENT = "#1f6feb"
DARK = "#16233a"
LIGHT = "#f4f7fb"
GOOD = "#1a7f37"
WARN = "#b54708"

STYLE = f"""
QMainWindow, QWidget#Root {{ background: {LIGHT}; }}
QLabel#Title {{ color: white; font-size: 19px; font-weight: 600; }}
QLabel#Subtitle {{ color: #c7d6ee; font-size: 12px; }}
QFrame#Header {{ background: {DARK}; border: none; }}
QGroupBox {{
    background: white; border: 1px solid #d7e0ee; border-radius: 10px;
    margin-top: 14px; padding: 10px 8px 8px 8px; font-weight: 600; color: #223;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; }}
QPushButton {{
    background: {ACCENT}; color: white; border: none; border-radius: 8px;
    padding: 9px 14px; font-weight: 600;
}}
QPushButton:hover {{ background: #3b82f6; }}
QPushButton#Secondary {{ background: white; color: {ACCENT}; border: 1px solid {ACCENT}; }}
QPushButton#Secondary:hover {{ background: #eaf2ff; }}
QTableWidget {{
    background: white; border: 1px solid #d7e0ee; border-radius: 8px;
    gridline-color: #e6ecf5; selection-background-color: #dbeafe; selection-color: #16233a;
}}
QHeaderView::section {{
    background: #eef3fa; color: #223; border: none; border-right: 1px solid #dfe7f2;
    border-bottom: 1px solid #dfe7f2; padding: 7px; font-weight: 600;
}}
QTabWidget::pane {{ border: 1px solid #d7e0ee; border-radius: 10px; background: white; top: -1px; }}
QTabBar::tab {{
    background: transparent; color: #4a5a72; padding: 9px 16px; margin-right: 4px;
    border-top-left-radius: 8px; border-top-right-radius: 8px; font-weight: 600;
}}
QTabBar::tab:selected {{ background: white; color: {ACCENT}; border: 1px solid #d7e0ee; border-bottom: 1px solid white; }}
QFrame#Card {{ background: white; border: 1px solid #d7e0ee; border-radius: 10px; }}
QLabel#CardValue {{ font-size: 20px; font-weight: 700; color: {DARK}; }}
QLabel#CardName {{ color: #5b6b83; font-size: 12px; }}
QLabel#CardHint {{ color: {GOOD}; font-size: 12px; font-weight: 600; }}
QTextBrowser {{ background: white; border: 1px solid #d7e0ee; border-radius: 8px; padding: 10px; }}
QStatusBar {{ background: white; color: #4a5a72; }}
"""


def number(value: float, digits: int = 3) -> str:
    """Число с запятой в качестве десятичного разделителя."""
    return f"{value:.{digits}f}".replace(".", ",")


def card(name: str, value: str = "—", hint: str = "") -> tuple[QFrame, QLabel, QLabel]:
    """Карточка с крупным значением."""
    frame = QFrame()
    frame.setObjectName("Card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 10, 14, 10)
    title = QLabel(name)
    title.setObjectName("CardName")
    value_label = QLabel(value)
    value_label.setObjectName("CardValue")
    value_label.setWordWrap(True)
    hint_label = QLabel(hint)
    hint_label.setObjectName("CardHint")
    hint_label.setWordWrap(True)
    layout.addWidget(title)
    layout.addWidget(value_label)
    layout.addWidget(hint_label)
    return frame, value_label, hint_label


def table(headers: list[str], rows: int = 0, editable: bool = False) -> QTableWidget:
    widget = QTableWidget(rows, len(headers))
    widget.setHorizontalHeaderLabels(headers)
    widget.verticalHeader().setVisible(False)
    widget.setAlternatingRowColors(True)
    if not editable:
        widget.setEditTriggers(QTableWidget.NoEditTriggers)
    widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    return widget


def item(text: str, bold: bool = False, color: str | None = None) -> QTableWidgetItem:
    cell = QTableWidgetItem(text)
    cell.setTextAlignment(Qt.AlignCenter)
    if bold:
        font = cell.font()
        font.setBold(True)
        cell.setFont(font)
    if color:
        cell.setBackground(QColor(color))
    return cell


class MainWindow(QMainWindow):
    """Главное окно лабораторной работы."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Методы распознавания: минимальное расстояние и алгоритм восприятия")
        self.resize(1380, 880)
        self.task: Task = default_task()
        self._build()
        self.recalculate()

    # ---------- интерфейс ----------

    def _build(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QFrame()
        header.setObjectName("Header")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(20, 12, 20, 12)
        header_layout.setSpacing(2)
        title = QLabel("Лабораторная работа № 3. Методы распознавания")
        title.setObjectName("Title")
        subtitle = QLabel(
            "Экспертная система выбирает типовой проект склада по пяти признакам. "
            "Используются решающие функции на основе минимального расстояния и алгоритм восприятия."
        )
        subtitle.setObjectName("Subtitle")
        subtitle.setWordWrap(True)
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        outer.addWidget(header)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(16, 12, 16, 10)
        layout.setSpacing(12)

        cards = QHBoxLayout()
        cards.setSpacing(12)
        self.card_samples, self.card_samples_value, self.card_samples_hint = card("Обучающее множество")
        self.card_distance, self.card_distance_value, self.card_distance_hint = card("Минимальное расстояние")
        self.card_perceptron, self.card_perceptron_value, self.card_perceptron_hint = card("Алгоритм восприятия")
        self.card_agreement, self.card_agreement_value, self.card_agreement_hint = card("Новые объекты")
        for widget in (self.card_samples, self.card_distance, self.card_perceptron, self.card_agreement):
            cards.addWidget(widget)
        layout.addLayout(cards)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_samples(), "Обучающее множество")
        self.tabs.addTab(self._tab_scaled(), "Безразмерные признаки")
        self.tabs.addTab(self._tab_distance(), "Минимальное расстояние")
        self.tabs.addTab(self._tab_perceptron(), "Алгоритм восприятия")
        self.tabs.addTab(self._tab_recognize(), "Распознавание")
        self.tabs.addTab(self._tab_plot(), "Геометрия")
        self.tabs.addTab(self._tab_complexity(), "Сложность")
        self.tabs.addTab(self._tab_help(), "Как считается")
        layout.addWidget(self.tabs, 1)
        outer.addWidget(body, 1)
        self.statusBar().showMessage("Готово")

    def _tab_samples(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        headers = ["Объект", "Класс"] + [feature.title.replace(", ", ",\n").replace(
            "единовременного хранения", "единовременного\nхранения") for feature in self.task.features]
        self.samples_table = table(headers, len(self.task.samples), editable=True)
        self.samples_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.samples_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.samples_widgets: list[list[QDoubleSpinBox]] = []
        for row, sample in enumerate(self.task.samples):
            self.samples_table.setItem(row, 0, item(sample.name, bold=True))
            self.samples_table.setItem(row, 1, item(self.task.class_names[sample.class_index].split(" —")[0],
                                                    color="#fff2cc" if sample.class_index else "#e6f4ea"))
            editors = []
            for column, value in enumerate(sample.values, start=2):
                editor = QDoubleSpinBox(maximum=1_000_000.0, decimals=2, singleStep=1.0)
                editor.setValue(float(value))
                editor.valueChanged.connect(self.recalculate)
                editor.setStyleSheet("padding: 3px;")
                self.samples_table.setCellWidget(row, column, editor)
                editors.append(editor)
            self.samples_widgets.append(editors)
        layout.addWidget(self.samples_table, 1)

        buttons = QHBoxLayout()
        reset = QPushButton("Вернуть исходные данные")
        reset.setObjectName("Secondary")
        reset.clicked.connect(self.reset_task)
        buttons.addWidget(reset)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        return widget

    def _tab_scaled(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        headers = ["Объект", "Класс"] + [feature.short for feature in self.task.features]
        self.scaled_table = table(headers, len(self.task.samples) + 1)
        layout.addWidget(self.scaled_table)
        note = QLabel(
            "Признаки измеряются в разных единицах, поэтому каждый столбец делится на своё "
            "наибольшее значение. После этого все признаки лежат в диапазоне от 0 до 1 и их можно "
            "сравнивать между собой. Последняя строка — делители."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #4a5a72;")
        layout.addWidget(note)
        return widget

    def _tab_distance(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(QLabel("Прототип класса — средний объект этого класса:"))
        headers = ["Класс"] + [feature.short for feature in self.task.features] + ["Свободный член"]
        self.prototype_table = table(headers, len(self.task.class_names))
        self.prototype_table.setFixedHeight(120)
        layout.addWidget(self.prototype_table)
        self.distance_formulas = QLabel()
        self.distance_formulas.setWordWrap(True)
        self.distance_formulas.setStyleSheet("font-size: 13px; color: #16233a; padding: 4px;")
        layout.addWidget(self.distance_formulas)
        layout.addWidget(QLabel("Проверка обучающего множества:"))
        self.distance_table = table(["Объект", "Класс", "D1", "D2", "Расстояние до ТП1",
                                     "Расстояние до ТП2", "Распознан как", "Верно?"],
                                    len(self.task.samples))
        layout.addWidget(self.distance_table, 1)
        return widget

    def _tab_perceptron(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        self.perceptron_summary = QLabel()
        self.perceptron_summary.setWordWrap(True)
        self.perceptron_summary.setStyleSheet("font-size: 13px; color: #16233a; padding: 4px;")
        layout.addWidget(self.perceptron_summary)
        layout.addWidget(QLabel("Ход обучения: каждая строка — подстановка очередного объекта в функцию"))
        headers = ["Шаг", "Объект", "Класс", "D12", "Результат", "Счётчик"] + \
                  [f"W{i + 1}" for i in range(len(self.task.features))] + ["W0"]
        self.steps_table = table(headers)
        layout.addWidget(self.steps_table, 1)
        return widget

    def _tab_recognize(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(QLabel("Объекты, подлежащие распознаванию:"))
        self.unknown_table = table(["Объект", "D1", "D2", "Расстояние до ТП1", "Расстояние до ТП2",
                                    "Минимальное\nрасстояние", "D12", "Алгоритм\nвосприятия", "Совпали?"],
                                   len(self.task.unknown))
        layout.addWidget(self.unknown_table)

        box = QGroupBox("Распознать свой объект")
        form = QFormLayout(box)
        self.custom_inputs: list[QDoubleSpinBox] = []
        for feature, sample in zip(self.task.features, [self.task.unknown[2].values]):
            pass
        for index, feature in enumerate(self.task.features):
            editor = QDoubleSpinBox(maximum=1_000_000.0, decimals=2, singleStep=1.0)
            editor.setValue(float(self.task.unknown[2].values[index]))
            editor.valueChanged.connect(self.update_custom)
            form.addRow(feature.title, editor)
            self.custom_inputs.append(editor)
        self.custom_result = QLabel()
        self.custom_result.setWordWrap(True)
        self.custom_result.setStyleSheet("font-size: 13px; color: #16233a;")
        form.addRow("Результат", self.custom_result)
        layout.addWidget(box)
        layout.addStretch(1)
        return widget

    def _tab_plot(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        controls = QHBoxLayout()
        self.axis_x = QComboBox()
        self.axis_y = QComboBox()
        for feature in self.task.features:
            self.axis_x.addItem(feature.name)
            self.axis_y.addItem(feature.name)
        self.axis_x.setCurrentIndex(0)
        self.axis_y.setCurrentIndex(1)
        self.axis_x.currentIndexChanged.connect(self.draw_plot)
        self.axis_y.currentIndexChanged.connect(self.draw_plot)
        controls.addWidget(QLabel("По горизонтали:"))
        controls.addWidget(self.axis_x, 1)
        controls.addWidget(QLabel("По вертикали:"))
        controls.addWidget(self.axis_y, 1)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 1)
        note = QLabel("Точки — объекты обучающего множества, квадраты — прототипы классов, "
                      "звёздочки — распознаваемые объекты. Линия — граница классов из алгоритма "
                      "восприятия на плоскости двух выбранных признаков: остальные три признака "
                      "взяты средними, поэтому для конкретного объекта граница проходит иначе — "
                      "решение принимается по всем пяти признакам сразу.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #4a5a72;")
        layout.addWidget(note)
        return widget

    def _tab_complexity(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        self.complexity_table = table(["Этап", "Оценка сложности", "Пояснение"])
        self.complexity_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        layout.addWidget(self.complexity_table)
        self.timing = QLabel()
        self.timing.setWordWrap(True)
        self.timing.setStyleSheet("color: #4a5a72;")
        layout.addWidget(self.timing)
        return widget

    def _tab_help(self) -> QWidget:
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(12, 12, 12, 12)
        view = QTextBrowser()
        view.setHtml(HELP_HTML)
        layout.addWidget(view)
        return holder

    # ---------- расчёт ----------

    def collect_task(self) -> Task:
        """Сбор значений признаков из полей окна."""
        task = default_task()
        for row, editors in enumerate(self.samples_widgets):
            task.samples[row].values = [editor.value() for editor in editors]
        return task

    def reset_task(self) -> None:
        """Возврат к исходным данным."""
        original = default_task()
        for row, sample in enumerate(original.samples):
            for column, value in enumerate(sample.values):
                self.samples_widgets[row][column].setValue(float(value))
        for index, value in enumerate(original.unknown[2].values):
            self.custom_inputs[index].setValue(float(value))
        self.recalculate()

    def recalculate(self) -> None:
        """Полный пересчёт обоих методов."""
        start = time.perf_counter()
        self.task = self.collect_task()
        self.scaled = scale_task(self.task)
        self.model = build_min_distance(self.scaled)
        self.perceptron = train_perceptron(self.scaled)
        self.results = [recognize(self.scaled, self.model, self.perceptron, sample)
                        for sample in self.task.unknown]
        self.elapsed = time.perf_counter() - start

        self._fill_scaled()
        self._fill_distance()
        self._fill_perceptron()
        self._fill_unknown()
        self._fill_complexity()
        self.draw_plot()
        self.update_custom()
        self._fill_cards()

    def _fill_cards(self) -> None:
        by_distance, by_perceptron = training_accuracy(self.scaled, self.model, self.perceptron)
        total = len(self.task.samples)
        self.card_samples_value.setText(f"{total} объектов")
        self.card_samples_hint.setText(f"{len(self.task.features)} признаков, "
                                       f"{len(self.task.class_names)} класса")
        self.card_distance_value.setText(f"{by_distance} из {total}")
        self.card_distance_hint.setText("распознано верно")
        self.card_perceptron_value.setText(f"{by_perceptron} из {total}")
        self.card_perceptron_hint.setText(f"поправок при обучении: {self.perceptron.corrections}")
        agree = sum(1 for result in self.results if result.agree)
        self.card_agreement_value.setText(f"{agree} из {len(self.results)}")
        self.card_agreement_hint.setText("методы дали одинаковый ответ")

    def _fill_scaled(self) -> None:
        widget = self.scaled_table
        widget.setRowCount(len(self.task.samples) + 1)
        for row, (sample, values) in enumerate(zip(self.task.samples, self.scaled.values)):
            widget.setItem(row, 0, item(sample.name, bold=True))
            widget.setItem(row, 1, item(self.task.class_names[sample.class_index].split(" —")[0],
                                        color="#fff2cc" if sample.class_index else "#e6f4ea"))
            for column, value in enumerate(values, start=2):
                widget.setItem(row, column, item(number(value)))
        last = len(self.task.samples)
        widget.setItem(last, 0, item("Делитель (максимум)", bold=True))
        widget.setItem(last, 1, item(""))
        for column, value in enumerate(self.scaled.maximums, start=2):
            widget.setItem(last, column, item(f"{value:g}".replace(".", ","), bold=True, color="#eef3fa"))

    def _fill_distance(self) -> None:
        widget = self.prototype_table
        for row, name in enumerate(self.task.class_names):
            widget.setItem(row, 0, item(name.split(" —")[0], bold=True))
            for column, value in enumerate(self.model.prototypes[row], start=1):
                widget.setItem(row, column, item(number(value)))
            widget.setItem(row, len(self.task.features) + 1, item(number(self.model.constants[row])))
        formulas = [self.model.formula(index, self.task.features)
                    for index in range(len(self.task.class_names))]
        self.distance_formulas.setText(
            "<b>Решающие функции:</b><br>" + "<br>".join(formulas) +
            "<br>Объект относится к классу, у которого значение функции больше.")

        widget = self.distance_table
        widget.setRowCount(len(self.task.samples))
        for row, (sample, values) in enumerate(zip(self.task.samples, self.scaled.values)):
            scores = self.model.values(values)
            distances = self.model.distances(values)
            chosen = scores.index(max(scores))
            correct = chosen == sample.class_index
            cells = [sample.name, self.task.class_names[sample.class_index].split(" —")[0]]
            cells += [number(score) for score in scores]
            cells += [number(value) for value in distances]
            cells += [self.task.class_names[chosen].split(" —")[0], "да" if correct else "нет"]
            for column, text in enumerate(cells):
                widget.setItem(row, column, item(text, bold=column == 0,
                                                 color=None if correct else "#fde8e8"))

    def _fill_perceptron(self) -> None:
        model = self.perceptron
        status = ("функция построена" if model.converged
                  else "достигнут лимит проверок; сходимость не подтверждена")
        self.perceptron_summary.setText(
            f"<b>Итог обучения:</b> {status}. Проверок объектов: {len(model.steps)}, "
            f"поправок: {model.corrections}.<br>"
            f"<b>Решающая функция:</b> {model.formula(self.task.features)}<br>"
            "Положительное значение — первый класс, отрицательное — второй."
        )
        widget = self.steps_table
        steps = model.steps
        widget.setRowCount(len(steps))
        for row, step in enumerate(steps):
            cells = [str(step.number), step.sample_name,
                     self.task.class_names[step.class_index].split(" —")[0],
                     number(step.value), "верно" if step.correct else "поправка",
                     str(step.counter)]
            cells += [number(weight) for weight in step.weights] + [number(step.bias)]
            for column, text in enumerate(cells):
                widget.setItem(row, column, item(text, color=None if step.correct else "#fff4e5"))

    def _fill_unknown(self) -> None:
        widget = self.unknown_table
        widget.setRowCount(len(self.results))
        for row, result in enumerate(self.results):
            cells = [result.name]
            cells += [number(score) for score in result.distance_scores]
            cells += [number(value) for value in result.distances]
            cells += [self.task.class_names[result.distance_class].split(" —")[0],
                      number(result.perceptron_value),
                      self.task.class_names[result.perceptron_class].split(" —")[0],
                      "да" if result.agree else "нет"]
            for column, text in enumerate(cells):
                widget.setItem(row, column, item(text, bold=column == 0,
                                                 color="#e6f4ea" if result.agree else "#fde8e8"))

    def _fill_complexity(self) -> None:
        rows = complexity_note(self.task, self.perceptron)
        widget = self.complexity_table
        widget.setRowCount(len(rows))
        for row, (stage, order, note) in enumerate(rows):
            widget.setItem(row, 0, item(stage, bold=True))
            widget.setItem(row, 1, item(order))
            cell = QTableWidgetItem(note)
            cell.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            widget.setItem(row, 2, cell)
        widget.resizeRowsToContents()
        self.timing.setText(
            f"Весь расчёт (приведение признаков, прототипы, обучение и распознавание) занял "
            f"{self.elapsed * 1000:.1f} мс".replace(".", ",") +
            ". При фиксированных числе классов и числе признаков время распознавания "
            "не зависит от числа обучающих объектов."
        )

    def draw_plot(self) -> None:
        """Геометрическая интерпретация: объекты, прототипы и граница классов."""
        if not hasattr(self, "scaled"):
            return
        x_index = self.axis_x.currentIndex()
        y_index = self.axis_y.currentIndex()
        self.figure.clear()
        axes = self.figure.add_subplot(111)
        colors = [GOOD, WARN]
        for class_index, name in enumerate(self.task.class_names):
            points = [row for row, sample in zip(self.scaled.values, self.task.samples)
                      if sample.class_index == class_index]
            axes.scatter([point[x_index] for point in points], [point[y_index] for point in points],
                         s=90, color=colors[class_index], alpha=0.75,
                         label=name.split(" —")[0], edgecolors="white")
            prototype = self.model.prototypes[class_index]
            axes.scatter([prototype[x_index]], [prototype[y_index]], marker="s", s=180,
                         color=colors[class_index], edgecolors=DARK, linewidths=1.4,
                         label=f"прототип {name.split(' —')[0]}")
        for result in self.results:
            axes.scatter([result.point[x_index]], [result.point[y_index]], marker="*", s=260,
                         color=ACCENT, edgecolors="white", zorder=5)
            axes.annotate(result.name, (result.point[x_index], result.point[y_index]),
                          textcoords="offset points", xytext=(10, 8), fontsize=9, color=ACCENT)

        # граница классов: остальные признаки зафиксированы на средних значениях
        weights = self.perceptron.weights
        if x_index != y_index and abs(weights[y_index]) > 1e-9:
            means = [sum(column) / len(column) for column in zip(*self.scaled.values)]
            rest = self.perceptron.bias + sum(
                weights[i] * means[i] for i in range(len(weights)) if i not in (x_index, y_index))
            xs = [0.0, 1.05]
            ys = [-(rest + weights[x_index] * x) / weights[y_index] for x in xs]
            axes.plot(xs, ys, color=ACCENT, linestyle="--", linewidth=1.8, label="граница при средних остальных признаках")
        axes.set_xlabel(self.task.features[x_index].name + " (безразмерный)")
        axes.set_ylabel(self.task.features[y_index].name + " (безразмерный)")
        axes.set_title("Проекция объектов на два признака; решение принимается по всем пяти", fontsize=11)
        axes.set_xlim(-0.05, 1.15)
        axes.set_ylim(-0.05, 1.15)
        axes.grid(alpha=0.3)
        axes.legend(fontsize=8, loc="lower right")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def update_custom(self) -> None:
        """Распознавание объекта, введённого вручную."""
        if not hasattr(self, "model"):
            return
        values = [editor.value() for editor in self.custom_inputs]
        result = recognize(self.scaled, self.model, self.perceptron, Sample("Свой объект", values, -1))
        first = self.task.class_names[result.distance_class]
        second = self.task.class_names[result.perceptron_class]
        agreement = ("оба метода согласны" if result.agree
                     else "методы дали разные классы; решение требует отдельного рассмотрения")
        self.custom_result.setText(
            f"Минимальное расстояние: <b>{first}</b> "
            f"(D1 = {number(result.distance_scores[0])}, D2 = {number(result.distance_scores[1])})<br>"
            f"Алгоритм восприятия: <b>{second}</b> (D12 = {number(result.perceptron_value)})<br>"
            f"{agreement}."
        )


HELP_HTML = """
<h3>Что делает программа</h3>
<p>Экспертная система выбирает типовой проект для нового склада. Обучающее множество —
двенадцать уже построенных складов: шесть по проекту ТП1 и шесть по проекту ТП2.
Каждый склад описан пятью признаками.</p>

<h3>Приведение признаков</h3>
<p>Грузооборот измеряется в тысячах тонн, запас хранения — в тоннах, площадь — в тысячах
квадратных метров. Чтобы признаки можно было сравнивать, каждый столбец делится на своё
наибольшее значение обучающего множества. Его неотрицательные значения лежат в диапазоне
от 0 до 1; новый объект может выйти за этот диапазон.</p>

<h3>Метод 1. Решающие функции на основе минимального расстояния</h3>
<p>Для каждого класса строится прототип — «средний» объект этого класса. Решающая функция:</p>
<p style="text-align:center"><b>Di = 2·(X1·Pi1 + … + XM·PiM) − (Pi1² + … + PiM²)</b></p>
<p>Объект относится к классу с наибольшим значением функции. Это равносильно выбору
ближайшего прототипа по обычному расстоянию, только считается быстрее.</p>

<h3>Метод 2. Алгоритм восприятия</h3>
<p>Строится одна разделяющая функция <b>D12 = W1·X1 + … + WM·XM + W0</b>. Коэффициенты
начинаются с нуля, затем объекты перебираются по кругу:</p>
<ul>
<li>объект первого класса даёт D12 &gt; 0, объект второго класса — D12 &lt; 0: функция не меняется,
счётчик правильных ответов растёт;</li>
<li>ошибка на объекте первого класса: признаки прибавляются к коэффициентам, W0 растёт на единицу,
счётчик сбрасывается;</li>
<li>ошибка на объекте второго класса: признаки вычитаются, W0 уменьшается на единицу,
счётчик сбрасывается.</li>
</ul>
<p>Обучение заканчивается, когда функция правильно распознаёт подряд все объекты
обучающего множества. Весь ход обучения показан на вкладке «Алгоритм восприятия».</p>

<h3>Геометрия</h3>
<p>Объекты — точки в пространстве пяти признаков; D12 = 0 задаёт гиперплоскость.
На графике показано её сечение при средних значениях остальных признаков. Решение использует все пять.</p>

"""


def run_application() -> int:
    """Запуск приложения."""
    application = QApplication(sys.argv)
    application.setStyleSheet(STYLE)
    application.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    return application.exec()
