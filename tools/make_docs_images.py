"""Рисунки для README: сравнение двух методов и блок-схема алгоритма восприятия.

Запуск: python tools/make_docs_images.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from recognition.data import default_task
from recognition.methods import build_min_distance, scale_task, train_perceptron

OUT = os.path.join(ROOT, "docs", "img")
ACCENT = "#1f6feb"
DARK = "#16233a"
GREEN = "#1a7f37"
ORANGE = "#b54708"


def methods_scheme() -> None:
    """Два метода распознавания рядом: прототипы и разделяющая граница."""
    task = default_task()
    scaled = scale_task(task)
    model = build_min_distance(scaled)
    perceptron = train_perceptron(scaled)
    means = [sum(column) / len(column) for column in zip(*scaled.values)]

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    colors = [GREEN, ORANGE]
    for panel in axes:
        for class_index, name in enumerate(task.class_names):
            points = [row for row, sample in zip(scaled.values, task.samples)
                      if sample.class_index == class_index]
            panel.scatter([p[0] for p in points], [p[1] for p in points], s=90,
                          color=colors[class_index], alpha=0.75, edgecolors="white",
                          label=name.split(" —")[0])
        panel.set_xlabel("Грузооборот (безразмерный)")
        panel.set_ylabel("Запас хранения (безразмерный)")
        panel.set_xlim(-0.05, 1.1)
        panel.set_ylim(-0.05, 1.1)
        panel.grid(alpha=0.3)

    # слева: прототипы и расстояния
    for class_index in range(len(task.class_names)):
        prototype = model.prototypes[class_index]
        axes[0].scatter([prototype[0]], [prototype[1]], marker="s", s=200,
                        color=colors[class_index], edgecolors=DARK, linewidths=1.5)
        axes[0].annotate(f"прототип {task.class_names[class_index].split(' —')[0]}",
                         (prototype[0], prototype[1]), textcoords="offset points",
                         xytext=(10, 12), fontsize=9, color=DARK)
    point = scaled.unknown[2]
    axes[0].scatter([point[0]], [point[1]], marker="*", s=300, color=ACCENT, edgecolors="white", zorder=5)
    for class_index in range(len(task.class_names)):
        prototype = model.prototypes[class_index]
        axes[0].plot([point[0], prototype[0]], [point[1], prototype[1]],
                     color=colors[class_index], linestyle=":", linewidth=1.6,
                     label=f"расстояние по 5 признакам: {model.distances(point)[class_index]:.3f}")
    axes[0].set_title("Проекция на X1 и X2:\nрасстояния вычисляются по пяти признакам", fontsize=11)
    axes[0].legend(fontsize=8, loc="lower right")

    # справа: разделяющая граница
    weights = perceptron.weights
    rest = perceptron.bias + sum(weights[i] * means[i] for i in range(2, len(weights)))
    xs = [0.0, 1.1]
    ys = [-(rest + weights[0] * x) / weights[1] for x in xs]
    axes[1].plot(xs, ys, color=ACCENT, linewidth=2, label="D12 = 0")
    axes[1].fill_between(xs, ys, 1.1, color=GREEN, alpha=0.07)
    axes[1].fill_between(xs, -0.05, ys, color=ORANGE, alpha=0.07)
    axes[1].scatter([point[0]], [point[1]], marker="*", s=300, color=ACCENT, edgecolors="white", zorder=5)
    axes[1].set_title("Сечение D12 = 0:\nX3–X5 фиксированы на средних значениях", fontsize=11)
    axes[1].legend(fontsize=8, loc="lower right")

    figure.suptitle("Два способа распознавания на одних и тех же данных", fontsize=13,
                    color=DARK, fontweight="bold")
    figure.tight_layout()
    figure.savefig(os.path.join(OUT, "methods.png"), dpi=150, facecolor="white")
    plt.close(figure)


def _block(axes, x, y, w, h, text, color):
    axes.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                  linewidth=1.6, edgecolor=color, facecolor="white"))
    axes.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5, color=DARK)


def _arrow(axes, start, end, text="", color="#6b7c93", offset=(0.25, 0.0)):
    axes.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                   color=color, linewidth=1.3))
    if text:
        axes.text((start[0] + end[0]) / 2 + offset[0], (start[1] + end[1]) / 2 + offset[1], text,
                  fontsize=9, color=color, ha="center", va="center",
                  bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none"))


def perceptron_scheme() -> None:
    """Блок-схема алгоритма восприятия."""
    figure, axes = plt.subplots(figsize=(9.5, 8.6))
    _block(axes, 3.0, 9.4, 4.0, 0.8, "Все коэффициенты равны нулю,\nсчётчик Е = 0", ACCENT)
    _block(axes, 3.0, 8.1, 4.0, 0.7, "Взять очередной объект", ACCENT)
    _block(axes, 3.0, 6.8, 4.0, 0.7, "Подставить его в D12", ACCENT)
    _block(axes, 3.0, 5.3, 4.0, 0.9, "Объект распознан верно?\n(D12 > 0 для первого класса,\nD12 < 0 для второго)", ORANGE)
    _block(axes, 0.2, 3.6, 3.4, 0.9, "Да: функция не меняется,\nЕ = Е + 1", GREEN)
    _block(axes, 6.2, 3.6, 3.4, 0.9, "Нет: поправка коэффициентов,\nЕ = 0", "#b3261e")
    _block(axes, 6.2, 2.2, 3.4, 0.9, "первый класс: прибавить признаки\nвторой класс: вычесть признаки", "#b3261e")
    _block(axes, 3.0, 0.8, 4.0, 0.9, "Е равен числу объектов?\nДа — функция построена", DARK)

    _arrow(axes, (5.0, 9.4), (5.0, 8.8))
    _arrow(axes, (5.0, 8.1), (5.0, 7.5))
    _arrow(axes, (5.0, 6.8), (5.0, 6.2))
    _arrow(axes, (3.0, 5.75), (1.9, 5.75), "да", GREEN, offset=(0.0, 0.22))
    _arrow(axes, (1.9, 5.75), (1.9, 4.5))
    _arrow(axes, (7.0, 5.75), (7.9, 5.75), "нет", "#b3261e", offset=(0.0, 0.22))
    _arrow(axes, (7.9, 5.75), (7.9, 4.5))
    _arrow(axes, (7.9, 3.6), (7.9, 3.1))
    _arrow(axes, (1.9, 3.6), (1.9, 1.25))
    _arrow(axes, (1.9, 1.25), (3.0, 1.25))
    _arrow(axes, (7.9, 2.2), (7.9, 1.25))
    _arrow(axes, (7.9, 1.25), (7.0, 1.25))
    axes.add_patch(FancyArrowPatch((3.0, 1.0), (0.6, 1.0), arrowstyle="-", color="#6b7c93", linewidth=1.2))
    axes.add_patch(FancyArrowPatch((0.6, 1.0), (0.6, 8.45), arrowstyle="-", color="#6b7c93", linewidth=1.2))
    _arrow(axes, (0.6, 8.45), (3.0, 8.45), "нет — следующий объект", "#6b7c93", offset=(0.0, 0.25))

    axes.text(4.8, 10.6, "Алгоритм восприятия", ha="center", fontsize=13, color=DARK, fontweight="bold")
    axes.text(4.8, 10.3, "строит разделяющую функцию D12 = W1·X1 + … + WM·XM + W0",
              ha="center", fontsize=9.5, color="#4a5a72")
    axes.set_xlim(0, 9.8)
    axes.set_ylim(0.4, 11.0)
    axes.axis("off")
    figure.tight_layout()
    figure.savefig(os.path.join(OUT, "perceptron.png"), dpi=150, facecolor="white")
    plt.close(figure)


def training_progress() -> None:
    """Как менялось значение решающей функции в ходе обучения."""
    task = default_task()
    scaled = scale_task(task)
    perceptron = train_perceptron(scaled)
    steps = perceptron.steps
    numbers = [step.number for step in steps]
    values = [step.value for step in steps]
    colors = [GREEN if step.correct else "#b3261e" for step in steps]

    figure, axes = plt.subplots(figsize=(11, 4.4))
    axes.bar(numbers, values, color=colors, width=0.7)
    axes.axhline(0, color=DARK, linewidth=1.2)
    for step in steps:
        if not step.correct:
            axes.annotate("поправка", (step.number, step.value), textcoords="offset points",
                          xytext=(0, 8 if step.value > 0 else -16), ha="center", fontsize=7.5,
                          color="#b3261e")
    axes.set_xlabel("номер проверки объекта")
    axes.set_ylabel("значение D12")
    axes.set_title("Обучение алгоритмом восприятия: зелёные столбцы — объект распознан верно, "
                   "красные — потребовалась поправка", fontsize=11, color=DARK)
    axes.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(os.path.join(OUT, "training.png"), dpi=150, facecolor="white")
    plt.close(figure)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    methods_scheme()
    perceptron_scheme()
    training_progress()
    print("рисунки сохранены в", OUT)


if __name__ == "__main__":
    main()
