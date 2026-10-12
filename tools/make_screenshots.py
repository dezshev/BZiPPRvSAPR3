"""Снимки окна приложения для README.

Скрипт открывает окно, проходит по вкладкам и сохраняет их в docs/img/app_*.png.
Запуск: python tools/make_screenshots.py
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PySide6.QtWidgets import QApplication  # noqa: E402

from recognition.ui import STYLE, MainWindow  # noqa: E402

OUT = os.path.join(ROOT, "docs", "img")
TABS = ["app_samples", "app_scaled", "app_distance", "app_perceptron",
        "app_recognize", "app_plot", "app_complexity", "app_help"]


def wait(application: QApplication, seconds: float) -> None:
    """Пауза с обработкой событий окна."""
    end = time.time() + seconds
    while time.time() < end:
        application.processEvents()
        time.sleep(0.02)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    application = QApplication(sys.argv)
    application.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    wait(application, 1.2)

    for index, name in enumerate(TABS):
        window.tabs.setCurrentIndex(index)
        wait(application, 0.8)
        window.grab().save(os.path.join(OUT, f"{name}.png"))
        print("сохранено", f"{name}.png")

    window.tabs.setCurrentIndex(5)
    wait(application, 0.6)
    window.figure.savefig(os.path.join(OUT, "geometry.png"), dpi=150, facecolor="white")
    print("сохранён график geometry.png")
    window.close()


if __name__ == "__main__":
    main()
