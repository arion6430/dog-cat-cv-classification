# Классификатор кошек и собак

End-to-end ML-проект компьютерного зрения: дообученный **ResNet18** на датасете
Microsoft Cats vs Dogs, упакованный в веб-приложение **Streamlit** и
контейнеризированный с **Docker**. Сделан как portfolio-работа,
демонстрирующая полный ML-цикл — от подготовки данных через transfer learning
и оценку модели до деплоя.

## Демо

![Streamlit-интерфейс](reports/figures/confusion_matrix.png)

Загружаешь изображение через UI → модель возвращает предсказанный класс
(`cat` / `dog`), уверенность и bar-chart вероятностей.

## Быстрый старт

### 1. Установка зависимостей

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
```

### 2. Подготовка данных

Скачай датасет [Microsoft Cats vs Dogs](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset)
с Kaggle и распакуй так, чтобы получилась структура:

```
data/raw/PetImages/
├── Cat/
│   ├── 0.jpg
│   └── ...
└── Dog/
    ├── 0.jpg
    └── ...
```

Или через Kaggle CLI:

```bash
kaggle datasets download -d shaunthesheep/microsoft-catsvsdogs-dataset -p data/raw --unzip
```

### 3. Обучение

```bash
python -m src.train
python -m src.train --epochs 10 --finetune-after 2   # с дообучением backbone
```

После тренировки появятся `models/best_model.pt` и
`reports/figures/training_curves.png`.

### 4. Оценка

```bash
python -m src.evaluate
```

Сохраняет `reports/metrics.json` и `reports/figures/confusion_matrix.png`.

### 5. Веб-приложение

```bash
streamlit run app/streamlit_app.py
```

Открой <http://localhost:8501>.

### 6. Docker

```bash
docker build -t cat-dog-classifier .
docker run --rm -p 8501:8501 cat-dog-classifier
```

## Данные

- **Источник:** Microsoft Cats vs Dogs (≈25 000 размеченных изображений, 12 500
  кошек + 12 500 собак).
- **Сплиты:** 80% train / 10% val / 10% test, shuffle с фиксированным seed →
  разбиение воспроизводимо (`data/processed/splits.json`).
- Повреждённые JPEG фильтруются на этапе построения сплитов (полный декод в
  RGB, а не только `Image.verify()`).

## Модель

- **Backbone:** ResNet18, предобучен на ImageNet.
- **Голова:** один линейный слой `512 → 2`.
- **Рецепт обучения:** замораживаем backbone, обучаем только голову с
  Adam (lr=1e-3) одну эпоху, затем размораживаем последний блок (`layer4`)
  и дообучаем на lr=1e-4. Scheduler: `ReduceLROnPlateau` по val loss.
- **Пайплайн изображений:**
  - train: `RandomResizedCrop(224)`, `RandomHorizontalFlip`, `ColorJitter`,
    ImageNet-нормализация.
  - val/test: `Resize(256) → CenterCrop(224)`, та же нормализация.

**Почему ResNet18:** маленький (≈45 МБ), тренируется несколько минут на CPU и
даёт > 98% accuracy на этой задаче — этого хватает, чтобы продемонстрировать
полный пайплайн без тяжёлого железа.

## Результаты

Метрики на hold-out test-выборке (2 501 изображение, ≈10% датасета). Модель
обучалась на CPU одну полную эпоху (только голова), ResNet18 предобучен на
ImageNet.

| Метрика | Значение |
|---------|--------:|
| Accuracy | **0.9832** |
| Precision | 0.9936 |
| Recall | 0.9732 |
| F1 | 0.9833 |

Графики (`reports/figures/`):

- `confusion_matrix.png` — ошибки по классам на тесте.
- `training_curves.png` — loss/accuracy по эпохам (генерируется в `src.train`).

## Структура проекта

```
.
├── app/
│   └── streamlit_app.py      # веб-UI
├── src/
│   ├── config.py             # гиперпараметры, пути, device
│   ├── dataset.py            # CatDogDataset, сплиты, трансформации
│   ├── model.py              # build_model / load_model
│   ├── train.py              # цикл обучения
│   ├── evaluate.py           # метрики на test + confusion matrix
│   └── predict.py            # инференс-хелпер, общий для UI и CLI
├── data/                     # (в .gitignore) сырые + обработанные данные
├── models/                   # (в .gitignore) обученные веса
├── reports/                  # metrics.json, training_curves, confusion_matrix
├── notebooks/                # EDA и эксперименты
├── Dockerfile
├── requirements.txt
└── README.md
```

## Стек

- **Обучение:** PyTorch, torchvision
- **Метрики:** scikit-learn, matplotlib
- **Веб-UI:** Streamlit, Pillow
- **Деплой:** Docker (CPU-only образ на базе `python:3.11-slim`)

## Возможные улучшения

- Разморозить весь backbone для дополнительного выигрыша в accuracy.
- Попробовать более сильные модели: EfficientNet, ConvNeXt.
- Экспорт в ONNX / TorchScript для ускорения инференса на CPU.
- Grad-CAM-визуализация — показать, на какие области смотрит модель.
- Трекинг экспериментов: MLflow или Weights & Biases.
- Публичный деплой на HuggingFace Spaces.

## Лицензия

Код распространяется под MIT License (см. `LICENSE`). Датасет Microsoft Cats
vs Dogs имеет собственные условия использования от Microsoft Research.
