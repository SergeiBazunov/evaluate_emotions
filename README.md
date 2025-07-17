# Оценка моделей распознавания эмоций на наборе **CK+**

Этот репозиторий содержит утилиту `evaluate.py` для сравнения качества двух систем распознавания эмоций на наборе коротких видеороликов **CK+** (в формате MP4, директория `ck_videos/`):

1. **Local** – библиотека [DeepFace](https://github.com/serengil/deepface) (запускается локально, работает в среднем ~70 fps).
2. **Cloud** – модель **GPT-4o Vision** (интерфейс OpenAI API, точность выше, но латентность ~1 с).

Скрипт выводит стандартные метрики классификации (accuracy, macro-F1, confusion-matrix) и собирает статистику задержек.

---

## 1. Подготовка данных

1. Скачайте оригинальный датасет **CK+** и конвертируйте последовательности изображений в короткие ролики `.mp4` (уже выполнено в ветке `ck_videos/`).
2. Сгенерируйте CSV файл со списком путей и эмоций:

```bash
python make_ck_csv.py --videos_dir ck_videos --out_csv ck_paths.csv
```

Маппинг «`contempt` → `neutral`» выполняется автоматически.

---

## 2. Установка зависимостей

```bash
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r emotional_project/requirements.txt

# Для облачного режима добавьте openai
pip install openai tqdm scikit-learn pandas opencv-python
```

> ⚠️ Для запуска **local-режима** потребуется GPU-драйвер + `tensorflow` / `torch`, которые ставятся DeepFace-ом автоматически. Для **cloud-режима** достаточно CPU.

---

## 3. Быстрый старт

### 3.1 Локальный предиктор (DeepFace)

```bash
python evaluate.py \
  --mode local \
  --csv ck_paths.csv \
  --root ck_videos \
  --backend skip \
  --no_enforce \
  --no_preprocess \
  --max_per_class 0     # 0 = без ограничения, весь датасет
```

Средний результат: **66 % accuracy**, **macro-F1≈0.53**, латентность ~7 ms/кадр.

### 3.2 Облачный предиктор (GPT-4o Vision)

```bash
export OPENAI_API_KEY="sk-..."     # Windows PowerShell: $Env:OPENAI_API_KEY="sk-..."

python evaluate.py \
  --mode cloud \
  --api_key $OPENAI_API_KEY \
  --csv ck_paths.csv \
  --root ck_videos \
  --max_per_class 10 \
  --no_enforce --no_preprocess \
  --conf_thres 0.4
```

Результат на подсэмпле 10 × 7 роликов: **87 % accuracy**, **macro-F1≈0.88**, латентность ≈1.2 s (P90 1.7 s).

---

## 4. Аргументы командной строки

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--csv` | `ckextended.csv` | CSV с колонками `path,label`. |
| `--root` | `ck_videos` | Корневая папка с видео. |
| `--mode` | — | `local` или `cloud`. |
| `--api_key` | — | Ключ OpenAI для cloud-режима. |
| `--details` | `False` | Печатает предсказание каждой записи. |
| `--backend` | `retinaface` | Детектор лиц DeepFace (`retinaface|opencv|mtcnn|skip`). |
| `--no_enforce` | `False` | Не требовать наличие лица (`enforce_detection=False`). |
| `--list_emotions` | `False` | Вывести список эмоций DeepFace и выйти. |
| `--no_preprocess` | `False` | Пропустить upscale + RGB-конвертацию. |
| `--max_per_class` | `10` | Обрезать датасет (0 = без ограничения). |
| `--conf_thres` | `0.5` | Порог уверенности DeepFace. |
| `--exclude_neutral` | `False` | Не учитывать `neutral` в метриках. |

---

## 5. Формат вывода

После обработки скрипт печатает JSON c агрегированными метриками, например:

```json
{
  "model": "local",
  "samples": 327,
  "accuracy": 66.0,
  "macro_f1": 0.53,
  "latency_ms_avg": 7.1,
  "latency_ms_p90": 8.9
}
```

Ниже идёт подробный `classification_report` и матрица ошибок.

---

## 6. Советы по ускорению DeepFace

* Используйте `--backend skip` (обойдёт детектор лиц).
* Передавайте `--no_preprocess`, если вход уже RGB и нужного размера.
* Ограничьте длину датасета `--max_per_class` во время экспериментов.

---

## 7. Лицензия и цитирование

Датасет CK+ распространяется на своих условиях. Код проекта распространяется под лицензией MIT. 
