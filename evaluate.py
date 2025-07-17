#!/usr/bin/env python
"""evaluate.py
Тестирование распознавания эмоций (DeepFace vs OpenAI GPT-4 Vision)

Запуск примеров:
  python evaluate.py --mode local  --csv ck_paths.csv --root ck_videos
  python evaluate.py --mode cloud  --csv ck_paths.csv --root ck_videos --api_key $OPENAI_API_KEY
"""
from __future__ import annotations

import argparse, pathlib, time, base64, sys, json
from typing import List

import pandas as pd
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm

# ---------------------------- Predictors --------------------------------- #
class Predictor:
    """Базовый интерфейс."""
    name: str = "base"

    def predict(self, frame) -> str:
        raise NotImplementedError

class LocalPredictor(Predictor):
    """DeepFace predictor"""
    name = "local"

    def __init__(self, detector_backend: str = "retinaface", enforce: bool = True):
        """detector_backend: retinaface, opencv, mtcnn, skip и др.
        enforce: если False, DeepFace не будет ругаться при отсутствии лиц.
        """
        from deepface import DeepFace  # импорт внутри, чтобы не тянуть зависимость при cloud
        self._backend = DeepFace
        self._actions = ["emotion"]
        self._detector_backend = detector_backend
        self._enforce = enforce

    def predict(self, frame):
        # Сначала пытаемся с детектором лиц (точнее)
        try:
            res = self._backend.analyze(
                frame,
                actions=["emotion"],
                detector_backend=self._detector_backend,
                enforce_detection=self._enforce,
            )
        except Exception:
            # fallback: без детекции (возвращает neutral, но лучше, чем ошибка)
            res = self._backend.analyze(
                frame,
                actions=["emotion"],
                detector_backend="skip",
                enforce_detection=False,
            )
        emotions = res[0]["emotion"]
        top_label = max(emotions, key=emotions.get).lower()
        confidence = emotions[top_label]
        return top_label, confidence

class CloudPredictor(Predictor):
    """OpenAI Vision predictor"""
    name = "cloud"

    def __init__(self, api_key: str):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)

    _PROMPT = (
        "Определи эмоцию человека на изображении. Ответь ОДНИМ СЛОВОМ "
        "на английском из списка: angry, disgust, fear, happy, neutral, sad, surprise. "
        "Не добавляй пояснений."
    )

    def predict(self, frame):
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img64 = base64.b64encode(buf).decode()
        resp = self._client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img64}"}},
                    ],
                }
            ],
            max_tokens=5,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip().lower()
        label = raw.split()[0]          # берём первое слово
        allowed = {"angry","disgust","fear","happy","neutral","sad","surprise"}
        if label not in allowed:
            label = "neutral"           # или label = "unknown" и отфильтровать
        return label, 1.0

# ------------------------------------------------------------------------- #

# ----------- Pre-processing (upsample & grayscale→RGB) -------------------- #

def prepare_frame(frame, target: int = 224):
    """Приводит изображение к 3-канальному формату и увеличивает до target px."""
    global _SKIP_PREPROCESS
    if _SKIP_PREPROCESS:
        return frame
    if frame is None:
        return frame
    # grayscale → BGR
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = frame.shape[:2]
    if min(h, w) < target:
        frame = cv2.resize(frame, (target, target), interpolation=cv2.INTER_CUBIC)
    return frame

# ------------- helpers ---------------------------------------------------- #

def get_first_frame(cap: cv2.VideoCapture):
    """Возвращает первый кадр ролика (pos=0)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, frame = cap.read()
    return frame


def get_frames(cap: cv2.VideoCapture, max_frames:int=3):
    """Возвращает список до max_frames кадров, равномерно распределённых по ролику."""
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step = max(frames_total // max_frames, 1)
    frames = []
    for i in range(0, frames_total, step):
        if len(frames) >= max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    return frames or [cap.read()[1]]  # если что-то пошло не так, вернём первый кадр

def get_last_frame(cap):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total-1)
    _, f = cap.read();  return f


def evaluate(
    csv_path: pathlib.Path,
    root: pathlib.Path,
    predictor: Predictor,
    show_details: bool = False,
    max_per_class: int = 0,
    conf_thres: float = 0.5,
    exclude_neutral: bool = False,
):
    df = pd.read_csv(csv_path)
    if not {"path", "label"}.issubset(df.columns):
        print("CSV должен содержать столбцы path,label", file=sys.stderr)
        sys.exit(1)

    # --- опционально ограничиваем выборку: не более max_per_class видео на эмоцию ---
    if max_per_class > 0:
        df = df.groupby('label', as_index=False).head(max_per_class).reset_index(drop=True)

    y_true: List[str] = []
    y_pred: List[str] = []
    latencies: List[float] = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Видеоролики", unit="vid"):
        vid_path = root / row["path"]
        label = str(row["label"]).lower()
        if not vid_path.exists():
            print(f"⚠️  файл не найден: {vid_path}")
            continue
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"⚠️  не удалось открыть: {vid_path}")
            continue

        frame = get_first_frame(cap)
        cap.release()

        if (idx % 50) == 0 and idx:
            print(f"🔄  Обработано {idx} видео")

        fr_prep = prepare_frame(frame)
        t0 = time.perf_counter()
        pred, conf = predictor.predict(fr_prep)
        lat = (time.perf_counter() - t0) * 1000

        # маппинг истинной метки к формату deepface
        map_ck_to_df = {
            "anger": "angry",
            "sadness": "sad",  # CK+ sadness -> DeepFace sad
        }
        label_mapped = map_ck_to_df.get(label, label)

        # детальный вывод перед фильтрацией
        if show_details:
            tqdm.write(
                f"{vid_path.name}: true={label.lower()} mapped={label_mapped} | pred={pred} conf={conf:.3f}"
            )

        # фильтры
        if conf < conf_thres:
            continue
        if exclude_neutral and (label_mapped == "neutral" or pred == "neutral"):
            continue

        y_true.append(label_mapped)
        y_pred.append(pred)
        latencies.append(lat)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    avg_lat = float(np.mean(latencies))
    p90 = float(np.percentile(latencies, 90))

    results = {
        "model": predictor.name,
        "samples": len(y_true),
        "accuracy": round(acc * 100, 2),
        "macro_f1": round(f1, 3),
        "latency_ms_avg": round(avg_lat, 1),
        "latency_ms_p90": round(p90, 1),
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))

    # --- подробные метрики ---
    try:
        print("\nClassification report:\n", classification_report(y_true, y_pred, digits=2))
        labels_sorted = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
        print("Confusion matrix (rows=true, cols=pred):")
        # печатаем красивее
        header = "        " + "  ".join(f"{l:>8}" for l in labels_sorted)
        print(header)
        for lbl, row_cm in zip(labels_sorted, cm):
            row_str = "  ".join(f"{v:8d}" for v in row_cm)
            print(f"{lbl:>8}  {row_str}")
    except Exception as e:
        print("Не удалось вывести подробные метрики:", e)


def main():
    ap = argparse.ArgumentParser(description="Evaluate emotion recognition models")
    ap.add_argument("--csv", default="ckextended.csv", help="CSV file with path,label columns")
    ap.add_argument("--root", default="ck_videos", help="Root directory with videos")
    ap.add_argument("--mode", choices=["local", "cloud"], required=True)
    ap.add_argument("--api_key", help="OpenAI API key (for cloud mode)")
    ap.add_argument("--details", action="store_true", help="Показывать предсказание для каждого видео")
    ap.add_argument("--backend", default="retinaface", help="DeepFace detector backend (retinaface|opencv|mtcnn|skip)")
    ap.add_argument("--no_enforce", action="store_true", help="Не требовать наличия лица (enforce_detection=False)")
    ap.add_argument("--list_emotions", action="store_true", help="Вывести поддерживаемые DeepFace эмоции и выйти")
    ap.add_argument("--no_preprocess", action="store_true", help="Пропускать предварительную обработку кадров")
    ap.add_argument("--max_per_class", type=int, default=10, help="Ограничить количество видео на эмоцию (0 - не ограничивать)")
    ap.add_argument("--conf_thres", type=float, default=0.5, help="Порог уверенности DeepFace (prediction confidence)")
    ap.add_argument("--exclude_neutral", action="store_true", help="Исключить neutral из расчёта метрик")
    args = ap.parse_args()

    csv_path = pathlib.Path(args.csv)
    root = pathlib.Path(args.root)

    if args.list_emotions:
        # выводим список эмоций DeepFace и выходим
        from deepface import DeepFace
        import numpy as _np
        dummy = _np.zeros((10, 10, 3), dtype=_np.uint8)
        emotions = DeepFace.analyze(dummy, actions=["emotion"], enforce_detection=False)[0]["emotion"].keys()
        print("DeepFace emotions:", ", ".join(emotions))
        return

    if args.mode == "local":
        predictor = LocalPredictor(detector_backend=args.backend, enforce=not args.no_enforce)
    else:
        if not args.api_key:
            print("--api_key required for cloud mode", file=sys.stderr)
            sys.exit(1)
        predictor = CloudPredictor(args.api_key)

    global _SKIP_PREPROCESS
    _SKIP_PREPROCESS = args.no_preprocess

    evaluate(
        csv_path,
        root,
        predictor,
        show_details=args.details,
        max_per_class=args.max_per_class,
        conf_thres=args.conf_thres,
        exclude_neutral=args.exclude_neutral,
    )


if __name__ == "__main__":
    main() 