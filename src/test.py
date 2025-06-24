# test.py
import glob
import os
import torch
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from constant import TEST_DIR, BEST_MODEL_DIR, TRAIN_DIR
from model_ResNet import ResNetModel
from preprocess import tensorize_image
from utils import prediction
from data import getLabeledListAsDictionary

def test(img_size=(224,224), batch_size=16, use_cuda=True):
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    # Model yükle
    num_classes = len(getLabeledListAsDictionary(TRAIN_DIR))
    model = ResNetModel(50, num_classes)
    checkpoint = torch.load(BEST_MODEL_DIR, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    # Test verisi
    test_paths = glob.glob(os.path.join(TEST_DIR, "*", "*"))

    # Tahminleri toplama
    all_preds = []
    for i in tqdm.tqdm(range(0, len(test_paths), batch_size), desc="Testing", ncols=80):
        batch_paths = test_paths[i:i+batch_size]
        batch_imgs  = tensorize_image(batch_paths, img_size, device.type == "cuda")
        with torch.no_grad():
            outputs = model(batch_imgs)
            preds = prediction(outputs).cpu().numpy().tolist()
        all_preds.extend(preds)

    # Gerçek ve tahmin isimleri
    label_dict = getLabeledListAsDictionary(TRAIN_DIR)
    print(f"Label Dictionary: {label_dict}")
    pred_names = [label_dict[int(p)] for p in all_preds]
    fact_names = [path.split(os.sep)[-2] for path in test_paths]

    df = pd.DataFrame({"Actual": fact_names, "Prediction": pred_names})
    df["Match"] = df["Actual"] == df["Prediction"]

    # Confusion Matrix DataFrame
    cm = pd.crosstab(df["Actual"], df["Prediction"], rownames=["Actual"], colnames=["Predicted"])

    # Seaborn ile görsel Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Per-class stats
    stats = []
    for cls in cm.index:
        correct = cm.at[cls, cls] if cls in cm.columns else 0
        total = cm.loc[cls].sum()
        acc = 100 * correct / total if total > 0 else 0.0
        stats.append({"Class": cls, "Correct": int(correct),
                      "Total": int(total), "Accuracy (%)": round(acc, 2)})
    stats_df = pd.DataFrame(stats)
    print("\nPer-Class Stats:")
    print(stats_df.to_string(index=False))

    # En iyi / en kötü sınıf
    best = stats_df.loc[stats_df["Accuracy (%)"].idxmax()]
    worst = stats_df.loc[stats_df["Accuracy (%)"].idxmin()]
    print(f"\nBest Class : {best['Class']} with {best['Accuracy (%)']}%")
    print(f"Worst Class: {worst['Class']} with {worst['Accuracy (%)']}%")

    # Genel metrikler
    print("\nClassification Report:")
    print(classification_report(df["Actual"], df["Prediction"], target_names=cm.index.tolist()))

    # Toplam skor
    total_samples   = len(df)
    total_correct   = int(df["Match"].sum())
    total_incorrect = total_samples - total_correct
    overall_acc     = 100 * total_correct / total_samples if total_samples > 0 else 0.0

    print(f"\nTotal samples   : {total_samples}")
    print(f"Total correct   : {total_correct}")
    print(f"Total incorrect : {total_incorrect}")
    print(f"Overall Accuracy: {overall_acc:.2f}%\n")

if __name__ == "__main__":
    test()
