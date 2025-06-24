# train.py
import os, tqdm, numpy as np, torch, torch.nn as nn, torch.optim as optim
from data import getPathListAndLabelsOfPlants, getLabeledListAsDictionary, findEquivalentOfLabels
from constant import TRAIN_DIR, VALID_DIR
from preprocess import tensorize_image
from model_ResNet import ResNetModel
from utils import AverageMeter, Plot, SaveBestModel, prediction

def train(epochs=6, batch_size=16, img_size=(224, 224), lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs("../models", exist_ok=True)

    # Model, loss, optim
    num_classes = len(getLabeledListAsDictionary(TRAIN_DIR))
    model = ResNetModel(50, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    save_best_model = SaveBestModel()

    # Veri hazırlığı
    train_paths, train_labels = getPathListAndLabelsOfPlants(TRAIN_DIR)
    valid_paths, _ = getPathListAndLabelsOfPlants(VALID_DIR)
    label_dict = getLabeledListAsDictionary(TRAIN_DIR)
    valid_labels = findEquivalentOfLabels(valid_paths, label_dict, VALID_DIR)

    # Shuffle
    idx_tr = np.random.permutation(len(train_paths))
    idx_va = np.random.permutation(len(valid_paths))
    train_paths  = list(np.array(train_paths)[idx_tr])
    train_labels = list(np.array(train_labels)[idx_tr])
    valid_paths  = list(np.array(valid_paths)[idx_va])
    valid_labels = list(np.array(valid_labels)[idx_va])

    steps_per_epoch = len(train_paths) // batch_size
    loss_hist, val_loss_hist = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = correct = total = 0

        for step in tqdm.tqdm(range(steps_per_epoch), ncols=80, desc=f"Epoch {epoch+1}/{epochs}"):
            bp = train_paths[step*batch_size:(step+1)*batch_size]
            bl = torch.tensor(train_labels[step*batch_size:(step+1)*batch_size], dtype=torch.long).to(device)
            imgs = tensorize_image(bp, img_size, device.type=="cuda")
               
               # Eşitlik kontrolü: resim sayısı != etiket sayısı → batch atla
            if imgs.size(0) != bl.size(0):
                print(f"Batch atlandı: {len(bp)} görsel bekleniyordu, {imgs.size(0)} tane okunabildi.")
                continue
            
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, bl)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = prediction(out)
            total += bl.size(0)
            correct += (preds == bl).sum().item()

        train_acc = 100 * correct / total
        avg_loss = running_loss / steps_per_epoch
        loss_hist.append(avg_loss)

        # ---- VALIDATION ----
        model.eval()
        val_running_loss = val_correct = val_total = 0
        with torch.no_grad():
            # batch’ler halinde dönüyoruz
            for v_step in range((len(valid_paths) + batch_size - 1)//batch_size):
                start = v_step * batch_size
                end   = start + batch_size
                vp    = valid_paths[start:end]
                vl    = valid_labels[start:end]

                if not vp: break
                v_imgs   = tensorize_image(vp, img_size, device.type=="cuda")
                v_labels = torch.tensor(vl, dtype=torch.long).to(device)

                v_out = model(v_imgs)
                v_loss = criterion(v_out, v_labels)

                val_running_loss += v_loss.item() * v_labels.size(0)
                v_preds = prediction(v_out)
                val_total += v_labels.size(0)
                val_correct += (v_preds == v_labels).sum().item()

        avg_val_loss = val_running_loss / val_total
        val_loss_hist.append(avg_val_loss)
        val_acc = 100 * val_correct / val_total

        print(f"\nEpoch {epoch+1}: "
              f"TrainLoss {avg_loss:.4f} | ValLoss {avg_val_loss:.4f} | "
              f"TrainAcc {train_acc:.2f}% | ValAcc {val_acc:.2f}%")
        save_best_model(v_loss, epoch, model, optimizer, criterion)

    # Grafikler
    Plot(range(epochs), loss_hist, ylabel="Train Loss", save=True)
    Plot(range(epochs), val_loss_hist, ylabel="Val Loss", save=True)

    # Son modeli de tam olarak kaydet
    torch.save(model, "../models/last_model.pth")
