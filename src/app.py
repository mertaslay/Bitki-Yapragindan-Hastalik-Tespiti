# src/app.py
import os
import uuid
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import torch
from PIL import Image
from torchvision import transforms
from data import getLabeledListAsDictionary
from constant import TRAIN_DIR, BEST_MODEL_DIR
from preprocess import tensorize_image

# --- Dizin ayarları ---
BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR   = os.path.join(BASE_DIR, "static")

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)

UPLOAD_FOLDER = os.path.join(STATIC_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Cihaz seçimi ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Modeli yükle (hem full-model hem checkpoint dict destekli) ---
raw = torch.load(os.path.join(BASE_DIR, BEST_MODEL_DIR), map_location=device)

# Eğer full model objesi kaydettiyseniz raw bir nn.Module olacaktır
if isinstance(raw, torch.nn.Module):
    model = raw

# Eğer checkpoint dict kaydettiyseniz, içinden state_dict çekip yeniden oluştur
elif isinstance(raw, dict):
    from model_ResNet import ResNetModel
    num_classes = len(getLabeledListAsDictionary(TRAIN_DIR))
    model = ResNetModel(50, num_classes)
    # checkpoint dict içinde 'model_state_dict' anahtarı varsa onu, yoksa raw’ın tamamını yükle
    sd = raw.get('model_state_dict', raw)
    model.load_state_dict(sd)
else:
    raise RuntimeError(f"Unsupported checkpoint type: {type(raw)}")

model = model.to(device)
model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename   = None
    error      = None

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not file.filename:
            error = "Lütfen bir dosya seçin."
            return render_template('index.html', error=error)

        # Unique filename
        orig_name   = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{orig_name}"
        filepath    = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(filepath)

        # Debug
        print("Uploaded path:", filepath)

        # Tahmin
        img_t  = tensorize_image([filepath], (224, 224), device.type == "cuda")

        with torch.no_grad():
            out      = model(img_t)
            pred_idx = torch.argmax(out, dim=1).item()

        # Debug
        print("Raw model output:", out)
        print("Predicted idx:", pred_idx)

        # Label dict
        label_dict = getLabeledListAsDictionary(TRAIN_DIR)
        print("Label Dictionary:", label_dict)

        # Prediction
        prediction = label_dict.get(pred_idx, "Unknown")
        filename   = unique_name

    return render_template(
        'index.html',
        prediction=prediction,
        filename=filename,
        error=error
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
