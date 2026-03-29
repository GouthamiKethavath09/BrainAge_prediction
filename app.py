import streamlit as st
import torch
import numpy as np
import nibabel as nib
import torch.nn as nn
import torch.nn.functional as F

# -------- MODEL --------
class BrainAgeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(128*11*13*4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# -------- LOAD MODEL --------
model = BrainAgeModel()
model.load_state_dict(torch.load("brain_age_model.pth", map_location="cpu"))
model.eval()

# -------- PREPROCESS --------
def load_mri(file):
    img = nib.load(file)
    return img.get_fdata()

def normalize(img):
    return (img - np.mean(img)) / np.std(img)

def resize(img):
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
    img = F.interpolate(img, size=(91,109,91), mode='trilinear')
    return img.squeeze().numpy()

def preprocess(file):
    img = load_mri(file)
    img = normalize(img)
    img = resize(img)
    return img

# -------- UI --------
st.title("🧠 Brain Age Prediction System")

uploaded_file = st.file_uploader("Upload MRI (.nii file)")

if uploaded_file:
    with open("temp.nii", "wb") as f:
        f.write(uploaded_file.read())

    img = preprocess("temp.nii")
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()

    pred_age = model(img).item()

    st.write("### Predicted Brain Age:", round(pred_age, 2))

    actual_age = 40
    bag = pred_age - actual_age

    st.write("Brain Age Gap:", round(bag, 2))

    if bag > 5:
        st.error("⚠️ High Risk")
    else:
        st.success("✅ Normal")