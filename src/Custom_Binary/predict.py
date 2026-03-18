import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

from model import MR

def predict_mushroom(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    model = MR().to(device)

    # Raw weight names with torch.compile prefixes
    raw_state_dict = torch.load(model_path, map_location=device, weights_only=True)

    clean_state_dict = {}

    # Clean weight names
    for key, value in raw_state_dict.items():
        clean_key = key.replace('_orig_mod.', '')
        clean_state_dict[clean_key] = value

    model.load_state_dict(clean_state_dict)
    model.eval()

    # Make prediction
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            raw_logit = model(image_tensor)
            # Convert to probability
            probability = torch.sigmoid(raw_logit).item()

    if probability > 0.5:
        print("Mushroom is INEDIBLE")
        print(f"Prediction confidence: {probability * 100:.2f}%")
    else:
        print("Mushroom is EDIBLE")
        print(f"Prediction confidence: {(1 - probability) * 100:.2f}%")


if __name__ == '__main__':
    TARGET_IMAGE = "./user_images/Laetiporus_sulphureus_EDIBLE.jpg"
    WEIGHTS = "./models/custom_binary_best_mushroom_roulette.pth"

    predict_mushroom(TARGET_IMAGE, WEIGHTS)