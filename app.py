from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
import torch
import numpy as np
import io
import torchvision.transforms as transforms
from model import MyNN  # import the model class


device = torch.device("cpu")
no_convs: int = 2
conv_channels: list[int] = [30, 50]
kernel_sizes: list[int] = [3, 4]
pool_sizes: list[int] = [2, 2]
num_hidden_layers: int = 3
n_layer: int = 101
dropout_rate: float = 0.312516596075442
learnrate: float = 0.0009910923438815737
weight_decay: float = 0.0002464037290977692
epochs: int = 47
batch_size: int = 43
model = MyNN(
        no_convs=no_convs,
        input_features=1,
        conv_channels=conv_channels,
        kernel_sizes=kernel_sizes,
        pool_sizes=pool_sizes,
        num_hidden_layers=num_hidden_layers,
        n_layer=n_layer,
        dropout_rate=dropout_rate,
        out_dim=10,
    )
model = model.to(device)  # initialize architecture
model.load_state_dict(torch.load("models/my_model_weights.pth", map_location=torch.device("cpu")))


model.eval()

app = FastAPI(title="FashionMNIST CNN Inference API")



LABELS = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}




class PredictionResponse(BaseModel):
    outputs: list = Field(..., description="Raw model output probabilities (list of 10 floats)")
    filename: str = Field(..., description="Name of the uploaded image")
    predicted_class: int = Field(..., description="Predicted label index (0–9)")
    label_name: str = Field(..., description="Human-readable class name")
    confidence: float = Field(..., ge=0, description="Model confidence score")




def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    
    transform = transforms.Compose([
        transforms.Grayscale(),          # convert to grayscale
        transforms.Resize((28, 28)),     # resize to 28x28
        transforms.ToTensor(),           # convert to tensor (0–1)
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = transform(image).unsqueeze(0)
    return image









# @app.post("/predict/", response_model=PredictionResponse)
# async def predict(file: UploadFile = File(...)):
    
#     if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
#         raise HTTPException(status_code=400, detail="Only .png, .jpg, or .jpeg files are supported")

#     try:
#         image_bytes = await file.read()
#         img_tensor = preprocess_image(image_bytes)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")



# #Debugging
#     # --- debug snippet: paste inside /predict after img_tensor is created ---

#     # 1) shape & pixel stats
#     print("DEBUG: img_tensor.shape", img_tensor.shape)   # should be (1,1,28,28)
#     print("DEBUG: img min/max/mean:", float(img_tensor.min()), float(img_tensor.max()), float(img_tensor.mean()))

#     # 2) check that model has non-zero weights (basic sanity)
#     total_params = sum(p.numel() for p in model.parameters())
#     zero_params = sum((p==0).sum().item() for p in model.parameters())
#     print(f"DEBUG: total params={total_params}, zero params={zero_params}")

#     # 3) print mean/std of a few named parameters (to check they are non-trivial)
#     for name, p in model.named_parameters():
#         if "weight" in name:
#             print(f"DEBUG param {name}: mean={p.data.mean().item():.4e}, std={p.data.std().item():.4e}")
#         # print a few only
#         break

#     # 4) forward pass check: model output and probabilities
#     with torch.no_grad():
#         logits = model(img_tensor.to(next(model.parameters()).device))
#         probs = torch.softmax(logits, dim=1)
#     print("DEBUG logits:", logits.cpu().numpy().tolist())
#     print("DEBUG probs:", probs.cpu().numpy().tolist())
#     print("DEBUG predicted_class (argmax):", int(torch.argmax(probs, dim=1).item()), "conf(prob):", float(probs.max().item()))

#     # 5) Optional: test with a random tensor to see if model output changes
#     r = torch.rand_like(img_tensor)
#     with torch.no_grad():
#         l_r = model(r.to(next(model.parameters()).device))
#         p_r = torch.softmax(l_r, dim=1)
#     print("DEBUG random input probs:", p_r.cpu().numpy().tolist())


#     with torch.no_grad():
#         outputs = model(img_tensor)     #######
#         _,predicted =torch.max(outputs,1)
#         probs = torch.softmax(logits, dim=1)             # (1,10)
#         pred_class = int(torch.argmax(probs, dim=1).item())
#         confidence = float(probs.max().item())

#     return PredictionResponse(
#         outputs=outputs.squeeze(0).tolist(),
#         filename=file.filename,
#         predicted_class=pred_class,
#         label_name=LABELS[predicted.item()],
#         confidence=round(confidence, 4)
#     )







@app.post("/predict/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # --- basic file checks + reading ---
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Only .png, .jpg, or .jpeg files are supported")

    try:
        image_bytes = await file.read()
        img_tensor = preprocess_image(image_bytes)   # shape: (1,1,28,28)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # ---------------- DEBUG (remove these prints in production) ----------------
    print("DEBUG: img_tensor.shape", img_tensor.shape)   # should be (1,1,28,28)
    print("DEBUG: img min/max/mean:", float(img_tensor.min()), float(img_tensor.max()), float(img_tensor.mean()))

    total_params = sum(p.numel() for p in model.parameters())
    zero_params = sum((p==0).sum().item() for p in model.parameters())
    print(f"DEBUG: total params={total_params}, zero params={zero_params}")

    # print first weight stats for sanity
    for name, p in model.named_parameters():
        if "weight" in name:
            print(f"DEBUG param {name}: mean={p.data.mean().item():.4e}, std={p.data.std().item():.4e}")
            break

    # ensure tensor & model on same device
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)

    # forward pass and probabilities (single, consistent forward pass)
    with torch.no_grad():
        logits = model(img_tensor)             # shape (1,10)
        probs = torch.softmax(logits, dim=1)   # shape (1,10)
        pred_class = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs.max().item())

    # debug prints (inspect these in the terminal)
    print("DEBUG logits:", logits.cpu().numpy().tolist())
    print("DEBUG probs:", probs.cpu().numpy().tolist())
    print("DEBUG predicted_class (argmax):", pred_class, "conf(prob):", confidence)

    # Optional: check model response to random input (dev only)
    r = torch.rand_like(img_tensor)
    with torch.no_grad():
        r_logits = model(r)
        r_probs = torch.softmax(r_logits, dim=1)
    print("DEBUG random input probs:", r_probs.cpu().numpy().tolist())
    # ---------------- end DEBUG ----------------

    # Build response - return probabilities (not raw logits)
    return PredictionResponse(
        outputs=probs.squeeze(0).cpu().tolist(),   # list of 10 probabilities
        filename=file.filename,
        predicted_class=pred_class,
        label_name=LABELS[pred_class],
        confidence=round(confidence, 4)
    )
