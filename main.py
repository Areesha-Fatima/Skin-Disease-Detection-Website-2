# import numpy as np
# import cv2
# import tensorflow as tf
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from fastapi.staticfiles import StaticFiles
# from PIL import Image
# import io
# import os

# app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Load the model
# model_path = 'I:/Mobilenetv2 custom train on my pc/Final FastAPI before integrating in mobile app/app/best_model_updated.h5'
# model = tf.keras.models.load_model(model_path)
# last_conv_layer_name = "out_relu"

# class_indices = {
#     0: ['Acne and Rosacea Photos', "Use acne-specific creams like benzoyl peroxide."],
#     1: ['Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', "Seek medical advice for potential skin cancer."],
#     2: ['Atopic Dermatitis Photos', "Use moisturizers and corticosteroid creams."],
#     3: ['Bullous Disease Photos', "Consult a dermatologist for blisters treatment."],
#     4: ['Cellulitis Impetigo and other Bacterial Infections', "Antibiotic treatment may be needed."],
#     5: ['Eczema Photos', "Apply hydrating lotions and avoid allergens."],
#     6: ['Exanthems and Drug Eruptions', "Discontinue the suspected drug and consult a doctor."],
#     7: ['Hair Loss Photos Alopecia and other Hair Diseases', "Consult a specialist for hair loss treatments."],
#     8: ['Herpes HPV and other STDs Photos', "Antiviral medication may be prescribed."],
#     9: ['Light Diseases and Disorders of Pigmentation', "Consult a dermatologist for pigmentation treatments."],
#     10: ['Lupus and other Connective Tissue diseases', "Consult a doctor for autoimmune disease management."],
#     11: ['Melanoma Skin Cancer Nevi and Moles', "Consult an oncologist for melanoma evaluation."],
#     12: ['Nail Fungus and other Nail Disease', "Antifungal treatment might be needed."],
#     13: ['Poison Ivy Photos and other Contact Dermatitis', "Apply anti-itch creams or corticosteroids."],
#     14: ['Psoriasis pictures Lichen Planus and related diseases', "Use specific psoriasis creams and consult a dermatologist."],
#     15: ['Scabies Lyme Disease and other Infestations and Bites', "Use prescribed lotions for scabies or tick removal."],
#     16: ['Seborrheic Keratoses and other Benign Tumors', "Consult a dermatologist for benign tumors."],
#     17: ['Systemic Disease', "Consult a doctor for systemic disease management."],
#     18: ['Tinea Ringworm Candidiasis and other Fungal Infections', "Antifungal creams or medications may be needed."],
#     19: ['Urticaria Hives', "Use antihistamines or consult a doctor for hives."],
#     20: ['Vascular Tumors', "Consult a doctor for tumor evaluation."],
#     21: ['Vasculitis Photos', "Consult a specialist for vasculitis treatment."],
#     22: ['Warts Molluscum and other Viral Infections', "Use wart removal treatments or consult a doctor."]
# }

# def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
#     grad_model = tf.keras.models.Model(
#         [model.inputs[0]], [model.get_layer(last_conv_layer_name).output, model.output]
#     )
#     with tf.GradientTape() as tape:
#         last_conv_layer_output, preds = grad_model(img_array)
#         if pred_index is None:
#             pred_index = tf.argmax(preds[0])
#         class_channel = preds[:, pred_index]

#     grads = tape.gradient(class_channel, last_conv_layer_output)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     last_conv_layer_output = last_conv_layer_output[0]
#     heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)
#     heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#     return heatmap.numpy()

# # Updated image preprocessing function
# def preprocess_image(image: Image.Image):
#     # Resize the image to the same size you used in training (224x224)
#     img = image.resize((224, 224), Image.LANCZOS)
    
#     # Convert the image to a NumPy array and normalize it by dividing by 255.0
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
    
#     # Expand dimensions to create a batch (shape: [1, 224, 224, 3])
#     img_array = np.expand_dims(img_array, axis=0)
    
#     # Normalize the image (if you normalized during training, it's important to match that)
#     img_array = img_array / 255.0
    
#     return img_array

# @app.post("/predict/")
# async def upload_image(file: UploadFile = File(...)):
#     try:
#         image = Image.open(io.BytesIO(await file.read()))
        
#         # Preprocess the image
#         img_array = preprocess_image(image)

#         # Make a prediction
#         preds = model.predict(img_array)
#         pred_class = np.argmax(preds[0])
#         confidence = float(preds[0][pred_class])

#         # Generate class activation heatmap
#         heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

#         # Load the original image for superimposing
#         original_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2RGB)
#         original_image = cv2.resize(original_image, (224, 224))

#         # Resize heatmap to match the original image size
#         heatmap = cv2.resize(heatmap, (224, 224))

#         # Normalize heatmap
#         heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

#         # Apply colormap to heatmap
#         heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

#         # Superimpose heatmap on original image
#         superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap_colored, 0.4, 0)

#         # Save the superimposed image to the static folder
#         output_path = f"static/gradcam_output.png"
#         cv2.imwrite(output_path, superimposed_img)

#         # Get disease name and recommendation
#         disease_name, recommendation = class_indices[pred_class]

#         return JSONResponse(content={
#             "predicted_class": int(pred_class),
#             "disease_name": disease_name,
#             "confidence": confidence,
#             "recommendation": recommendation,
#             "heatmap_image_url": f"/static/gradcam_output.png"
#         })

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# # Run the application
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# oper wala code mein gradcam hein pr galat detect kar raha hein 

# Neche wala code bilkul sahi sirf grad cam daala dea 

import base64
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import os
import cv2


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# Load the model
model_path = r'C:\Users\92337\Desktop\final fastapi\app\best_model_updated.h5'

# Check if the model file exists
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Load the model
model = tf.keras.models.load_model(model_path)
last_conv_layer_name = "out_relu"

# Class indices and recommendations
class_indices = {
    0: ['Acne and Rosacea Photos', "Use acne-specific creams like benzoyl peroxide."],
    1: ['Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', "Seek medical advice for potential skin cancer."],
    2: ['Atopic Dermatitis Photos', "Use moisturizers and corticosteroid creams."],
    3: ['Bullous Disease Photos', "Consult a dermatologist for blisters treatment."],
    4: ['Cellulitis Impetigo and other Bacterial Infections', "Antibiotic treatment may be needed."],
    5: ['Eczema Photos', "Apply hydrating lotions and avoid allergens."],
    6: ['Exanthems and Drug Eruptions', "Discontinue the suspected drug and consult a doctor."],
    7: ['Hair Loss Photos Alopecia and other Hair Diseases', "Consult a specialist for hair loss treatments."],
    8: ['Herpes HPV and other STDs Photos', "Antiviral medication may be prescribed."],
    9: ['Light Diseases and Disorders of Pigmentation', "Consult a dermatologist for pigmentation treatments."],
    10: ['Lupus and other Connective Tissue diseases', "Consult a doctor for autoimmune disease management."],
    11: ['Melanoma Skin Cancer Nevi and Moles', "Consult an oncologist for melanoma evaluation."],
    12: ['Nail Fungus and other Nail Disease', "Antifungal treatment might be needed."],
    13: ['Poison Ivy Photos and other Contact Dermatitis', "Apply anti-itch creams or corticosteroids."],
    14: ['Psoriasis pictures Lichen Planus and related diseases', "Use specific psoriasis creams and consult a dermatologist."],
    15: ['Scabies Lyme Disease and other Infestations and Bites', "Use prescribed lotions for scabies or tick removal."],
    16: ['Seborrheic Keratoses and other Benign Tumors', "Consult a dermatologist for benign tumors."],
    17: ['Systemic Disease', "Consult a doctor for systemic disease management."],
    18: ['Tinea Ringworm Candidiasis and other Fungal Infections', "Antifungal creams or medications may be needed."],
    19: ['Urticaria Hives', "Use antihistamines or consult a doctor for hives."],
    20: ['Vascular Tumors', "Consult a doctor for tumor evaluation."],
    21: ['Vasculitis Photos', "Consult a specialist for vasculitis treatment."],
    22: ['Warts Molluscum and other Viral Infections', "Use wart removal treatments or consult a doctor."]
}

def preprocess_image(image: Image.Image):
    target_size = (224, 224)
    img_ratio = image.width / image.height
    target_ratio = target_size[0] / target_size[1]

    if img_ratio > target_ratio:  # Wider than target
        new_height = target_size[1]
        new_width = int(target_size[1] * img_ratio)
    else:  # Taller than target
        new_width = target_size[0]
        new_height = int(target_size[0] / img_ratio)

    # Resize the image
    image = image.resize((new_width, new_height), Image.LANCZOS)

    # Create a new blank image and paste the resized image
    new_image = Image.new("RGB", target_size, (255, 255, 255))
    x_offset = (target_size[0] - new_width) // 2
    y_offset = (target_size[1] - new_height) // 2
    new_image.paste(image, (x_offset, y_offset))

    # Convert image to array and normalize
    img_array = tf.keras.preprocessing.image.img_to_array(new_image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize to [0, 1]
    return img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs[0]], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

@app.post("/predict/")
async def upload_image(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))

        img_array = preprocess_image(image)

        # Make a prediction
        preds = model.predict(img_array)
        pred_class = np.argmax(preds[0])
        confidence = float(preds[0][pred_class])

        # Generate class activation heatmap
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

        # Load the original image for superimposing
        original_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2RGB)
        original_image = cv2.resize(original_image, (224, 224))

        # Resize heatmap to match the original image size
        heatmap = cv2.resize(heatmap, (224, 224))

        # Normalize heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        # Apply colormap to heatmap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

        # Superimpose heatmap on original image
        superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap_colored, 0.4, 0)

        # Save the superimposed image to the static folder
        output_path = "static/gradcam_output.png"
        cv2.imwrite(output_path, superimposed_img)

        # Get disease name and recommendation
        disease_name, recommendation = class_indices[pred_class]

        return JSONResponse(content={
            "predicted_class": int(pred_class),
            "disease_name": disease_name,
            "confidence": confidence,
            "recommendation": recommendation,
            "heatmap_image_url": f"/static/gradcam_output.png"
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ABOVE CODE MIXTURE

