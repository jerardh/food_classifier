from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from keras.layers import TFSMLayer
from langchain_core.messages import HumanMessage
import os
app = Flask(__name__)
model = tf.keras.models.load_model("./Model/model_v1_inceptionV3.h5")
os.environ['GOOGLE_API_KEY'] = os.getenv("API_KEY")
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.2)
class_labels=['burger', 'butter_naan', 'chai', 'chapati', 'chole_bhature', 'dal_makhani', 'dhokla', 'fried_rice', 'idli', 'jalebi', 'kaathi_rolls', 'kadai_paneer', 'kulfi', 'masala_dosa', 'momos', 'paani_puri', 'pakode', 'pav_bhaji', 'pizza', 'samosa']
@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files['image']
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((120, 120))  # match model input
        img_array = np.array(img, dtype=np.float32) / 255.0
        input_tensor = np.expand_dims(img_array, axis=0)

        prediction = model.predict(input_tensor)
        pred_idx = np.argmax(prediction, axis=1)[0]
        pred_label = class_labels[pred_idx]
        pred_prob = prediction[0][pred_idx]
        pred_list  = prediction.tolist()
        return jsonify({
            "prediction_prob": float(pred_prob),
            "predicted_class": pred_label
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route("/get_food_info", methods=["GET"])
def get_food_info():
    food_name = request.args.get("dish")
    if not food_name:
        return jsonify({"error": "Please provide a dish name with ?dish=..." }), 400

    prompt = f"""
    You are a nutrition assistant for Indian foods.
    Given the name of an Indian dish, return its main ingredients and approximate nutritional values per serving in JSON format.

    Output JSON with the following keys only:
    - dish_name (string)
    - ingredients (array of strings)

    Now return the same JSON for: **{food_name}**
    """

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        food_info = response.content.strip()
        return food_info, 200, {"Content-Type": "application/json"}

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)