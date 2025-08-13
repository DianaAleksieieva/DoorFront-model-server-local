from flask import Flask, jsonify, request, render_template, redirect, url_for
from detect import detect, detect_multiple_images
from datetime import datetime
from flask_cors import CORS
import os
import json
import pprint
from PIL import Image
import torchvision.transforms as transforms
import torch

pp = pprint.PrettyPrinter(indent=4)
app = Flask(__name__)
CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/")
def index():
    return render_template("home.html", empty=True, label_list=[], time=None)


@app.route("/home_detect", methods=["POST"])
def home_detect():
    # labels = [dict(name="label1", data="label-one"), dict(name="label2", data="label-two")]
    start = datetime.now()

    type_dict = {1: "door", 2: "knob", 3: "stairs"}
    nms = request.form.get("nms", None)
    url = request.form.get("url", None)
    print("home_detect: request.form -> ", request.form)
    if nms != None and url != None:
        label_list = list(detect(min(abs(float(nms)), 1), url))
        result = list(
            map(
                lambda x: dict(
                    bbox=x[0], type=type_dict.get(x[1], "Error"), score=x[2]
                ),
                label_list,
            )
        )
        end = datetime.now()
        difference = (end - start).total_seconds()
        return render_template(
            "home.html", empty=len(result) == 0, label_list=result, time=difference
        )
    else:
        return redirect(url_for("index"))


@app.route("/model/detect", methods=["POST"])
def detect_route():
    type_dict = {1: "door", 2: "knob", 3: "stairs"}
    nms = request.form.get("nms", None)
    url = request.form.get("url", None)
    print("detect: request.form -> ", request.form)
    if url != None:
        label_list = list(detect(min(abs(float(nms)), 1), url))
        result = list(
            map(
                lambda x: dict(
                    bbox=x[0], type=type_dict.get(x[1], "Error"), score=x[2]
                ),
                label_list,
            )
        )
        response = jsonify(result)

        # Enable Access-Control-Allow-Origin
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response
    else:
        return jsonify(code=404, msg="Please input correct parameters!")


@app.route("/model/detect_all", methods=["POST"])
def detect_all_route():
    type_dict = {1: "door", 2: "knob", 3: "stairs"}
    nms = request.form.get("nms", None)
    image_list = request.form.get("img_list", None)
    # print("detect: request.form -> ", request.form)
    if image_list != None:
        result = detect_multiple_images(min(abs(float(nms)), 1), json.loads(image_list))
        final_result = [
            {
                "image_id": prediction["image_id"],
                "labels": list(
                    map(
                        lambda x: dict(
                            bbox=x[0], type=type_dict.get(x[1], "Error"), score=x[2]
                        ),
                        prediction["labels"],
                    )
                ),
            }
            for prediction in result
        ]
        # pp.pprint(final_result)
        response = jsonify(final_result)

        # Enable Access-Control-Allow-Origin
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response
    else:
        return jsonify(code=404, msg="Please input correct parameters!")
    
@app.route("/model/detect_file", methods=["POST"])
def detect_file_route():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        # Read NMS threshold
        nms = float(request.form.get("nms", 0.2))
        nms = min(abs(nms), 1)

        # Read image from upload
        image_file = request.files["image"]
        img = Image.open(image_file.stream).convert("RGB")

        # Convert to tensor
        transform = transforms.ToTensor()
        img_tensor = transform(img)
        img_tensor = torch.unsqueeze(img_tensor, dim=0)

        # Load model
        from detect import load_model, get_result, load_path
        model = load_model(load_path("model.pkl"))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        with torch.no_grad():
            predictions = model(img_tensor.to(device))[0]
            result = get_result(nms, predictions)

        # Map class IDs to labels
        type_dict = {1: "door", 2: "knob", 3: "stairs"}
        parsed = [
            {"bbox": box, "type": type_dict.get(cls, "Unknown"), "score": float(score)}
            for box, cls, score in result
        ]

        return jsonify(parsed)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # print(torch.cuda.is_available())
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
