"""
Keras CNN model (.keras / .h5 / SavedModel) -> ONNX

Install:
    pip install tensorflow tf2onnx onnx onnxruntime pillow

Usage:
    python keras_to_onnx.py my_model.keras model.onnx
"""
import subprocess
import sys

import numpy as np
import onnxruntime as ort
import tensorflow as tf

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "model.keras"   # ya model.h5
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "model.onnx"
OPSET = 17


def load(path):
    # compile=False -> optimizer/loss ki dikkat nahi aayegi, inference ke liye kaafi hai
    model = tf.keras.models.load_model(path, compile=False)
    model.summary()
    print("Input shape :", model.input_shape)   # e.g. (None, 224, 224, 3)  -> NHWC
    print("Output shape:", model.output_shape)
    return model


def export_method_1(model, out):
    """Keras 3 (TF 2.16+ / Keras >= 3.10) ka built-in ONNX export."""
    model.export(out, format="onnx")


def export_method_2(model, out):
    """Sabse reliable: pehle SavedModel, phir tf2onnx CLI."""
    model.export("saved_model_tmp")   # Keras 3.  Keras 2 ho to: model.save("saved_model_tmp")
    subprocess.check_call([
        sys.executable, "-m", "tf2onnx.convert",
        "--saved-model", "saved_model_tmp",
        "--output", out,
        "--opset", str(OPSET),
    ])


def export_method_3(model, out):
    """Purane Keras 2 / TF <= 2.15 ke liye direct tf2onnx."""
    import tf2onnx
    spec = (tf.TensorSpec(model.inputs[0].shape, tf.float32, name="input"),)
    tf2onnx.convert.from_keras(model, input_signature=spec, opset=OPSET, output_path=out)


def export(model, out):
    for fn in (export_method_1, export_method_2, export_method_3):
        try:
            print(f"\n>>> Trying {fn.__name__} ...")
            fn(model, out)
            print(f"Exported with {fn.__name__}")
            return
        except Exception as e:
            print(f"{fn.__name__} failed: {type(e).__name__}: {str(e)[:300]}")
    raise SystemExit("Teeno methods fail hue - upar ka error message copy karke bhejo.")


def verify(model, onnx_path):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    print("\nONNX input :", inp.name, inp.shape, inp.type)
    print("ONNX output:", sess.get_outputs()[0].name, sess.get_outputs()[0].shape)

    shape = [1 if d is None else d for d in model.input_shape]
    dummy = np.random.rand(*shape).astype(np.float32)

    keras_out = model.predict(dummy, verbose=0)
    onnx_out = sess.run(None, {inp.name: dummy})[0]

    np.testing.assert_allclose(keras_out, onnx_out, rtol=1e-3, atol=1e-4)
    print("Verified: Keras aur ONNX ka output match kar raha hai")


def predict_image(onnx_path, image_path, size=(224, 224), scale_255=True):
    """
    Deployment me aise use karo.
    IMPORTANT: preprocessing bilkul training jaisi honi chahiye.
      - Agar model ke andar Rescaling(1/255) layer hai -> scale_255=False
      - Agar training me manually /255 kiya tha       -> scale_255=True
    """
    from PIL import Image

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    img = Image.open(image_path).convert("RGB").resize(size)
    x = np.asarray(img, dtype=np.float32)
    if scale_255:
        x = x / 255.0
    x = np.expand_dims(x, 0)          # (1, H, W, C)  <- Keras NHWC, transpose mat karna

    probs = sess.run(None, {name: x})[0][0]
    return int(np.argmax(probs)), probs


if __name__ == "__main__":
    m = load(MODEL_PATH)
    export(m, OUT_PATH)
    verify(m, OUT_PATH)
