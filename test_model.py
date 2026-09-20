import numpy as np
import onnxruntime as ort

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Load ONNX model
session = ort.InferenceSession("brain_tumor_model.onnx", providers=["CPUExecutionProvider"])
inp_name = session.get_inputs()[0].name
print(f"Input: {inp_name}, shape: {session.get_inputs()[0].shape}")

# Test with multiple random images
for i in range(5):
    dummy = np.random.rand(1, 150, 150, 3).astype(np.float32) * 255
    result = session.run(None, {inp_name: dummy})[0][0]
    pred = CLASS_NAMES[np.argmax(result)]
    conf = np.max(result)
    print(f"Test {i+1}: {pred} ({conf:.4f}) | probs={[f'{p:.4f}' for p in result]} | sum={sum(result):.6f}")

# Compare with TensorFlow original
print("\n--- Comparing ONNX vs TensorFlow ---")
import tensorflow as tf
orig_model = tf.keras.models.load_model("brain_tumor_model.keras")

for i in range(3):
    dummy = np.random.rand(1, 150, 150, 3).astype(np.float32) * 255
    tf_result = orig_model(dummy, training=False).numpy()[0]
    onnx_result = session.run(None, {inp_name: dummy})[0][0]
    diff = np.max(np.abs(tf_result - onnx_result))
    print(f"Test {i+1}:")
    print(f"  TF:   {[f'{p:.6f}' for p in tf_result]}")
    print(f"  ONNX: {[f'{p:.6f}' for p in onnx_result]}")
    print(f"  Diff: {diff:.10f}")
    print(f"  Match: {np.allclose(tf_result, onnx_result, atol=1e-3)}")
