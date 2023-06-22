import tf2onnx
import numpy as np
import onnxruntime as rt
import os
import tensorflow as tf
import efficientnet.tfkeras

img = tf.keras.utils.load_img("testimage/sad.jpg", target_size=(197, 197, 3))

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

img = tf.keras.utils.img_to_array(img)
img = img / 255

x = np.expand_dims(img, axis=0)

model = tf.keras.models.load_model('Ensemble.h5')
print(model.input_shape)
spec = (tf.TensorSpec((None, None, None, 3), tf.float32, name="input"),)
output_path = "ensemble.onnx"
print("Converted model to ONNX:")

print(output_path)

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
output_names = [n.name for n in model_proto.graph.output]
print(output_names)

providers = ['CPUExecutionProvider']
m = rt.InferenceSession(output_path, providers=providers)
input_name = m.get_inputs()[0].name
output_name = m.get_outputs()[0].name
print(input_name)
print(output_name)
onnx_pred = m.run(output_names, {"input": x})
res = int(np.argmax(onnx_pred))
print("Result of ONNX model:")
print(emotion_dict[res])

# make sure ONNX and keras have the same results
preds = model.predict(x)
maxindex = int(np.argmax(preds))
print("Result of Keras model:")
print(emotion_dict[maxindex])

