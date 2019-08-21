from keras.models import model_from_json
import tensorflow as tf
from uv_radiation_forecasting import data_preprocess

x_tarin , x_test , y_train , y_test = data_preprocess()

from keras.models import model_from_json
import tensorflow as tf
from uv_radiation_forecasting import data_preprocess

x_train , x_test , y_train , y_test = data_preprocess()
json_file = open('model_uv_forecasting.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model_uv_forecasting.h5')
print("Loaded model from disk")

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(
  allow_soft_placement=True, log_device_placement=True))

loaded_model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
loaded_model.fit(x_train , y_train , batch_size = 256 , epochs = 120)
print(loaded_model.evaluate(x_test , y_test))

model_json = loaded_model.to_json()
with open("model_uv_forecasting2.json", "w") as json_file:
    json_file.write(model_json)
loaded_model.save_weights("model_uv_forecasting2.h5")
print("Saved model to disk")
