import tensorflow.keras.models as models
import numpy as np

path_weight = 'weights/prop_tiramisu_weights_67_12_func_10-e7_decay150.hdf5'

with open('tiramisu_fc_dense67_model_12_func.json') as model_file:
    tiramisu = models.model_from_json(model_file.read())

tiramisu.load_weights(path_weight)

tiramisu.summary()

test_data = np.load('./data/test_data.npy')
test_data = test_data.reshape((233, 224, 224, 3))

result = tiramisu.predict(test_data)

print(max(result), min(result))
