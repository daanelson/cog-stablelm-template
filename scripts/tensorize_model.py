#!/usr/bin/env python

from tensorizer import TensorSerializer
import sys
sys.path.append('/src/')

from config import LOCAL_PATH, load_model

print(f'Saving model to {LOCAL_PATH}')

model = load_model()
print(model.dtype)
serializer = TensorSerializer(LOCAL_PATH)
serializer.write_module(model)
serializer.close()
