import tensorflow as tf

print("TensorFlow 버전:", tf.__version__)
print("CUDA 지원:", tf.test.is_built_with_cuda())
print("사용 가능한 GPU 목록:", tf.config.list_physical_devices('GPU'))

if tf.config.list_physical_devices('GPU'):
    print("✅ GPU를 사용할 수 있습니다!")
else:
    print("❌ GPU를 사용할 수 없습니다.")