#!/usr/bin/env python3
"""
TensorFlow/Keras Compatibility Check
Test different import methods for TensorFlow 2.x vs 3.x
"""

import sys

def test_tensorflow_imports():
    """Test TensorFlow and Keras imports"""
    print("=== TensorFlow/Keras Import Test ===")
    
    # Test TensorFlow
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} imported successfully")
        tf_available = True
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    # Test Keras imports - try multiple methods
    keras_methods = [
        ("Direct Keras", "import keras"),
        ("TensorFlow Keras", "from tensorflow import keras"),
        ("TensorFlow.keras", "import tensorflow.keras as keras")
    ]
    
    working_method = None
    
    for method_name, import_statement in keras_methods:
        try:
            exec(import_statement)
            print(f"✓ {method_name}: {import_statement} - SUCCESS")
            if working_method is None:
                working_method = method_name
        except ImportError as e:
            print(f"✗ {method_name}: {import_statement} - FAILED: {e}")
    
    if working_method:
        print(f"\n✓ Best import method: {working_method}")
    else:
        print("\n✗ No working Keras import method found")
        return False
    
    # Test specific components
    print("\n=== Component Import Test ===")
    
    # Test models
    try:
        if working_method == "Direct Keras":
            from keras.models import Sequential, Model
        elif working_method == "TensorFlow Keras":
            from tensorflow.keras.models import Sequential, Model
        else:
            from tensorflow.keras.models import Sequential, Model
        print("✓ Models (Sequential, Model) imported")
    except ImportError as e:
        print(f"✗ Models import failed: {e}")
    
    # Test layers
    try:
        if working_method == "Direct Keras":
            from keras.layers import Dense, Dropout, BatchNormalization, Input
        elif working_method == "TensorFlow Keras":
            from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
        else:
            from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
        print("✓ Layers (Dense, Dropout, etc.) imported")
    except ImportError as e:
        print(f"✗ Layers import failed: {e}")
    
    # Test optimizers
    try:
        if working_method == "Direct Keras":
            from keras.optimizers import Adam, RMSprop, SGD
        elif working_method == "TensorFlow Keras":
            from tensorflow.keras.optimizers import Adam, RMSprop, SGD
        else:
            from tensorflow.keras.optimizers import Adam, RMSprop, SGD
        print("✓ Optimizers (Adam, RMSprop, SGD) imported")
    except ImportError as e:
        print(f"✗ Optimizers import failed: {e}")
    
    # Test regularizers
    try:
        if working_method == "Direct Keras":
            from keras.regularizers import l1, l2, l1_l2
        elif working_method == "TensorFlow Keras":
            from tensorflow.keras.regularizers import l1, l2, l1_l2
        else:
            from tensorflow.keras.regularizers import l1, l2, l1_l2
        print("✓ Regularizers (l1, l2, l1_l2) imported")
    except ImportError as e:
        print(f"✗ Regularizers import failed: {e}")
    
    # Test callbacks
    try:
        if working_method == "Direct Keras":
            from keras.callbacks import EarlyStopping, ReduceLROnPlateau
        elif working_method == "TensorFlow Keras":
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        else:
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        print("✓ Callbacks (EarlyStopping, ReduceLROnPlateau) imported")
    except ImportError as e:
        print(f"✗ Callbacks import failed: {e}")
    
    # Test basic model creation
    print("\n=== Basic Model Creation Test ===")
    try:
        model = Sequential([
            Dense(64, activation='relu', input_shape=(10,)),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        print("✓ Basic Sequential model created and compiled successfully")
        print(f"  Model has {model.count_params()} parameters")
        
        # Test model prediction with dummy data
        import numpy as np
        dummy_data = np.random.random((5, 10))
        predictions = model.predict(dummy_data, verbose=0)
        print(f"✓ Model prediction test successful: shape {predictions.shape}")
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False
    
    print("\n=== GPU Test ===")
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ {len(gpus)} GPU(s) detected:")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
        else:
            print("ℹ No GPU detected, using CPU")
    except Exception as e:
        print(f"⚠ GPU check failed: {e}")
    
    return True

def generate_import_code(working_method):
    """Generate the correct import code based on working method"""
    print(f"\n=== Recommended Import Code ===")
    
    if working_method == "Direct Keras":
        print("""
# For TensorFlow 3.x style (recommended if available)
try:
    import tensorflow as tf
    import keras
    from keras.models import Sequential, Model
    from keras.layers import Dense, Dropout, BatchNormalization, Input
    from keras.optimizers import Adam, RMSprop, SGD
    from keras.regularizers import l1, l2, l1_l2
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
    TF_VERSION = "3.x"
except ImportError:
    TF_AVAILABLE = False
""")
    else:
        print("""
# For TensorFlow 2.x style (fallback)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.optimizers import Adam, RMSprop, SGD
    from tensorflow.keras.regularizers import l1, l2, l1_l2
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
    TF_VERSION = "2.x"
except ImportError:
    TF_AVAILABLE = False
""")

def main():
    """Main test function"""
    print("TensorFlow/Keras Compatibility Check")
    print("====================================")
    
    success = test_tensorflow_imports()
    
    if success:
        print("\n🎉 TensorFlow/Keras compatibility check PASSED!")
        print("Your environment should work with the neural network builders.")
    else:
        print("\n❌ TensorFlow/Keras compatibility check FAILED!")
        print("Please install or fix TensorFlow:")
        print("  pip install tensorflow")
        print("  or")
        print("  pip install tensorflow-cpu")

if __name__ == "__main__":
    main()