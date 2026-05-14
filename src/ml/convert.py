import os
import argparse
import tensorflowjs as tfjs
from tensorflow.keras.models import load_model

def convert_model(model_path='models/graph_classifier.h5', output_dir='web/public/model'):
    """
    Converts a saved Keras model to TensorFlow.js format.
    """
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"📦 Loading model from {model_path}...")
    try:
        model = load_model(model_path)
        print(f"🚀 Converting to TensorFlow.js format in {output_dir}...")
        tfjs.converters.save_keras_model(model, output_dir)
        print("✅ Conversion successful!")
    except Exception as e:
        print(f"❌ Conversion failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Keras model to TF.js")
    parser.add_argument("--model", type=str, default="models/graph_classifier.h5", help="Path to input .h5 model")
    parser.add_argument("--out", type=str, default="web/public/model", help="Output directory for TF.js files")
    
    args = parser.parse_args()
    convert_model(args.model, args.out)
