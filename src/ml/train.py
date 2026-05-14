import os
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model

def run_training(data_dir='data/dataset', output_path='models/graph_classifier.h5', epochs=10):
    """
    Handles the end-to-end training process.
    """
    img_size = (128, 128)
    batch_size = 16

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset not found at {data_dir}. Run data_gen.py first.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("📊 Loading and Preprocessing Data...")
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    print(f"✅ Data loaded. Found {train_gen.num_classes} classes.")

    model = create_model(input_shape=(*img_size, 3), num_classes=train_gen.num_classes)

    print("🧠 Starting Model Training...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs
    )

    print(f"💾 Saving trained model to {output_path}...")
    model.save(output_path)
    print("✨ Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Graph Classifier Model")
    parser.add_argument("--data", type=str, default="data/dataset", help="Path to dataset")
    parser.add_argument("--out", type=str, default="models/graph_classifier.h5", help="Path to save model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    
    args = parser.parse_args()
    run_training(args.data, args.out, args.epochs)
