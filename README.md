# GraphClassifier

# 🧠 Prediction Analyzer - Forecast Graph Classifier using CNN

This project presents a Convolutional Neural Network (CNN) that classifies forecast graph images into four categories: `good`, `passable`, `bad`, and `none`. It includes automated dataset generation, model training via Jupyter Notebook, and guidelines to extend the dataset with new images.

---

## 📂 Project Structure

```
├── data.py                       # Script to generate synthetic training graphs
├── Graph_Classifier.ipynb  # Jupyter Notebook containing full CNN model pipeline
├── retail_store_inventory.csv   # Sample unrelated data (not used in model pipeline)
├── graph_classifier_dataset_final/ (this is generated using data.py)
│   ├── good/                    # Graphs with low noise (high prediction quality)
│   ├── passable/                # Graphs with moderate noise (acceptable quality)
│   ├── bad/                     # Graphs with high noise (poor prediction quality)
│   └── none/                    # Random/non-forecast visuals (e.g., pie, scatter)
```

---

## 🧪 Dataset Generation (`data.py`)

Run `data.py` to generate a synthetic dataset categorized into four classes. Each class contains graph images created using `matplotlib`.

* **Forecast-style graphs:** Sine-wave patterns with added noise
* **Excel-style graphs:** Simulated monthly sales forecasts
* **None graphs:** Pie, scatter, and other unrelated charts

### ➕ Add More Graphs

You can manually add new `.png` graph images to the relevant folders inside `graph_classifier_dataset_final/`:

```
graph_classifier_dataset_final/
    ├── good/        # Add new 'good' prediction graphs here
    ├── passable/    # Add 'passable' ones here
    ├── bad/         # Add poorly generated graphs here
    └── none/        # Add irrelevant visuals here
```

Ensure added images are clear and meaningful. The model will train based on these.

---

## 🧠 Model Overview (`Graph_Classifier_CNN_Cleaned.ipynb`)

This notebook covers:

* Image preprocessing and dataset loading
* CNN architecture creation using Keras
* Model training and validation
* Final evaluation and prediction examples

⚠️ Before running, make sure `graph_classifier_dataset_final/` is in the working directory.

---

## ⚙️ Usage Guide

1. Clone the repository:

```bash
git clone https://github.com/SakshiMalhotra18/GraphClassifier.git
cd GraphClassifier
```

2. Run dataset generator:

```bash
python data.py
```

3. Launch the notebook and train the model:

```bash
jupyter notebook Graph_Classifier.ipynb
```

---

## 📦 Dependencies

Ensure Python 3.10+ is installed. Required libraries:

```
matplotlib
numpy
tensorflow
scikit-learn
Pillow
jupyter
```

Install them with:

```bash
pip install -r requirements.txt
```

To generate `requirements.txt`:

```bash
pip freeze > requirements.txt
```

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

You're welcome to fork this project and contribute! Open a pull request with improvements, new graphs, or model tweaks.

---

## 📌 Notes

* This is a synthetic ML project meant for prototyping or educational use.
* Real-world extension would require domain-specific graphs and fine-tuned labeling.
* The system is compatible with any tool that generates `.png` forecast visuals.

---

Made with 💡 by Sakshi Malhotra
