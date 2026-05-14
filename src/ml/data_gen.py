import os
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple

class GraphGenerator:
    """
    Generates synthetic graph images for training the Graph Classifier.
    Categories: good, passable, bad, none.
    """
    
    def __init__(self, base_dir: str = 'data/dataset'):
        self.base_dir = base_dir
        self.labels = ['good', 'passable', 'bad', 'none']
        self._setup_dirs()

    def _setup_dirs(self):
        """Creates the necessary directory structure."""
        for label in self.labels:
            os.makedirs(os.path.join(self.base_dir, label), exist_ok=True)

    @staticmethod
    def _random_color() -> Tuple[float, float, float]:
        return tuple(np.random.rand(3,))

    @staticmethod
    def _random_style() -> str:
        return random.choice(['-', '--', '-.', ':'])

    def gen_forecast(self, label: str, idx: int):
        """Generates a standard line forecast with varying noise levels."""
        folder = os.path.join(self.base_dir, label)
        t = np.linspace(0, 10, 100)
        base = 1000 * (np.sin(t) + 0.3 * np.cos(2 * t) + 1.5)
        
        noise_map = {'good': 50, 'passable': 200, 'bad': 500}
        noise_level = noise_map.get(label, 200)
        
        pred = base + np.random.normal(0, noise_level, size=base.shape)
        
        plt.figure(figsize=(6, 3), dpi=100)
        plt.plot(t, base, color='black', linewidth=2, label='Actual')
        plt.plot(t, pred, color='orange', linestyle='--', linewidth=2, label='Forecast')
        plt.axis('off')
        plt.tight_layout()
        
        save_path = os.path.join(folder, f"{label}_forecast_{idx}.png")
        plt.savefig(save_path)
        plt.close()

    def gen_excel_style(self, label: str, idx: int):
        """Generates an Excel-like sales forecast graph."""
        folder = os.path.join(self.base_dir, label)
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        t = np.arange(12)
        actual = np.array([100, 110, 120, 135, 150, 140, 145, 160, 170, 180, 190, 200])
        
        noise_map = {'good': 5, 'passable': 20, 'bad': 70}
        noise_level = noise_map.get(label, 20)
        
        forecast = actual + np.random.normal(0, noise_level, size=actual.shape)
        
        plt.figure(figsize=(6, 3.5), dpi=100)
        plt.plot(t, actual, color=self._random_color(), linewidth=2, label='Actual Sales')
        plt.plot(t, forecast, color=self._random_color(), linestyle=self._random_style(), linewidth=2, label='Forecast')
        plt.xticks(t, months)
        plt.title('Monthly Sales Forecast Output')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(folder, f"{label}_excel_{idx}.png")
        plt.savefig(save_path)
        plt.close()

    def gen_none(self, idx: int):
        """Generates graphs that are NOT forecasting graphs (noise/distractors)."""
        folder = os.path.join(self.base_dir, 'none')
        plt.figure(figsize=(6, 3), dpi=100)
        kind = random.choice(['scatter', 'bar', 'text', 'pie', 'blank'])
        
        if kind == 'scatter':
            plt.scatter(np.random.rand(50), np.random.rand(50), c=self._random_color())
        elif kind == 'bar':
            plt.bar(np.arange(5), np.random.rand(5) * 100, color=self._random_color())
        elif kind == 'text':
            plt.text(0.5, 0.5, "DATA NOT AVAILABLE", ha='center', va='center', fontsize=14, fontweight='bold')
        elif kind == 'pie':
            plt.pie(np.random.rand(3), labels=['A', 'B', 'C'], colors=[self._random_color() for _ in range(3)])
        
        plt.axis('off')
        plt.tight_layout()
        
        save_path = os.path.join(folder, f"none_{idx}.png")
        plt.savefig(save_path)
        plt.close()

    def generate_all(self, count_per_label: int = 70):
        """Generates a complete dataset."""
        print(f"🚀 Generating dataset at: {self.base_dir}")
        for label in ['good', 'passable', 'bad']:
            print(f"  -> Generating '{label}' samples...")
            for i in range(count_per_label // 2):
                self.gen_forecast(label, i)
                self.gen_excel_style(label, i)
        
        print(f"  -> Generating 'none' samples...")
        for i in range(count_per_label):
            self.gen_none(i)
        
        print("✅ Dataset generation complete.")

if __name__ == "__main__":
    gen = GraphGenerator()
    gen.generate_all()
