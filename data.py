import os, numpy as np, matplotlib.pyplot as plt, random

def random_color(): return tuple(np.random.rand(3,))
def random_style(): return random.choice(['-', '--', '-.', ':'])
base_dir = 'graph_classifier_dataset_final'
for label in ['good', 'passable', 'bad', 'none']:
    os.makedirs(os.path.join(base_dir, label), exist_ok=True)

def gen_forecast(label, idx, folder):
    t = np.linspace(0, 10, 100)
    base = 1000 * (np.sin(t)+0.3*np.cos(2*t)+1.5)
    noise = {'good':50, 'passable':200, 'bad':500}[label]
    pred = base + np.random.normal(0, noise, size=base.shape)
    plt.figure(figsize=(6,3))
    plt.plot(t, base, color='black', linewidth=2)
    plt.plot(t, pred, color='orange', linestyle='--', linewidth=2)
    plt.axis('off'); plt.tight_layout()
    plt.savefig(f"{folder}/{label}_classic_{idx}.png", dpi=100); plt.close()

def gen_excel_style(label, idx, folder):
    t = np.arange(12)
    actual = np.array([100,110,120,135,150,140,145,160,170,180,190,200])
    noise = {'good':5, 'passable':20, 'bad':70}[label]
    forecast = actual + np.random.normal(0, noise, size=actual.shape)
    plt.figure(figsize=(6,3.5))
    plt.plot(t, actual, color=random_color(), linewidth=2, label='Actual')
    plt.plot(t, forecast, color=random_color(), linestyle=random_style(), linewidth=2, label='Forecast')
    plt.xticks(t, ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    plt.title('Sales Forecast'); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{folder}/{label}_excel_{idx}.png", dpi=100); plt.close()

def gen_none(idx, folder):
    plt.figure(figsize=(6, 3))
    kind = random.choice(['scatter','bar','text','pie','blank'])
    if kind=='scatter':
        plt.scatter(np.random.rand(50), np.random.rand(50), c=np.random.rand(3,))
    elif kind=='bar':
        plt.bar(np.arange(5), np.random.rand(5)*100, color=np.random.rand(3,))
    elif kind=='text':
        plt.text(0.5, 0.5, "Forecast Unavailable", ha='center', fontsize=14)
    elif kind=='pie':
        plt.pie(np.random.rand(3), labels=['A','B','C'], colors=[np.random.rand(3,) for _ in range(3)])
    plt.axis('off'); plt.tight_layout()
    plt.savefig(f"{folder}/none_{idx}.png", dpi=100); plt.close()

for label in ['good','passable','bad']:
    folder = os.path.join(base_dir, label)
    for i in range(50): gen_forecast(label, i, folder)
    for i in range(20): gen_excel_style(label, i, folder)

for i in range(50):
    gen_none(i, os.path.join(base_dir, 'none'))

print("✅ Final dataset with 4 categories (70/70/70/50) created at:", base_dir)
