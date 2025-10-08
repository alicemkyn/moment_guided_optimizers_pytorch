import matplotlib.pyplot as plt
import os

def show_all_results():
    res_dir = "results"
    files = [f for f in os.listdir(res_dir) if f.endswith(".png")]
    for f in sorted(files):
        img = plt.imread(os.path.join(res_dir, f))
        plt.figure(figsize=(5,5))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f)
        plt.show()

if __name__ == "__main__":
    show_all_results()
