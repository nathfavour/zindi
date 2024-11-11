# train.py
from model import RunwayDetector
import glob
import pandas as pd

def main():
    # Load data paths
    optical_paths = glob.glob('data/optical/*.tif')
    sar_paths = glob.glob('data/sar/*.tif')
    labels = pd.read_csv('data/labels.csv')
    
    detector = RunwayDetector()
    
    # Preprocess all images
    optical_images = []
    sar_images = []
    for opt_path, sar_path in zip(optical_paths, sar_paths):
        opt_img, sar_img = detector.preprocess_image(opt_path, sar_path)
        optical_images.append(opt_img)
        sar_images.append(sar_img)
    
    # Train model
    detector.train(np.array(optical_images), np.array(sar_images), labels['runway'].values)

if __name__ == '__main__':
    main()