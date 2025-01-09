#This is a Python Program created to detect whether an image is AI generated. 

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import os

def analyze_image(image_path):
    """
    Analyzes the image for AI-generated qualities using refined heuristics:
    - Checks for overly smooth textures (low variance in gradients).
    - Detects asymmetries in the image (mirroring differences).
    - Evaluates edge consistency and uniformity using Canny edge detection.
    - Adds an analysis of unnatural facial structures or artifacts (experimental).
    """
    try:
        # Load and preprocess the image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Check for smooth textures using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        smooth_texture = "Yes" if laplacian_var < 50 else "No"

        # Check for asymmetries by mirroring
        h, w = gray.shape
        left_half = gray[:, :w // 2]
        right_half = cv2.flip(gray[:, w // 2:], 1)
        asymmetry_score = np.sum((left_half - right_half) ** 2) / (h * (w // 2))
        asymmetric = "Yes" if asymmetry_score > 1000 else "No"

        # Evaluate edge consistency using Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / (h * w)
        consistent_edges = "No" if edge_density < 0.015 or edge_density > 0.08 else "Yes"

        # Check for unnatural features (e.g., artifacts or missing details)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_area_ratio = sum(cv2.contourArea(c) for c in contours) / (h * w)
        unnatural_features = "Yes" if contour_area_ratio > 0.01 else "No"

        # Adjust AI detection criteria
        ai_generated = (
            "Yes" if (
                smooth_texture == "Yes" or
                consistent_edges == "No" or
                unnatural_features == "Yes"
            ) else "No"
        )

        # Combine results into a dictionary with raw measurements
        analysis_results = {
            "Hyperrealistic/Synthetic Aesthetics": f"{smooth_texture} (Laplacian Variance: {laplacian_var:.2f})",
            "Overly Consistent Details": f"{consistent_edges} (Edge Density: {edge_density:.4f})",
            "Asymmetric/Distorted Features": f"{asymmetric} (Asymmetry Score: {asymmetry_score:.2f})",
            "Unnatural Features/Artifacts": f"{unnatural_features} (Contour Area Ratio: {contour_area_ratio:.4f})",
            "AI Generated?": ai_generated,
        }
        return analysis_results
    except Exception as e:
        return {"Error": f"Failed to analyze image: {e}"}

class ImageEvaluatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Detector")
        self.image_counter = 0

        # Create UI Elements
        self.image_label = tk.Label(self.root, text="No Image Loaded", font=("Arial", 14))
        self.image_label.pack(pady=10)

        self.canvas = tk.Canvas(self.root, width=400, height=400)
        self.canvas.pack()

        self.result_text = tk.Text(self.root, height=15, width=60, state="disabled")
        self.result_text.pack(pady=10)

        self.select_button = tk.Button(self.root, text="Select Image", command=self.load_image)
        self.select_button.pack(pady=5)

        self.next_button = tk.Button(self.root, text="Evaluate Another Image", command=self.reset, state="disabled")
        self.next_button.pack(pady=5)

    def load_image(self):
        """Allows the user to select an image file to evaluate."""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if not file_path:
            return

        try:
            self.image = Image.open(file_path)
            self.image.thumbnail((400, 400))
            self.photo = ImageTk.PhotoImage(self.image)

            self.canvas.create_image(200, 200, image=self.photo)
            self.image_label.config(text=f"Image {self.image_counter + 1}")

            # Perform analysis and display results
            self.display_results(file_path)

            self.select_button.config(state="disabled")
            self.next_button.config(state="normal")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def display_results(self, file_path):
        """Analyzes the image and displays the results."""
        results = analyze_image(file_path)

        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)

        self.result_text.insert(tk.END, f"Analysis Results for Image {self.image_counter + 1}\n")
        self.result_text.insert(tk.END, f"Image Path: {os.path.basename(file_path)}\n\n")
        for key, value in results.items():
            self.result_text.insert(tk.END, f"{key}: {value}\n")

        self.result_text.config(state="disabled")

    def reset(self):
        """Resets the application for a new image evaluation."""
        self.image_counter += 1
        self.image_label.config(text="No Image Loaded")
        self.canvas.delete("all")

        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state="disabled")

        self.select_button.config(state="normal")
        self.next_button.config(state="disabled")

# Main Application Execution
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEvaluatorApp(root)
    root.mainloop()
