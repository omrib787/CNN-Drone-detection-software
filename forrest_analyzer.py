import os
import sys
import torch
import torchvision
from torchvision import transforms
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from tkinter import simpledialog
import joblib
from torchvision import models

class RandomForestImageAnalyzer:
    def __init__(self, root):
        """
        GUI tool to analyze images with a trained RandomForest model
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Random Forest Image Analyzer")
        self.root.geometry("1200x800")
        
        # Variables
        self.model = None
        self.model_path = None
        self.image_dir = None
        self.image_files = []
        self.current_image_idx = -1
        self.classes = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Feature extractor using ResNet50
        self.feature_extractor = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.feature_extractor = torch.nn.Sequential(*list(self.feature_extractor.children())[:-1])
        self.feature_extractor.eval()
        self.feature_extractor.to(self.device)
        
        # Create GUI
        self.create_gui()
    
    def create_gui(self):
        """Create the GUI layout"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top frame for controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Model selection button
        ttk.Label(control_frame, text="Random Forest Model:").pack(side=tk.LEFT, padx=5)
        self.model_label = ttk.Label(control_frame, text="No model selected", width=40)
        self.model_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=5)
        
        # Test directory selection
        ttk.Button(control_frame, text="Select Test Directory", command=self.select_test_dir).pack(side=tk.LEFT, padx=20)
        
        # Navigation frame
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(nav_frame, text="Previous Image", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next Image", command=self.next_image).pack(side=tk.LEFT, padx=5)
        self.image_counter = ttk.Label(nav_frame, text="No images loaded")
        self.image_counter.pack(side=tk.LEFT, padx=20)
        
        # Create a frame to hold the image and prediction results
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left side: Image display
        image_frame = ttk.LabelFrame(content_frame, text="Image")
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.image_label = ttk.Label(image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Right side: Prediction results
        results_frame = ttk.LabelFrame(content_frame, text="Prediction Results")
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Prediction text area
        self.prediction_text = tk.Text(results_frame, height=10, width=40)
        self.prediction_text.pack(fill=tk.X, padx=10, pady=10)
        
        # Prediction chart
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, results_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
    
    def load_model(self):
        """Load a trained RandomForest model"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select RandomForest Model File",
                filetypes=[("Scikit-learn Models", "*.pkl")]
            )
            
            if not file_path:
                return
            
            self.status_var.set("Loading model...")
            self.root.update_idletasks()
            
            # Load the trained model
            self.model = joblib.load(file_path)
            
            # Ask the user to provide the classes
            class_str = simpledialog.askstring(
                "Input Classes", 
                "Enter class names separated by commas, in the same order as the model was trained:"
            )
            if class_str:
                self.classes = [c.strip() for c in class_str.split(',')]
            else:
                messagebox.showerror("Error", "Classes are required to interpret the model's predictions.")
                return

            self.model_path = file_path
            self.model_label.config(text=os.path.basename(file_path))
            
            self.status_var.set(f"Model loaded successfully. Classes: {', '.join(self.classes)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_var.set("Error loading model")
            import traceback
            traceback.print_exc()

    def select_test_dir(self):
        """Select a directory containing test images"""
        if not self.model:
            messagebox.showwarning("Warning", "Please load a model first")
            return
        
        dir_path = filedialog.askdirectory(
            title="Select Test Images Directory"
        )
        
        if not dir_path:
            return
        
        self.image_dir = dir_path
        self.load_image_files()
    
    def load_image_files(self):
        """Load all image files from the selected directory"""
        self.image_files = []
        
        if not os.path.exists(self.image_dir):
            messagebox.showerror("Error", "Selected directory does not exist")
            return
        
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.image_files.append(os.path.join(root, file))
        
        if not self.image_files:
            messagebox.showinfo("Info", "No image files found in the selected directory")
            self.image_counter.config(text="No images loaded")
            return
        
        self.current_image_idx = 0
        self.image_counter.config(text=f"Image 1 of {len(self.image_files)}")
        self.status_var.set(f"Loaded {len(self.image_files)} images from {self.image_dir}")
        
        self.load_current_image()
    
    def load_current_image(self):
        """Load and analyze the current image"""
        if not self.image_files or self.current_image_idx < 0 or self.current_image_idx >= len(self.image_files):
            return
        
        image_path = self.image_files[self.current_image_idx]
        self.analyze_image(image_path)
    
    def next_image(self):
        """Go to the next image"""
        if not self.image_files:
            return
        
        self.current_image_idx = (self.current_image_idx + 1) % len(self.image_files)
        self.image_counter.config(text=f"Image {self.current_image_idx + 1} of {len(self.image_files)}")
        self.load_current_image()
    
    def prev_image(self):
        """Go to the previous image"""
        if not self.image_files:
            return
        
        self.current_image_idx = (self.current_image_idx - 1) % len(self.image_files)
        self.image_counter.config(text=f"Image {self.current_image_idx + 1} of {len(self.image_files)}")
        self.load_current_image()
    
    def analyze_image(self, image_path):
        """Analyze the current image with the loaded model"""
        try:
            self.status_var.set(f"Analyzing image: {os.path.basename(image_path)}...")
            self.root.update_idletasks()
            
            # Load and display the image
            img = Image.open(image_path).convert("RGB")
            img_display = img.copy()
            if img_display.width > 500 or img_display.height > 500:
                img_display.thumbnail((500, 500))
            
            img_tk = ImageTk.PhotoImage(img_display)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk
            
            # Preprocess image and extract features
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = preprocess(img)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.feature_extractor(input_batch).squeeze().cpu().numpy()

            # Get predictions from the RandomForest model
            probabilities = self.model.predict_proba(features.reshape(1, -1))[0]
            predicted_idx = np.argmax(probabilities)
            predicted_class = self.classes[predicted_idx]
            
            # Display results
            self.display_prediction_results(predicted_class, probabilities)
            
            true_class = "Unknown"
            for cls in self.classes:
                if cls.lower() in image_path.lower():
                    true_class = cls
                    break
            
            self.status_var.set(f"Analysis complete. True class (from path): {true_class}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze image: {str(e)}")
            self.status_var.set("Error analyzing image")
            import traceback
            traceback.print_exc()
    
    def display_prediction_results(self, predicted_class, probabilities):
        """Display prediction results in the GUI"""
        self.prediction_text.delete(1.0, tk.END)
        self.prediction_text.insert(tk.END, f"Predicted class: {predicted_class}\n\n")
        self.prediction_text.insert(tk.END, "Class probabilities:\n")
        
        for i, cls in enumerate(self.classes):
            prob = probabilities[i]
            self.prediction_text.insert(tk.END, f"{cls}: {prob*100:.2f}%\n")
        
        predicted_idx = self.classes.index(predicted_class)
        start_pos = self.prediction_text.search(predicted_class, "3.0", tk.END)
        if start_pos:
            line = start_pos.split('.')[0]
            self.prediction_text.tag_add("highlight", f"{line}.0", f"{line}.end")
            self.prediction_text.tag_config("highlight", background="yellow")
        
        self.ax.clear()
        y_pos = np.arange(len(self.classes))
        probs_to_plot = [p * 100 for p in probabilities]
        
        bars = self.ax.bar(y_pos, probs_to_plot, align='center', alpha=0.7)
        bars[predicted_idx].set_color('red')
        
        self.ax.set_xticks(y_pos)
        self.ax.set_xticklabels(self.classes, rotation=45, ha='right')
        self.ax.set_ylabel('Probability (%)')
        self.ax.set_title('Random Forest Class Probabilities')
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{probs_to_plot[i]:.1f}%', ha='center', va='bottom')
        
        self.figure.tight_layout()
        self.canvas.draw()

def main():
    root = tk.Tk()
    app = RandomForestImageAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main()