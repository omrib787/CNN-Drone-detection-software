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

class ResNetImageAnalyzer:
    def __init__(self, root):
        """
        GUI tool to analyze images with a trained ResNet model without augmentation
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("ResNet Image Analyzer (No Augmentation)")
        self.root.geometry("1200x800")
        
        # Variables
        self.model = None
        self.model_path = None
        self.image_dir = None
        self.image_files = []
        self.current_image_idx = -1
        self.classes = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.softmax = torch.nn.Softmax(dim=1)
        
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
        ttk.Label(control_frame, text="ResNet Model:").pack(side=tk.LEFT, padx=5)
        self.model_label = ttk.Label(control_frame, text="No model selected", width=40)
        self.model_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=5)
        
        # Test directory selection
        ttk.Button(control_frame, text="Select Test Directory", command=self.select_test_dir).pack(side=tk.LEFT, padx=20)
        
        # Single image selection
        ttk.Button(control_frame, text="Select Single Image", command=self.select_single_image).pack(side=tk.LEFT, padx=5)
        
        # Navigation frame
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(nav_frame, text="Previous Image", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next Image", command=self.next_image).pack(side=tk.LEFT, padx=5)
        self.image_counter = ttk.Label(nav_frame, text="No images loaded")
        self.image_counter.pack(side=tk.LEFT, padx=20)
        
        # File path display
        self.file_path_var = tk.StringVar()
        self.file_path_var.set("No image selected")
        ttk.Label(nav_frame, textvariable=self.file_path_var, width=60).pack(side=tk.LEFT, padx=5)
        
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
        """Load a trained ResNet model"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select ResNet Model File",
                filetypes=[("PyTorch Models", "*.pth")]
            )
            
            if not file_path:
                return
            
            self.status_var.set("Loading model...")
            self.root.update_idletasks()
            
            # Look for classes.txt in the same directory
            classes_path = os.path.join(os.path.dirname(file_path), "classes.txt")
            
            # Try to load classes from a text file if it exists
            if os.path.exists(classes_path):
                with open(classes_path, 'r') as f:
                    self.classes = [line.strip() for line in f.readlines()]
            else:
                # Try to find classes in the model directory
                model_dir = os.path.dirname(file_path)
                result_dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
                
                for d in result_dirs:
                    class_file = os.path.join(model_dir, d, "classes.txt")
                    if os.path.exists(class_file):
                        with open(class_file, 'r') as f:
                            self.classes = [line.strip() for line in f.readlines()]
                        break
                
                # If still not found, use default classes
                if not self.classes:
                    from tkinter import simpledialog
                    # Ask the user to provide the classes
                    class_input = messagebox.askquestion(
                        "Classes Required", 
                        "No classes file found. Do you want to specify the classes manually?"
                    )
                    
                    if class_input == 'yes':
                        class_str = simpledialog.askstring(
                            "Input Classes", 
                            "Enter class names separated by commas:"
                        )
                        if class_str:
                            self.classes = [c.strip() for c in class_str.split(',')]
                        else:
                            # Default classes if user cancels input
                            self.classes = ["bird", "drone", "plane", "UAV"]
                    else:
                        # Default classes if user says no
                        self.classes = ["bird", "drone", "plane", "UAV"]
            
            # Setup ResNet model
            self.model = self.setup_model(len(self.classes))
            
            # Load the trained weights
            try:
                checkpoint = torch.load(file_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                else:
                    self.model.load_state_dict(checkpoint)
                
                # Set model to evaluation mode
                self.model.eval()
                
                self.model_path = file_path
                self.model_label.config(text=os.path.basename(file_path))
                
                # Save classes to a file for future use
                with open(os.path.join(os.path.dirname(file_path), "classes.txt"), 'w') as f:
                    for cls in self.classes:
                        f.write(f"{cls}\n")
                
                self.status_var.set(f"Model loaded successfully. Classes: {', '.join(self.classes)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model weights: {str(e)}")
                self.status_var.set("Error loading model weights")
                raise
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_var.set("Error loading model")
            import traceback
            traceback.print_exc()
    
    def setup_model(self, num_classes):
        """Set up a ResNet model with the correct number of classes"""
        model = torchvision.models.resnet50(weights=None)
        
        # Modify the final fully connected layer for our number of classes
        num_features = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),  # Using a standard dropout rate
            torch.nn.Linear(num_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes)
        )
        
        # Move to device
        model = model.to(self.device)
        return model
    
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
    
    def select_single_image(self):
        """Select a single image to analyze"""
        if not self.model:
            messagebox.showwarning("Warning", "Please load a model first")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not file_path:
            return
        
        # Reset navigation and set this as the only image
        self.image_files = [file_path]
        self.current_image_idx = 0
        self.image_counter.config(text="Image 1 of 1")
        self.analyze_image(file_path)
    
    def load_image_files(self):
        """Load all image files from the selected directory"""
        self.image_files = []
        
        # Check if directory exists
        if not os.path.exists(self.image_dir):
            messagebox.showerror("Error", "Selected directory does not exist")
            return
        
        # Get all image files (including subdirectories)
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
        
        # Load the first image
        self.load_current_image()
    
    def load_current_image(self):
        """Load and analyze the current image"""
        if not self.image_files or self.current_image_idx < 0 or self.current_image_idx >= len(self.image_files):
            return
        
        image_path = self.image_files[self.current_image_idx]
        self.file_path_var.set(image_path)
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
            img = Image.open(image_path)
            img_display = img.copy()
            if img_display.width > 500 or img_display.height > 500:
                img_display.thumbnail((500, 500))
            
            img_tk = ImageTk.PhotoImage(img_display)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk  # Keep a reference
            
            # Preprocess image for model - NO AUGMENTATION, just resize and normalize
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            # Convert RGBA to RGB if necessary
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            elif img.mode == 'L':  # Handle grayscale too
                img = img.convert('RGB')

            input_tensor = preprocess(img)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                output = self.model(input_batch)
                probabilities = self.softmax(output)[0]
                
            # Get predicted class
            _, predicted_idx = torch.max(output, 1)
            predicted_class = self.classes[predicted_idx.item()]
            
            # Convert to numpy for display
            probabilities_np = probabilities.cpu().numpy()
            
            # Display results
            self.display_prediction_results(predicted_class, probabilities_np)
            
            # Extract true class from the file path if possible
            true_class = "Unknown"
            for cls in self.classes:
                if cls.lower() in image_path.lower():
                    true_class = cls
                    break
            
            self.status_var.set(f"Analysis complete. True class (estimated from path): {true_class}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze image: {str(e)}")
            self.status_var.set("Error analyzing image")
            import traceback
            traceback.print_exc()
    
    def display_prediction_results(self, predicted_class, probabilities):
        """Display prediction results in the GUI"""
        # Clear previous results
        self.prediction_text.delete(1.0, tk.END)
        
        # Add prediction text
        self.prediction_text.insert(tk.END, f"Predicted class: {predicted_class}\n\n")
        self.prediction_text.insert(tk.END, "Class probabilities:\n")
        
        for i, cls in enumerate(self.classes):
            prob = probabilities[i] if i < len(probabilities) else 0
            self.prediction_text.insert(tk.END, f"{cls}: {prob*100:.2f}%\n")
        
        # Highlight the predicted class line
        predicted_idx = self.classes.index(predicted_class)
        start_pos = self.prediction_text.search(predicted_class, "3.0", tk.END)
        if start_pos:
            line = start_pos.split('.')[0]
            self.prediction_text.tag_add("highlight", f"{line}.0", f"{line}.end")
            self.prediction_text.tag_config("highlight", background="yellow")
        
        # Plot bar chart
        self.ax.clear()
        y_pos = np.arange(len(self.classes))
        
        # Ensure probabilities array matches classes length
        probs_to_plot = []
        for i in range(len(self.classes)):
            if i < len(probabilities):
                probs_to_plot.append(probabilities[i] * 100)
            else:
                probs_to_plot.append(0)
        
        bars = self.ax.bar(y_pos, probs_to_plot, align='center', alpha=0.7)
        
        # Highlight the predicted class
        predicted_idx = self.classes.index(predicted_class)
        if predicted_idx < len(bars):
            bars[predicted_idx].set_color('red')
        
        self.ax.set_xticks(y_pos)
        self.ax.set_xticklabels(self.classes, rotation=45, ha='right')
        self.ax.set_ylabel('Probability (%)')
        self.ax.set_title('ResNet Class Probabilities')
        
        # Add values on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{probs_to_plot[i]:.1f}%', ha='center', va='bottom')
        
        self.figure.tight_layout()
        self.canvas.draw()

def main():
    from tkinter import simpledialog
    
    root = tk.Tk()
    app = ResNetImageAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main()