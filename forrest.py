# random_forest_model.py
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime   
import inspect
import shutil

    

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_min = -np.Inf
        self.delta = delta
        self.path = path
        
    def __call__(self, val_acc, model):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0
            
    def save_checkpoint(self, val_acc, model):
        if self.verbose:
            print(f'Validation accuracy increased ({self.val_acc_min:.6f} --> {val_acc:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_acc_min = val_acc

class ImageClassifier:
    def __init__(self, data_dir, image_size=224, batch_size=32, num_workers=4, 
                lr=1e-4, weight_decay=1e-4, epochs=10, dropout_rate=0.3, 
                label_smoothing=0.1, lr_decay_factor=0.1, lr_patience=3,
                early_stop_patience=5, device=None):
        print("CUDA available:", torch.cuda.is_available())

        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dropout_rate = dropout_rate
        self.label_smoothing = label_smoothing
        self.lr_decay_factor = lr_decay_factor  # New parameter
        self.lr_patience = lr_patience          # New parameter
        self.early_stop_patience = early_stop_patience  # New parameter
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.classes = None
        self.train_loader = None
        self.val_loader = None
        self.model = None
        self.results_dir = None
        self.training_log = []

    def create_results_directory(self):
        """Create a timestamped results directory"""
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        self.results_dir = os.path.join("results", f"run_{timestamp}")
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"Results will be saved to: {self.results_dir}")

    def save_code_to_file(self):
        """Save the current code to a text file in results directory"""
        current_file = inspect.getfile(inspect.currentframe())
        dest_file = os.path.join(self.results_dir, "model_code.txt")
        shutil.copy2(current_file, dest_file)
        print(f"Code saved to: {dest_file}")

    def save_training_log(self):
        """Save training log to text file with aligned columns"""
        log_file = os.path.join(self.results_dir, "training_log.txt")
        
        # Determine column widths
        epoch_width = len("Epoch")
        loss_width = 10  # Fixed width for loss values (enough for 0.0000)
        acc_width = 8    # Fixed width for accuracy values (enough for 100.00)
        
        with open(log_file, 'w') as f:
            # Header with fixed spacing
            f.write(f"{'Epoch':<{epoch_width}}  {'Train Loss':<{loss_width}}  {'Train Acc':<{acc_width}}  "
                    f"{'Val Loss':<{loss_width}}  {'Val Acc':<{acc_width}}\n")
            
            # Data rows with consistent spacing
            for epoch, log in enumerate(self.training_log):
                f.write(f"{epoch + 1:<{epoch_width}}  "
                        f"{log['train_loss']:>{loss_width}.4f}  "
                        f"{log['train_acc']:>{acc_width}.2f}  "
                        f"{log['val_loss']:>{loss_width}.4f}  "
                        f"{log['val_acc']:>{acc_width}.2f}\n")

    def setup_data_transforms(self):
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return train_transform, val_transform
    
    def load_datasets(self):
        train_transform, val_transform = self.setup_data_transforms()
        full_dataset = datasets.ImageFolder(self.data_dir, transform=train_transform)
        self.classes = full_dataset.classes
        print(f"Found {len(self.classes)} classes: {self.classes}")
        
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        val_dataset.dataset.transform = val_transform
        
        return train_dataset, val_dataset
    
    def setup_data_loaders(self):
        train_dataset, val_dataset = self.load_datasets()
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True  # This helps transfer data to GPU faster
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True  # This helps transfer data to GPU faster
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
    
    def setup_model(self):
        """Initialize and configure the ResNet50 model"""
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Modify the final fully connected layer
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, len(self.classes))
        )
        
        # Explicitly move model to device
        model = model.to(self.device)
        return model
    
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': running_loss / (total / self.batch_size),
                'acc': 100. * correct / total
            })
        
        train_loss = running_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        return train_loss, train_acc
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=self.classes, output_dict=True)
        
        # Return as a tuple with clear structure
        return {
            'val_loss': val_loss,
            'val_acc': val_acc,
            'confusion_matrix': cm,
            'report': report,
            'all_labels': all_labels,
            'all_preds': all_preds
        }
    
    def save_plot(self, fig, filename):
        """Save matplotlib figure to results directory"""
        plot_path = os.path.join(self.results_dir, filename)
        fig.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved to: {plot_path}")
    
    def plot_training_history(self, history):
        epochs = range(1, len(history['train_loss']) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(epochs, history['train_loss'], label='Train Loss')
        ax1.plot(epochs, history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(epochs, history['train_acc'], label='Train Accuracy')
        ax2.plot(epochs, history['val_acc'], label='Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training & Validation Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        self.save_plot(fig, "training_history.png")
    
    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        self.save_plot(plt.gcf(), "confusion_matrix.png")
    
    def plot_metrics(self, report):
        metrics = ['precision', 'recall', 'f1-score']
        x = np.arange(len(self.classes))
        width = 0.25
        
        plt.figure(figsize=(12, 6))
        for i, metric in enumerate(metrics):
            scores = [report[cls][metric] for cls in self.classes]
            plt.bar(x + i*width, scores, width, label=metric)
        
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title('Classification Metrics by Class')
        plt.xticks(x + width, self.classes, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        self.save_plot(plt.gcf(), "class_metrics.png")

    def evaluate_hard_test(self, hard_test_dir):
        """Evaluate model on hard test cases with more robust handling"""
        print(f"\nRunning Hard Test Evaluation on: {hard_test_dir}")
        
        try:
            # Setup transforms for hard test (same as validation)
            hard_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
            ])
            
            # Create dataset
            hard_dataset = datasets.ImageFolder(hard_test_dir, transform=hard_transform)
            if len(hard_dataset) == 0:
                raise ValueError("No images found in hard_test directory")
            
            # Store all filenames before creating loader
            all_files = [s[0] for s in hard_dataset.samples]
            
            # Filter to only include classes we trained on
            valid_samples = []
            for path, class_idx in hard_dataset.samples:
                class_name = hard_dataset.classes[class_idx]
                if class_name in self.classes:
                    valid_samples.append((path, self.classes.index(class_name)))
            
            if not valid_samples:
                raise ValueError("No valid images matching trained classes found in hard_test directory")

            # Create a new dataset with filtered samples
            hard_dataset.samples = valid_samples
            hard_dataset.targets = [s[1] for s in valid_samples]
            
            # Create loader
            hard_loader = DataLoader(
                hard_dataset,
                batch_size=min(self.batch_size, len(hard_dataset)),
                shuffle=False,
                num_workers=min(self.num_workers, 4)
            )
            
            # Evaluate
            self.model.eval()
            all_preds = []
            all_labels = []
            file_indices = []  # Store the original indices
            
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(tqdm(hard_loader, desc="Processing Hard Test Images")):
                    try:
                        inputs = inputs.to(self.device)
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        # Store the indices of files in this batch
                        start_idx = batch_idx * hard_loader.batch_size
                        end_idx = start_idx + len(labels)
                        file_indices.extend(range(start_idx, end_idx))
                    except Exception as e:
                        print(f"Error processing batch: {str(e)}")
                        continue
            
            if not all_preds:
                raise ValueError("No predictions were generated - evaluation failed")
            
            # Map back to original filenames using stored indices
            all_filenames = [all_files[i] for i in file_indices]
                    
            # Generate report
            report = classification_report(
                all_labels, 
                all_preds, 
                target_names=self.classes,
                output_dict=True
            )
            
            # Create detailed report with UTF-8 encoding
            report_file = os.path.join(self.results_dir, "hard_test_report.txt")
            with open(report_file, 'w', encoding='utf-8') as f:
                # Overall metrics
                f.write("="*50 + "\n")
                f.write(f"HARD TEST EVALUATION REPORT ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
                f.write(f"Directory: {hard_test_dir}\n")
                f.write(f"Total Images: {len(all_labels)}\n")
                f.write("="*50 + "\n\n")
                
                # Per-class section
                for class_name in self.classes:
                    # Get all files for this class
                    class_idx = self.classes.index(class_name)
                    class_mask = np.array(all_labels) == class_idx
                    class_files = np.array(all_filenames)[class_mask].tolist()
                    class_preds = np.array(all_preds)[class_mask].tolist()
                    
                    f.write(f"\n{'='*30}\n")
                    f.write(f"CLASS: {class_name.upper()}\n")
                    f.write(f"Images: {len(class_files)}\n")
                    f.write(f"{'='*30}\n\n")
                    
                    # Class metrics
                    f.write(f"Precision: {report[class_name]['precision']:.4f}\n")
                    f.write(f"Recall: {report[class_name]['recall']:.4f}\n")
                    f.write(f"F1-Score: {report[class_name]['f1-score']:.4f}\n")
                    f.write(f"Support: {report[class_name]['support']}\n\n")
                    
                    # Individual image results
                    f.write("IMAGE RESULTS:\n")
                    f.write("-"*20 + "\n")
                    for filename, pred in zip(class_files, class_preds):
                        pred_class = self.classes[pred]
                        correct = "[CORRECT]" if pred == class_idx else "[WRONG]"
                        f.write(f"{correct} {os.path.basename(filename):<30} -> {pred_class}\n")
            
            print(f"Hard test report saved to: {report_file}")
            
            # Save confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            self.plot_confusion_matrix(cm)
            
        except Exception as e:
            print(f"\nError during hard test evaluation: {str(e)}")
            print("Please verify:")
            print(f"1. The hard_test directory exists at: {hard_test_dir}")
            print("2. It contains subdirectories matching your training classes")
            print("3. Each subdirectory contains valid image files")
            print("4. The images are in supported formats (JPG, PNG, etc.)")

    def train(self):
        # Setup results directory
        self.create_results_directory()
        self.save_code_to_file()
        
        # Setup data and model
        self.setup_data_loaders()
        self.model = self.setup_model()
        
        # Freeze all layers except the final FC layer initially
        for name, param in self.model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        
        # Initialize loss criterion and optimizer
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.optimizer = optim.AdamW([
            {'params': [p for n, p in self.model.named_parameters() if 'fc' not in n], 'lr': self.lr/10},
            {'params': self.model.fc.parameters(), 'lr': self.lr}
        ], weight_decay=self.weight_decay)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max',
            factor=self.lr_decay_factor,
            patience=self.lr_patience,
            verbose=True
        )
        
        # Early stopper
        early_stopper = EarlyStopping(
            patience=self.early_stop_patience,
            verbose=True,
            delta=0.001,
            path=os.path.join(self.results_dir, "best_model.pth")
        )
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_acc = 0.0
        
        print(f"Starting training on {self.device}")
        for epoch in range(self.epochs):
            # Gradual unfreezing after epoch 5
            if epoch == 5:
                print("\nUnfreezing all layers for fine-tuning...")
                for param in self.model.parameters():
                    param.requires_grad = True
                # Recreate optimizer with new trainable parameters
                self.optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=self.lr/5,  # Slightly lower learning rate for fine-tuning
                    weight_decay=self.weight_decay
                )
            
            # Train epoch
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_results = self.validate()
            val_loss = val_results['val_loss']
            val_acc = val_results['val_acc']
            
            # Update scheduler and early stopper
            scheduler.step(val_acc)
            early_stopper(val_acc, self.model)
            
            if early_stopper.early_stop:
                print("Early stopping triggered")
                break
            
            # Save epoch results
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Add to training log
            self.training_log.append({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
            
            # Print epoch results
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, os.path.join(self.results_dir, "best_model.pth"))
                print("Saved new best model")
        
        # Training complete
        print("\nTraining complete!")
        print(f"Best Validation Accuracy: {best_acc:.2f}%")
        
        # Save training log
        self.save_training_log()
        
        # Generate and save plots
        self.plot_training_history(history)
        
        # Final evaluation
        val_results = self.validate()
        self.plot_confusion_matrix(val_results['confusion_matrix'])
        self.plot_metrics(val_results['report'])
        
        # Save classification report
        report_file = os.path.join(self.results_dir, "classification_report.txt")
        with open(report_file, 'w') as f:
            f.write(classification_report(
                val_results['all_labels'],
                val_results['all_preds'],
                target_names=self.classes
            ))
        print(f"Classification report saved to: {report_file}")
        
        # Run hard test evaluation (check multiple possible locations)
        hard_test_locations = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "hard_test"),
            r"C:\programing\Omri's copy\hard_test"
        ]
        
        for hard_test_dir in hard_test_locations:
            if os.path.exists(hard_test_dir):
                print(f"\nEvaluating hard test set at: {hard_test_dir}")
                self.evaluate_hard_test(hard_test_dir)
            else:
                print(f"\nHard test directory not found at {hard_test_dir}, skipping")

def main():
    config = {
        'data_dir': r"C:\programing\Omri's copy\img_set(with uncropped)",
        'image_size': 224,
        'batch_size': 64,
        'num_workers': 8,
        'lr': 3e-4,
        'weight_decay': 2e-4,
        'epochs': 30,
        'dropout_rate': 0.5,
        'label_smoothing': 0.2,
        'lr_decay_factor': 0.5,
        'lr_patience': 3,
        'early_stop_patience': 7
    }
    
    classifier = ImageClassifier(**config)
    classifier.train()

if __name__ == '__main__':
    # Create main results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    main()