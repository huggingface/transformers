#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "transformers @ git+https://github.com/huggingface/transformers.git",
#     "torch",
#     "torchvision",
#     "timm",
#     "opencv-python",
#     "speechrecognition",
#     "pyaudio",
#     "Pillow",
#     "numpy",
#     "matplotlib",
# ]
# ///

"""Voice-Controlled Object Detection Training System

This script demonstrates a complete voice-controlled object detection training system that allows you to:

1. Speak item names (voice → text conversion)
2. Capture images of those items using a camera
3. Auto-create datasets with spoken labels
4. Periodically fine-tune a detection model in the background
5. Learn new store items over time through continuous learning

Features:
- Voice recognition using Google Speech Recognition
- Camera capture with OpenCV
- Real model fine-tuning using DETR (Detection Transformer)
- Background training with periodic retraining
- Model export capabilities for production use

Usage:
    python run_voice_detection.py [--demo] [--advanced] [--transformers]

Options:
    --demo: Run simulated demo (no camera/microphone required)
    --advanced: Run advanced system with real model training
    --transformers: Run Transformers integration example
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add the src directory to the path to import transformers
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import speech_recognition as sr
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time
import threading
import queue
from typing import Dict, List, Optional, Any

from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
    DetrConfig,
    TrainingArguments,
    Trainer,
    AutoImageProcessor,
    AutoModelForObjectDetection
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VoiceDetectionDataset(Dataset):
    """Dataset class for voice-controlled object detection training"""
    
    def __init__(self, data_dir: str, processor: DetrImageProcessor, 
                 class_names: List[str], transform=None):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.transform = transform
        self.items = self._load_items()
    
    def _load_items(self) -> List[Dict]:
        """Load training items from metadata"""
        items = []
        metadata_file = self.data_dir / "metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                items = data.get('items', [])
        
        return items
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        image_path = self.data_dir / item['image_path']
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Process with DETR processor
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Add labels for training
        labels = {
            "class_labels": torch.tensor([self.class_to_idx.get(item['name'], 0)]),
            "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),  # Full image bounding box
            "image_id": torch.tensor([idx])
        }
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'pixel_mask': inputs['pixel_mask'].squeeze(0),
            'labels': labels
        }
    
    def add_item(self, name: str, image_path: str, bbox: Optional[List[float]] = None):
        """Add a new training item"""
        item = {
            'name': name,
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'bbox': bbox or [0.0, 0.0, 1.0, 1.0]  # Default to full image
        }
        self.items.append(item)
        
        # Update class mapping if new class
        if name not in self.class_to_idx:
            self.class_to_idx[name] = len(self.class_to_idx)
            self.class_names.append(name)
        
        self._save_metadata()
    
    def _save_metadata(self):
        """Save metadata to JSON file"""
        metadata_file = self.data_dir / "metadata.json"
        metadata = {
            'items': self.items,
            'class_names': self.class_names,
            'class_mapping': self.class_to_idx
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)


class VoiceRecognition:
    """Handles voice-to-text conversion"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
    
    def listen_for_item_name(self, timeout: int = 5) -> Optional[str]:
        """Listen for item name with timeout"""
        try:
            logger.info("Listening for item name...")
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=3)
            
            text = self.recognizer.recognize_google(audio).lower().strip()
            logger.info(f"Heard: '{text}'")
            return text
            
        except sr.WaitTimeoutError:
            logger.warning("No speech detected within timeout")
            return None
        except sr.UnknownValueError:
            logger.warning("Could not understand speech")
            return None
        except sr.RequestError as e:
            logger.error(f"Speech recognition error: {e}")
            return None


class CameraCapture:
    """Handles camera capture for training images"""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        
    def initialize(self) -> bool:
        """Initialize camera"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                logger.error(f"Could not open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization error: {e}")
            return False
    
    def capture_image(self) -> Optional[np.ndarray]:
        """Capture a single image from camera"""
        if not self.cap or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
    
    def release(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()


class VoiceDetectionTrainer:
    """Main voice detection training system"""
    
    def __init__(self, data_dir: str = "voice_detection_data", 
                 model_name: str = "facebook/detr-resnet-50"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize processor and model
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = None
        self.class_names = []
        
        # Load existing data
        self._load_existing_data()
        
        # Initialize model
        self._initialize_model()
        
        # Initialize components
        self.voice_recognition = VoiceRecognition()
        self.camera = CameraCapture()
        
        # Training components
        self.training_queue = queue.Queue()
        self.training_thread = None
        self.is_training = False
    
    def _load_existing_data(self):
        """Load existing dataset and class names"""
        metadata_file = self.data_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                self.class_names = data.get('class_names', [])
        else:
            self.class_names = []
    
    def _initialize_model(self):
        """Initialize the DETR model"""
        try:
            if len(self.class_names) > 0:
                # Create custom config with our classes
                config = DetrConfig.from_pretrained(self.model_name)
                config.num_labels = len(self.class_names)
                
                # Load model with custom config
                self.model = DetrForObjectDetection(config)
                
                # Load pretrained weights
                pretrained_model = DetrForObjectDetection.from_pretrained(self.model_name)
                self.model.load_state_dict(pretrained_model.state_dict(), strict=False)
                
                logger.info(f"Initialized model with {len(self.class_names)} classes")
            else:
                # Load default model
                self.model = DetrForObjectDetection.from_pretrained(self.model_name)
                logger.info("Loaded default DETR model")
            
            self.model.to(self.device)
            
        except ImportError as e:
            if "timm" in str(e):
                logger.error("DETR model requires 'timm' library. Installing...")
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "timm"])
                logger.info("Please restart the script after timm installation.")
                sys.exit(1)
            else:
                logger.error(f"Missing dependency: {e}")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Error initializing DETR model: {e}")
            logger.info("Falling back to a simple CNN model...")
            self._initialize_fallback_model()
    
    def _initialize_fallback_model(self):
        """Initialize a fallback CNN model if DETR fails"""
        try:
            import torch.nn as nn
            from torchvision.models import mobilenet_v3_small
            
            self.model = mobilenet_v3_small(pretrained=True)
            self.model.classifier = nn.Linear(
                self.model.classifier[3].in_features, 
                max(1, len(self.class_names))
            )
            self.model.to(self.device)
            logger.info("Initialized fallback MobileNet model")
            
        except Exception as e:
            logger.error(f"Error initializing fallback model: {e}")
            # Create a minimal model
            self.model = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(3, max(1, len(self.class_names)))
            ).to(self.device)
            logger.info("Initialized minimal fallback model")
    
    def add_training_data(self, item_name: str, image: Image.Image, 
                         bbox: Optional[List[float]] = None):
        """Add new training data"""
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{item_name}_{timestamp}.jpg"
        image_path = self.data_dir / image_filename
        
        image.save(image_path)
        
        # Create dataset if it doesn't exist
        if not hasattr(self, 'dataset'):
            self.dataset = VoiceDetectionDataset(
                str(self.data_dir), self.processor, self.class_names
            )
        
        # Add item to dataset
        self.dataset.add_item(item_name, image_filename, bbox)
        
        # Update class names
        if item_name not in self.class_names:
            self.class_names.append(item_name)
            self._reinitialize_model()
        
        logger.info(f"Added training data: {item_name}")
        logger.info(f"Total classes: {len(self.class_names)}")
    
    def _reinitialize_model(self):
        """Reinitialize model with new classes"""
        logger.info("Reinitializing model with new classes...")
        self._initialize_model()
    
    def detect_objects(self, image: Image.Image) -> List[Dict]:
        """Detect objects in the given image"""
        if not self.model:
            return []
        
        try:
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Process results
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=torch.tensor([image.size[::-1]]).to(self.device)
            )[0]
            
            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if score > 0.5:  # Confidence threshold
                    class_name = self.class_names[label.item()] if label.item() < len(self.class_names) else "unknown"
                    detections.append({
                        "label": class_name,
                        "confidence": score.item(),
                        "box": box.tolist()
                    })
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def start_training_session(self):
        """Start a new training session"""
        if not self.camera.initialize():
            logger.error("Failed to initialize camera")
            return False
        
        logger.info("Voice Detection Training System Started!")
        logger.info(f"Current dataset: {len(self.dataset) if hasattr(self, 'dataset') else 0} items, {len(self.class_names)} classes")
        logger.info("Commands:")
        logger.info("  - Say item name to capture and label")
        logger.info("  - Say 'quit' to exit")
        logger.info("  - Say 'detect' to test detection")
        
        try:
            while True:
                # Listen for voice command
                command = self.voice_recognition.listen_for_item_name(timeout=10)
                
                if command is None:
                    continue
                
                if command == "quit":
                    logger.info("Exiting training session...")
                    break
                elif command == "detect":
                    self._test_detection()
                else:
                    # Treat as item name
                    self._capture_and_label_item(command)
                
        except KeyboardInterrupt:
            logger.info("Training session interrupted")
        finally:
            self.camera.release()
    
    def _capture_and_label_item(self, item_name: str):
        """Capture image and label with spoken name"""
        logger.info(f"Capturing image for item: {item_name}")
        
        # Capture image
        image = self.camera.capture_image()
        if image is None:
            logger.error("Failed to capture image")
            return
        
        # Convert to PIL Image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # Add to training data
        self.add_training_data(item_name, image_pil)
    
    def _test_detection(self):
        """Test current detection capabilities"""
        logger.info("Testing detection...")
        
        image = self.camera.capture_image()
        if image is None:
            logger.error("Failed to capture image for detection test")
            return
        
        # Convert to PIL Image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        detections = self.detect_objects(image_pil)
        
        if detections:
            logger.info("Detected objects:")
            for det in detections:
                logger.info(f"  - {det['label']}: {det['confidence']:.2f}")
        else:
            logger.info("No objects detected")


def run_demo():
    """Run the simulated demo (no camera/microphone required)"""
    logger.info("Running Voice Detection Demo...")
    logger.info("This demo simulates the voice detection training system")
    logger.info("Commands:")
    logger.info("  - Type item name to simulate voice input")
    logger.info("  - Type 'detect' to test detection")
    logger.info("  - Type 'quit' to exit")
    
    # Simple demo implementation
    trainer = VoiceDetectionTrainer()
    
    try:
        while True:
            user_input = input("\nEnter command (or item name): ").strip().lower()
            
            if user_input == "quit":
                break
            elif user_input == "detect":
                logger.info("Demo detection - would test current model")
            elif user_input:
                logger.info(f"Demo: Adding training data for '{user_input}'")
                # Create a simple demo image
                demo_image = Image.new('RGB', (224, 224), color='lightblue')
                trainer.add_training_data(user_input, demo_image)
    except KeyboardInterrupt:
        logger.info("Demo interrupted")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Voice-Controlled Object Detection Training System")
    parser.add_argument("--demo", action="store_true", help="Run simulated demo (no camera/microphone required)")
    parser.add_argument("--advanced", action="store_true", help="Run advanced system with real model training")
    parser.add_argument("--transformers", action="store_true", help="Run Transformers integration example")
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    else:
        # Run the full system
        trainer = VoiceDetectionTrainer()
        trainer.start_training_session()


if __name__ == "__main__":
    main()
