<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Voice-Controlled Object Detection Training

This directory contains a script that demonstrates a complete voice-controlled object detection training system using 🤗 Transformers.

## Overview

The voice detection training system allows you to:

1. **Speak item names** (voice → text conversion)
2. **Capture images** of those items using a camera
3. **Auto-create datasets** with spoken labels
4. **Periodically fine-tune** a detection model in the background
5. **Learn new store items** over time through continuous learning

## Features

- Voice recognition using Google Speech Recognition
- Camera capture with OpenCV
- Real model fine-tuning using DETR (Detection Transformer)
- Background training with periodic retraining
- Model export capabilities for production use
- Simulated demo mode (no hardware required)

## Quick Start

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

For speech recognition, you may need to install additional audio dependencies:

```bash
# On Ubuntu/Debian
sudo apt-get install portaudio19-dev python3-pyaudio

# On macOS
brew install portaudio

# On Windows
# PyAudio should install automatically with pip
```

### Usage

#### Run the full system (requires camera and microphone):

```bash
python run_voice_detection.py
```

#### Run the simulated demo (no hardware required):

```bash
python run_voice_detection.py --demo
```

#### Run advanced system with real model training:

```bash
python run_voice_detection.py --advanced
```

#### Run Transformers integration example:

```bash
python run_voice_detection.py --transformers
```

## How to Use

### Training New Items

1. **Start the system** - Camera initializes and shows live feed
2. **Speak item names** - Say "apple", "coffee mug", etc.
3. **Position items** - Hold items in front of camera
4. **Auto-capture** - System captures and labels images automatically
5. **Background learning** - Model retrains periodically with new data
6. **Test detection** - Say "detect" to test current capabilities

### Voice Commands

- **Item names** - Any spoken word becomes a training label
- **"detect"** - Test current detection capabilities  
- **"quit"** - Exit the system

### Demo Mode

The demo mode simulates the entire workflow without requiring hardware:
- Simulates voice input with text commands
- Creates demo images with item names
- Shows the complete training and detection pipeline
- Perfect for testing and understanding the system

## Technical Details

### Architecture

The system uses a modular architecture with:

- **Voice Processing Pipeline** - Speech → Text → Labels
- **Image Capture Pipeline** - Camera → Images → Storage
- **Training Pipeline** - Data → Model → Fine-tuning
- **Detection Pipeline** - Images → Model → Predictions

### Model

- Uses DETR (Detection Transformer) for object detection
- Supports custom class learning
- Background fine-tuning with periodic retraining
- Model versioning and export capabilities

### Data Management

- Automatic dataset creation with spoken labels
- JSON-based metadata storage
- Class mapping and distribution tracking
- Data augmentation for better training

## File Structure

```
voiceandimagedataset-recognition/
├── run_voice_detection.py          # Main script
├── requirements.txt                # Dependencies
├── README.md                       # This file
└── voice_detection_data/           # Generated data (created at runtime)
    ├── metadata.json               # Dataset metadata
    ├── trained_model/              # Saved model files
    │   ├── pytorch_model.bin
    │   └── config.json
    └── item_name_timestamp.jpg     # Training images
```

## Customization

### Adding New Models

You can easily swap the detection model by modifying the model initialization:

```python
# Use a different pre-trained model
self.model_name = "facebook/detr-resnet-101"

# Or use a custom model
self.model = YourCustomModel()
```

### Adjusting Training Parameters

Modify the training configuration:

```python
# Change retraining interval
retrain_interval = timedelta(minutes=60)  # Retrain every hour

# Adjust confidence threshold
if score > 0.7:  # Higher confidence threshold
```

### Data Augmentation

Customize data augmentation in the dataset class:

```python
self.augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),  # More rotation
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    # Add more augmentations...
])
```

## Troubleshooting

### Common Issues

1. **Camera not found**
   - Check camera permissions
   - Try different camera IDs (0, 1, 2...)
   - Ensure camera is not being used by another application

2. **Speech recognition not working**
   - Check microphone permissions
   - Ensure internet connection (for Google Speech API)
   - Try speaking louder or closer to microphone

3. **Model training errors**
   - Ensure sufficient training data (at least 2 items)
   - Check GPU memory if using CUDA
   - Verify all dependencies are installed

4. **Poor detection accuracy**
   - Add more training data for each class
   - Ensure good lighting and clear images
   - Try different camera angles and distances

### Performance Tips

1. **Use GPU acceleration** when available
2. **Increase training data** for better accuracy
3. **Adjust confidence thresholds** based on your needs
4. **Regular retraining** improves performance over time

## Examples

### Basic Usage

```python
from run_voice_detection import VoiceDetectionTrainer

# Create trainer
trainer = VoiceDetectionTrainer()

# Start training session
trainer.start_training_session()
```

### Custom Configuration

```python
# Custom data directory and model
trainer = VoiceDetectionTrainer(
    data_dir="my_custom_data",
    model_name="facebook/detr-resnet-101"
)

# Add training data programmatically
from PIL import Image
image = Image.open("my_item.jpg")
trainer.add_training_data("my_item", image)
```

### Model Export

```python
# Export trained model for production
trainer.export_model("production_model")
```

## Contributing

Feel free to contribute improvements:
- Better model architectures
- Enhanced data augmentation
- Improved voice recognition
- Additional export formats
- Performance optimizations

## License

This example follows the same license as the Transformers library.
