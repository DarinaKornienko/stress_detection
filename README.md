---
title: Stress Detection Ensemble
colorFrom: blue
colorTo: red
sdk: streamlit
python_version: 3.10.12
sdk_version: 1.31.0
app_file: app.py
pinned: false
---

# Stress detectioin in human speech


> **Live Deployment:** [Try the Stress-O-Meter here](https://huggingface.co/spaces/dagnie/stress_detection)

## Project purpose

#### The main goal of this project is to compare how different neural network architectures perform at detecting stress in human speech. It includes handling the entire process from start to finish: beginning with raw data, moving through model research, architecture desing and training to finally creating a live tool that anyone can use.
---

## Phase 1: Research & Model Training
The project began with an experimental phase in Jupyter Notebooks to determine the most effective ensemble for capturing vocal tension.
This stage involved solving the core AI problem from scratch. All research logic and visualizations can be found in the project notebooks.

* **Data Preprocessing**: Cleaned and standardized the raw audio data so the models could read the features clearly
* **Feature Engineering**: Utilized Mel-spectrogram conversion and Power-to-DB scaling for temporal frequency analysis
* **Building the Networks**: Designed and compared multiple architectures, including CNN-LSTM, PANNs, and YAMNet, to see which handled vocal tension best
* **Training and Testing**: I trained these networks and tested their performance using real-world speech samples.

* **Results**: The project includes visualizations of the training accuracy and loss, proving how the models learned to identify stress patterns over time.
---

## Phase 2: Building the Application
Transitioning from research to a production-ready application required building a stable backend to handle multiple frameworks (PyTorch and TensorFlow) simultaneously.

### Technical Implementation:
* **Namespace Injection**: Implemented a custom module injection hack to ensure model weights loaded correctly across different system environments.
* **Standardization Pipeline**: Built a processing layer using `Pydub` that automatically converts user uploads (MP3, M4A, OGG) to the exact 16kHz Mono format the models require.

---

## Phase 3: Deployment & MLOps
The application is deployed on **Hugging Face Spaces** using a Continuous Deployment (CD) pipeline linked to this GitHub repository _(link at the top of a document)_.

### Key Deployment Achievements:
* **Cross-Platform Compatibility**: Developed an OS-aware FFmpeg configuration that supports both local Windows development and Linux-based cloud production.
* **Environment Stability**: To resolve build conflicts with modern Python versions, the app is forced to run on **Python 3.10.12**, ensuring compatibility for `Pillow`, `TensorFlow`, and `Torch`.
* **System Integration**: Integrated system-level dependencies via `packages.txt` to enable high-performance audio processing in the cloud.

---

## Local Installation & Usage

#### Requirements
* **Python**: 3.10.x is recommended for model compatibility.
* **FFmpeg**: Required for audio processing.
  * **Windows**: Place `ffmpeg.exe` and `ffprobe.exe` in the root folder.
  * **Linux**: Install via `sudo apt install ffmpeg`.

#### Installation
```bash
# Clone the repo
git clone [https://github.com/DarinaKornienko/stress_detection](https://github.com/DarinaKornienko/stress_detection)
cd stress_detection

# Install requirements
pip install -r requirements.txt

# Ensure model weights are downloaded (Git LFS)
git lfs pull
```
#### Run the App

```bash
streamlit run app.py
```