🛰️ WiFall-Guard: Wi-Fi CSI-Based Fall Detection
WiFall-Guard is a sophisticated machine learning pipeline designed to detect human falls by analyzing disturbances in Wi-Fi Channel State Information (CSI). Unlike camera-based systems, it ensures privacy while maintaining high detection accuracy in indoor environments.

🛠️ Technical Architecture
The system follows a modular "Pipeline" architecture to ensure scalability and maintainability:

Signal Preprocessing: Implements a 5th-order Butterworth Low-pass filter to eliminate high-frequency noise and environmental interference.

Feature Engineering: Extracts critical statistical identifiers from signal segments, including Mean, Variance, Standard Deviation, and Signal Energy (Key indicator for impact detection).

Overlapping Windowing: Utilizes a sliding window technique (Size: 100, Step: 20) to augment the dataset and capture temporal dependencies effectively.

Standardization: Features are normalized using StandardScaler to ensure uniform contribution to the model's decision-making process.

🤖 Model Performance
The core engine is powered by a Random Forest Classifier (200 Estimators), achieving high reliability in non-line-of-sight scenarios.

Accuracy: 93.33%

Recall (Fall Detection): 1.00 (Zero false negatives in critical fall events)

F1-Score: 0.90 (Weighted average)

📁 Project Structure
Plaintext

├── main.py              # Main execution script & orchestrator
├── data/                # CSI signal annotations (CSV)
├── models/              # Saved model (PKL) and Scaler
└── src/
    ├── preprocess.py    # Signal cleaning & filtering
    ├── features.py      # Statistical feature extraction
    └── model.py         # ML training & evaluation logic
🚀 Installation & Usage
Clone the repository and install dependencies:

Bash

pip install -r requirements.txt
Run the detection pipeline:

Bash

python main.py

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.