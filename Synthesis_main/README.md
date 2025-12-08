# 🧬 Synthesis - Multimodal Healthcare Data Generator

A comprehensive platform for generating synthetic multimodal healthcare data using advanced machine learning techniques. This project creates privacy-safe, realistic datasets that include structured tabular data, time-series medical data, and cross-modal synthesis for AI/ML research and healthcare analytics.

**Author:** Dasari Ranga Eswar  
**GitHub:** [@eswaroy](https://github.com/eswaroy)  
**Repository:** [synthieis](https://github.com/eswaroy/synthiesis)  
**License:** MIT

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Technology Stack](#-technology-stack)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Data Formats](#-data-formats)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### Core Capabilities
- **🏥 Synthetic Tabular Data Generation**: Creates realistic Electronic Health Records (EHR) with patient demographics, vitals, diagnoses, and lab results
- **📈 Time-Series Medical Data**: Generates temporal medical sensor data for continuous monitoring scenarios
- **🔄 Cross-Modal Data Synthesis**: Produces interconnected multi-modal healthcare datasets
- **🛡️ Privacy-Safe**: No real patient data - perfect for research and development without privacy concerns
- **⚙️ Customizable Parameters**: Control dataset characteristics including:
  - Number of samples
  - Patient attributes (age, gender, conditions)
  - Disease types and prevalence
  - Time ranges and sampling frequencies
- **📊 Multiple Export Formats**: CSV and JSON output for easy integration
- **🎨 Interactive Dashboard**: Web-based UI for configuration and visualization

### Use Cases
- AI/ML model training and testing
- Prototyping healthcare analytics applications
- Educational demonstrations
- Algorithm validation without patient data
- Benchmark dataset creation

---

## 📁 Project Structure

```
Synthesis_main/
├── client/                    # React-based frontend application
│   ├── src/
│   │   ├── components/       # Reusable React components
│   │   ├── pages/            # Page components
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── index.html
├── server/                    # Python/Flask backend API
│   ├── app/
│   │   ├── routes/           # API endpoints
│   │   ├── models/           # Database models
│   │   ├── ml/               # ML utilities
│   │   ├── middleware/       # Flask middleware
│   │   └── utils/            # Helper functions
│   ├── ml_service/           # GAN-based data generation service
│   │   ├── models.py         # Neural network architectures
│   │   ├── gan_trainer.py    # Training logic
│   │   ├── generate.py       # Data generation functions
│   │   └── config.py         # Configuration parameters
│   ├── config.py
│   ├── run.py               # Application entry point
│   ├── requirements.txt
│   └── test_app.py
├── package.json             # Root package configuration
├── LICENSE                  # MIT License
└── README.md               # This file
```

---

## 🛠️ Technology Stack

### Frontend
- **React 19** - UI framework
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **Framer Motion** - Animation library
- **Recharts** - Data visualization
- **Axios** - HTTP client
- **React Router DOM** - Client-side routing

### Backend
- **Python 3.8+** - Programming language
- **Flask 2.3.3** - Web framework
- **PyTorch** - Deep learning framework for GANs
- **MongoDB** - Database
- **NumPy & Pandas** - Data processing
- **Scikit-learn** - ML utilities

### DevOps & Tools
- **Docker** (optional) - Containerization
- **Git** - Version control
- **ESLint** - JavaScript linting

---

## 📦 Prerequisites

- **Node.js** 16.x or higher
- **Python** 3.8 or higher
- **npm** or **yarn** (for Node package management)
- **pip** (for Python package management)
- **MongoDB** (if using database features)
- **CUDA** (optional, for GPU acceleration with PyTorch)

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/eswaroy/synthiesis.git
cd Synthesis_main
```

### 2. Backend Setup

```bash
# Navigate to server directory
cd server

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
# Navigate to client directory
cd ../client

# Install Node dependencies
npm install
```

---

## ⚙️ Configuration

### Backend Configuration

Edit `server/config.py` or create a `.env` file:

```env
FLASK_APP=run.py
FLASK_ENV=development
MONGODB_URI=mongodb://localhost:27017/synthesis
SECRET_KEY=your-secret-key-here
GAN_BATCH_SIZE=32
GAN_EPOCHS=100
LEARNING_RATE=0.0002
```

### Frontend Configuration

Create `client/.env` file:

```env
VITE_API_URL=http://localhost:5000/api
```

---

## 💻 Usage

### Start the Backend Server

```bash
cd server
# Activate virtual environment first
python run.py
```

The API will be available at `http://localhost:5000`

### Start the Frontend Development Server

```bash
cd client
npm run dev
```

The application will be available at `http://localhost:5173`

### Build for Production

**Frontend:**
```bash
cd client
npm run build
```

**Backend:**
Simply run the Flask app with proper environment configuration.

---

## 🔌 API Endpoints

### Data Generation
- `POST /api/generate/tabular` - Generate synthetic tabular data
- `POST /api/generate/timeseries` - Generate synthetic time-series data
- `POST /api/generate/cross-modal` - Generate cross-modal data

### Data Management
- `GET /api/datasets` - List all generated datasets
- `GET /api/datasets/<id>` - Get specific dataset details
- `DELETE /api/datasets/<id>` - Delete a dataset
- `GET /api/datasets/<id>/download` - Download dataset as CSV/JSON

### Models
- `GET /api/models` - List available GAN models
- `POST /api/models/train` - Train a new model
- `GET /api/models/<id>/status` - Get training status

---

## 📊 Data Formats

### Tabular Data Output (CSV/JSON)
```json
{
  "patient_id": "P001",
  "age": 45,
  "gender": "M",
  "diagnosis": "Diabetes Type 2",
  "glucose_level": 145,
  "blood_pressure_systolic": 130,
  "blood_pressure_diastolic": 85,
  "heart_rate": 72,
  "cholesterol": 210,
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Time-Series Data Output
```json
[
  {
    "patient_id": "P001",
    "timestamp": "2025-01-15T10:00:00Z",
    "heart_rate": 72,
    "oxygen_saturation": 98,
    "respiration_rate": 16,
    "temperature": 37.2
  },
  ...
]
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards
- Follow PEP 8 for Python code
- Use ESLint for JavaScript/React code
- Write descriptive commit messages
- Add tests for new features

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Copyright (c) 2025 Dasari Ranga Eswar**

---

## 📚 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [React Documentation](https://react.dev/)
- [Vite Documentation](https://vitejs.dev/)

---

## 🆘 Support

For issues, questions, or suggestions:
- Open an issue on [GitHub](https://github.com/eswaroy/synthiesis/issues)
- Contact: [GitHub Profile](https://github.com/eswaroy)

---

**Built with ❤️ by [Dasari Ranga Eswar](https://github.com/eswaroy)**
