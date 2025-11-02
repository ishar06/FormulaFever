

![Formula 1 Banner](https://raw.githubusercontent.com/ishar06/FormulaFever/main/static/images/f1-banner.png)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-lightgrey?logo=flask)](https://flask.palletsprojects.com/)
[![FastF1](https://img.shields.io/badge/FastF1-Latest-red)](https://docs.fastf1.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


# 🏎️ FormulaFever 

> "Life is measured in achievement, not in years alone." - Bruce McLaren

Experience Formula 1 like never before with FormulaFever - your ultimate F1 analytics dashboard that brings the thrill of the track to your fingertips! 🏁

## 🌟 Key Features

### 🏃 Live Race Analytics
- Real-time race data visualization
- Lap time analysis
- Sector comparisons
- Track position tracking

### 👻 Ghost Lap Comparator
- Compare lap times between different drivers
- Visual sector-by-sector comparison
- Interactive lap time charts
- Head-to-head performance analysis

### 🏎️ Driver Analysis
- Comprehensive driver statistics
- Historical performance data
- Race-by-race progression
- Championship points tracking

### 🤝 Community Features
- User registration and authentication
- Discussion forums
- Race predictions sharing
- Community insights

### 📊 Race Analysis
- Previous year race comparisons
- Track evolution analysis
- Team performance metrics
- Weather impact analysis

### 🎯 Race Win Probability Predictor
- AI-powered race outcome predictions
- Driver performance forecasting
- Track-specific probability calculations
- Historical data-based predictions

### 🤖 AI Chatbot
- Formula 1 knowledge assistance
- Real-time race insights
- Historical race information
- Technical regulations explanations

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ishar06/FormulaFever.git
cd FormulaFever
```

2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows, use: env\\Scripts\\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a .env file in the root directory with:
```
GROQ_API_KEY=your_groq_api_key
SECRET_KEY=your_secret_key
```

5. Initialize the database:
```bash
flask shell
>>> from app import db
>>> db.create_all()
>>> exit()
```

## 🏆 The Team Behind FormulaFever

Meet the champions who brought this project to life through countless hours of coding, debugging, and optimization:

- **[Ishardeep Singh](https://github.com/ishar06)** - Project Lead & Backend Development
- **[Bhumika Nagpal](https://github.com/BhumikaNagpal)** - Data Analysis & ML Models
- **[Damanjeet Singh](https://github.com/daman-max)** - API and Integration Head
- **[Karandeep Kaur](https://github.com/karandeepkaur18)** - Database Architecture & Model Refiner

## 🔧 Contributing

We welcome contributions! Feel free to:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FastF1 Library
- Formula 1 for the amazing sport
- Our dedicated F1 community

---

<p align="center">Made with ❤️ and ⚡ by the FormulaFever Team</p>

> "To do something well is so worthwhile that to die trying to do it better cannot be foolhardy." - Bruce McLaren"

## 🗄️ Project Structure
```
├── app.py              # Main Flask application
├── analysis.py         # Data analysis functions
├── requirements.txt    # Python dependencies
├── static/            # Static files (CSS, JS, images)
├── templates/         # HTML templates
├── ff1_cache/         # FastF1 cache directory
└── instance/         # Instance-specific files
```

## 💻 Technologies Used

- **Backend**: Flask, Python
- **Data Analysis**: FastF1, Pandas, NumPy
- **AI/ML**: Groq AI, Scikit-learn
- **Frontend**: HTML, CSS, JavaScript, Plotly
- **Database**: SQLite, SQLAlchemy
- **Authentication**: Flask-Login
- **Caching**: Flask-Caching

## 🔑 API Keys Required

- Groq AI API Key (for chatbot functionality)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- FastF1 library developers
- Formula 1 for the data
- All contributors and community members
