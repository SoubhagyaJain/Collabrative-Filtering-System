# Collaborative Filtering Movie Recommendation System

A user-friendly movie recommendation application built using **Collaborative Filtering**. This system analyzes user ratings and finds similar users to recommend movies they might enjoy. The app uses machine learning (scikit-learn) and is deployed as an interactive web application with Streamlit.

## What is Collaborative Filtering?

Collaborative filtering is a technique that recommends items based on the preferences of similar users. The idea is simple:
- If User A and User B have rated movies similarly in the past
- Then they likely have similar tastes
- So we can recommend movies that User B liked to User A

## Project Overview

This project implements a **user-user collaborative filtering** system that:
- Analyzes movie ratings from multiple users
- Identifies users with similar rating patterns
- Recommends movies based on what similar users enjoyed
- Displays recommendations with detailed movie information and ratings

---

## 🚀 Quick Start Guide for Beginners

### Prerequisites

Before you start, make sure you have the following installed on your computer:

1. **Python 3.8 or higher** - [Download here](https://www.python.org/downloads/)
   - Check if installed: Open terminal/command prompt and type `python --version`

2. **Git** (optional, for cloning the repository) - [Download here](https://git-scm.com/)

### Installation Steps

#### Step 1: Clone or Download the Project

**Option A: Using Git (recommended)**
```bash
git clone https://github.com/SoubhagyaJain/Personal-collabrative-system.git
cd Personal-collabrative-system
```

**Option B: Download ZIP**
- Download the project as ZIP from GitHub and extract it to a folder

#### Step 2: Create a Virtual Environment

A virtual environment keeps this project's dependencies separate from other Python projects.

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal line, indicating the virtual environment is active.

#### Step 3: Install Required Packages

With the virtual environment activated, run:

```bash
pip install -r requirements.txt
```

This installs all the necessary libraries:
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **streamlit** - Web app framework
- **plotly** - Interactive charts and visualizations
- **scikit-learn** - Machine learning library
- **ipynb** - Jupyter notebook support

#### Step 4: Run the Application

```bash
streamlit run app.py
```

**What to expect:**
- Your terminal will show: `You can now view your Streamlit app in your browser.`
- A local URL will appear (usually `http://localhost:8501`)
- Your web browser should open automatically
- If not, manually open the URL shown in your terminal

### How to Use the Application

1. **Select a User:** Use the dropdown menu to choose a user name
2. **View Recommendations:** The app displays 10 recommended movies for that user
3. **See Details:** Scroll down to view detailed information and ratings for each recommended movie

---

## 📁 Project Files

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit web application - runs the interactive UI |
| `recommender.py` | Core recommendation engine - contains the collaborative filtering logic |
| `CollaborativeFiltering.ipynb` | Jupyter notebook with exploratory analysis and model development |
| `Movie_data.csv` | User ratings data (User, Movie_ID, Ratings) |
| `Movie_Id_Titles.csv` | Movie information (Movie_ID, Movie Title) |
| `requirements.txt` | List of Python packages needed for the project |
| `README.md` | This file - project documentation |

---

## 🛠️ Project Structure

```
Collabrative-Filtering-System/
├── app.py                    # Web application
├── recommender.py            # Recommendation engine
├── CollaborativeFiltering.ipynb  # Jupyter notebook
├── Movie_data.csv            # User ratings
├── Movie_Id_Titles.csv       # Movie titles
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
└── LICENSE                   # Project license
```

---

## 📊 How the Recommendation System Works

### Step-by-Step Process:

1. **Load Data:** Read user ratings from CSV files
2. **Calculate Similarity:** Use cosine similarity to find users with similar rating patterns
3. **Find Similar Users:** For a selected user, find the most similar users
4. **Get Recommendations:** Collect highly-rated movies from similar users that the target user hasn't watched
5. **Display Results:** Show top 10 recommendations with movie details

### Mathematical Concept:

The system uses **cosine similarity** to measure how similar two users are:
- Compares their rating vectors
- Values closer to 1 mean very similar tastes
- Values closer to 0 mean different tastes

---

## 🔍 Exploring the Code

### `app.py` - The Web Interface
- Creates the user interface using Streamlit
- Displays dropdowns, tables, and charts
- Shows recommended movies with ratings

### `recommender.py` - The Recommendation Engine
Main functions:
- `prepare_data()` - Loads and prepares data
- `movie_recommender_run()` - Generates recommendations for a user

### `CollaborativeFiltering.ipynb` - Analysis Notebook
- Exploratory data analysis
- Model development and testing
- Visualization of results

---

## 📋 Requirements

All requirements are listed in `requirements.txt`:
```
pandas==2.0.3
numpy==1.24.3
streamlit==1.28.1
plotly==5.17.0
scikit-learn==1.3.2
ipynb==0.5.1
```

---

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'streamlit'"
**Solution:** Make sure you've activated the virtual environment and installed requirements:
```bash
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### Problem: Port 8501 already in use
**Solution:** The app is trying to use a port that's already in use. Either:
- Close the other application using port 8501
- Or specify a different port:
```bash
streamlit run app.py --server.port 8502
```

### Problem: Module versions conflict
**Solution:** Create a fresh virtual environment:
```bash
deactivate  # Exit current environment
rm -rf venv  # Remove old environment (macOS/Linux)
# or rmdir /s venv (Windows)
python -m venv venv  # Create new one
# Activate and reinstall
```

---

## 💡 Tips for Beginners

1. **Virtual Environments:** Always use a virtual environment to avoid dependency conflicts
2. **Keep Terminal Open:** Keep the terminal window open while using the Streamlit app
3. **Explore the Code:** Read `recommender.py` to understand how recommendations work
4. **Try Different Users:** Select different users to see how recommendations change
5. **Check the Notebook:** `CollaborativeFiltering.ipynb` shows the data analysis process

---

## 📚 Learning Resources

- **Collaborative Filtering:** https://en.wikipedia.org/wiki/Collaborative_filtering
- **Scikit-learn Documentation:** https://scikit-learn.org/
- **Streamlit Guide:** https://docs.streamlit.io/
- **Pandas Tutorial:** https://pandas.pydata.org/docs/

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests

---

## 👤 Author

SoubhagyaJain - [GitHub Profile](https://github.com/SoubhagyaJain)

---

## ❓ FAQ

**Q: What data does this use?**
A: The system uses an IMDB movie dataset with user ratings and movie titles.

**Q: How many recommendations are shown?**
A: The system displays the top 10 movie recommendations for each user.

**Q: Can I use my own dataset?**
A: Yes! You can replace `Movie_data.csv` and `Movie_Id_Titles.csv` with your own data in the same format.

**Q: How long does it take to run?**
A: The app loads in seconds. Recommendations are generated instantly when you select a user.

---

**Happy Recommending! 🎬**
