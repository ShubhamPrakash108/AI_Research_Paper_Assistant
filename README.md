# 🧠 AI Research Assistant

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![LangChain](https://img.shields.io/badge/Built%20with-LangChain-green)
![Gemini](https://img.shields.io/badge/Powered%20by-Gemini-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)



A comprehensive AI-powered assistant designed to help researchers navigate the complex world of academic papers.

## 🌟 Overview


This project creates an intelligent research companion that helps academics:
- Discover prerequisite papers needed to understand complex research
- Evaluate and organize papers by difficulty level
- Interactively explore paper content through a user-friendly interface
- Ask questions about research documents and receive context-aware answers

The system leverages Google's Gemini LLM and various LangChain utilities to provide a seamless research experience.

## 🚀 Key Features

### 📚 Prerequisite Paper Prediction


Automatically identifies foundational papers needed to understand a target paper, making literature review more efficient.

### 📊 Paper Evaluation & Organization
- Assesses paper relevance to your target research
- Evaluates difficulty levels (Beginner, Intermediate, Expert)
- Downloads papers from arXiv
- Organizes them by difficulty for a structured learning path



### 🔍 Interactive Exploration

- View PDFs directly in the app
- Find related YouTube videos and academic blogs
- Query specific content within papers
- Chat interface powered by Gemini

<br clear="left">

### ❓ Document Q&A System
- Ask questions about your research documents
- Get answers with relevant context
- Save conversation history for later reference



## 🛠️ System Components

### Core Modules

| Module | Functionality |
|--------|---------------|
| **A1_reference_predictor.py** | Extracts text from target paper, suggests prerequisites |
| **A2_paper_check.py** | Validates relevance, downloads papers, evaluates difficulty |
| **A3_individiual_paper_Explorer.py** | Interactive Streamlit app for paper exploration |
| **A4_ques_and_ans.py** | Document Q&A system with conversation history |



## 📋 Setup Instructions

### Prerequisites
- Python 3.8+
- Required packages:
  ```bash
  pip install -r requirements.txt
  ```

### API Keys
You'll need:
- Google Gemini API key(s)
- YouTube API key (for video search functionality)

#### Important Note on API Rate Limits
While you can use a single Gemini API key, this project makes frequent API calls that may hit Google's rate limits (typically 2-30 requests per minute depending on model version). To ensure smooth operation, the system supports using multiple API keys to work around these limitations:

```python
# You can provide a single key:
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

# Or multiple keys to handle rate limiting:
GEMINI_API_KEYS = ["YOUR_GEMINI_API_KEY_1", "YOUR_GEMINI_API_KEY_2", ...]

# YouTube API is used for video search in A3_individiual_paper_Explorer.py
YOUTUBE_API_KEY = "YOUR_YOUTUBE_API_KEY"
```

The system will automatically rotate between available API keys to balance the request load.



### Folder Structure Setup
Create these folders before running:
```
├── target_RP/                     # For your target research paper
├── reference_research_papers/     # Where prerequisite papers will be stored
```

Place your research paper (PDF format) in the `target_RP/` folder.

## 🚀 Usage Guide



### 1. Identify Prerequisite Papers
```bash
python A1_reference_predictor.py
```
This generates `Research_Papers_DB.json` with suggested prerequisite papers.

### 2. Validate and Organize Papers
```bash
python A2_paper_check.py
```
This downloads relevant papers from arXiv and organizes them by difficulty.

### 3. Explore Papers Interactively
```bash
streamlit run A3_individiual_paper_Explorer.py
```
Use the web interface to view papers, find related content, and query paper content.



### 4. Ask Questions About Your Research
```bash
streamlit run A4_ques_and_ans.py
```
Use the Q&A interface to ask questions about all your documents.


## 📂 Project Structure

```
.
├── target_RP/                      # Target research paper PDF
├── reference_research_papers/      # Downloaded prerequisite papers
├── Research_Papers_DB.json         # Suggested prerequisite papers
├── cleaned_filtered_papers.json    # Validated prerequisite papers
├── A1_reference_predictor.py       # Prerequisite prediction script
├── A2_paper_check.py               # Paper validation script
├── A3_individiual_paper_Explorer.py # Interactive explorer app
├── A4_ques_and_ans.py              # Q&A system app
└── target_paper.json               # Stores target paper title
```

## 📄 Output Files

| File | Description |
|------|-------------|
| `output.json` | Raw extracted paper titles |
| `Research_Papers_DB.json` | Deduplicated paper titles |
| `filtered_papers.json` | Filtered essential prerequisite papers |
| `cleaned_filtered_papers.json` | Cleaned and validated paper titles |
| Downloaded PDFs | Stored in `reference_research_papers/` |



## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.



## 📜 Attribution

If you use this project, a shoutout would be awesome! 🙌 Credit helps it grow. 🚀

