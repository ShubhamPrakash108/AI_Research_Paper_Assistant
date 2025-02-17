# 🧠 AI Research Assistant

## 📄 Project Overview
This project is a comprehensive AI-powered research assistant designed to aid academic researchers in exploring research papers, identifying essential prerequisite works, evaluating the difficulty of reference papers, and providing an interactive platform to query documents.

The system leverages Google Gemini LLM models and various LangChain utilities to automate literature analysis, paper evaluation, and document querying.

## 🌟 Features
- ✅ **Prerequisite Paper Prediction**: Identify prerequisite research papers needed to understand a target paper.
- ✅ **Paper Evaluation and Download**: Evaluate the relevance and difficulty level of suggested papers and download them from arXiv.
- ✅ **Interactive Paper Exploration**: Search for related videos, blogs, and interactively query the content of research papers.
- ✅ **Question & Answer Interface**: Ask questions about research documents and receive context-aware responses.

## 🛠️ Components

### 1️⃣ A1_reference_predictor.py
- 📝 Extracts text from a target research paper PDF.
- ✂️ Splits the content into chunks and queries the Gemini model to suggest prerequisite research papers.
- 🗃️ Outputs the results to `Research_Papers_DB.json` after deduplication.

### 2️⃣ A2_paper_check.py
- 🗂️ Loads the prerequisite paper suggestions from `Research_Papers_DB.json`.
- 🔍 Validates the relevance of each suggested paper with respect to the target paper using the Gemini model.
- 📥 Downloads validated papers from arXiv.
- 🏷️ Evaluates the difficulty level (Beginner, Intermediate, Expert) of each downloaded paper.
- 🗄️ Renames and organizes the papers based on difficulty level.

### 3️⃣ A3_individiual_paper_Explorer.py
- 🖥️ Streamlit-based interactive app.
- 📄 Allows users to select and view PDF papers.
- 🔎 Generates search queries for YouTube videos and academic blogs related to the paper.
- 🧠 Embeds the content into FAISS for document querying.
- 💬 Chat interface to interact with paper content using the Gemini model.

### 4️⃣ A4_ques_and_ans.py
- 🖥️ Streamlit-based document Q&A system.
- 📄 Embeds all target and reference papers into FAISS.
- ❓ Enables users to query documents with questions and receive context-driven answers.
- 📝 Displays conversation history and allows downloading it as a text file.

## ⚙️ Setup
### 📋 Prerequisites
- 🐍 Python 3.8+
- 📦 Install required packages:
```bash
pip install -r requirements.txt
```


### 🔑 API Keys
Ensure you have multiple free Google Gemini API keys and a YouTube API key. Replace the placeholder values in the following sections in all scripts:
```python
GEMINI_API_KEYS = ["YOUR_GEMINI_API_KEY_1", "YOUR_GEMINI_API_KEY_2", ...]
YOUTUBE_API_KEY = "YOUR_YOUTUBE_API_KEY" # In A3_individiual_paper_Explorer.py
```

### 📄 Target Research Paper
📂 **Place the actual research paper in PDF format inside the `target_RP/` folder.**

## ▶️ Usage
### 1️⃣ Prerequisite Paper Prediction
```bash
python A1_reference_predictor.py
```
### 2️⃣ Paper Validation and Difficulty Evaluation
```bash
python A2_paper_check.py
```
### 3️⃣ Interactive Paper Exploration
```bash
streamlit run A3_individiual_paper_Explorer.py
```
### 4️⃣ Document Q&A
```bash
streamlit run A4_ques_and_ans.py
```

## 📂 Folder Structure
```
.
├── target_RP/                     # 📄 Target research paper PDF
├── reference_research_papers/     # 📄 Downloaded prerequisite papers
├── Research_Papers_DB.json        # 📜 Suggested prerequisite papers
├── cleaned_filtered_papers.json   # ✅ Validated prerequisite papers
├── A1_reference_predictor.py
├── A2_paper_check.py
├── A3_individiual_paper_Explorer.py
├── A4_ques_and_ans.py
└── target_paper.json               # 🗃️ JSON storing the title of the target paper
```

## 📝 Output Files
- 📄 `output.json`: Raw extracted paper titles.
- 📜 `Research_Papers_DB.json`: Deduplicated paper titles.
- ✅ `filtered_papers.json`: Filtered essential prerequisite papers.
- 📑 `cleaned_filtered_papers.json`: Cleaned and validated paper titles.
- 📂 Downloaded PDFs stored in `reference_research_papers/`.

