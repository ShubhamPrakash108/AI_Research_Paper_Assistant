from langchain_google_genai import ChatGoogleGenerativeAI
import json
import random
import re
import arxiv
import os
from langchain_community.document_loaders import PyPDFLoader

with open("target_paper.json", "r") as file:
    data = json.load(file)

provided_title = data["provided_title"]

print(provided_title)

GEMINI_API_KEYS = ["","","","","","","","","","",""]

with open("Research_Papers_DB.json", "r") as file:
    papers = json.load(file)



filtered_papers = []

for paper in papers:
    print(paper["title"])

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-thinking-exp",
        temperature=1,
        google_api_key=random.choice(GEMINI_API_KEYS),
        max_tokens=None,
        timeout=None
    )

    messages = [
        (
            "system",(
            # "You are a research assistant. Your task is to check if a given research paper is essential to understand '{provided_title}'. "
            # "If the paper is essential, respond with 'Yes'. If it is not, respond with 'No'. "
            # "Only reply with 'Yes' or 'No'. No explanations or extra text."

            # "Check if the given paper is essential to understand '{provided_title}'. "
            # "Respond only with 'Yes' or 'No'. Nothing else."

            # "You are an expert research assistant with comprehensive knowledge of academic research papers. "
            
            
            # f"Evaluate whether the provided research paper is crucial for understanding the research paper'{provided_title}' as prerequisites. "
            # "Consider closely related papers if they significantly contribute to understanding the topic. "
            # "Use your full expertise to ensure the most accurate assessment. "
            # "Respond with 'Yes' if the paper is essential; otherwise, respond with 'No'. "
            # "Give no explanations or additional text.")

            """
You are a research assistant specializing in scientific literature analysis.
Your task is to determine whether a given research paper is an essential prerequisite to understanding the target paper: '{provided_title}'.

Criteria for Selection:
1. **Fundamental Theories**: If the paper introduces core concepts that the target paper builds upon, it is essential.
2. **Key References**: If the target paper explicitly cites this paper as foundational, it is essential.
3. **Methodology Dependence**: If the target paper's experiments or models require an understanding of this paper’s approach, it is essential.
4. **Highly Cited in the Field**: If the paper is a widely recognized key work in this research area, it is likely essential.

Examples:
- Target Paper: "Attention is All You Need"
  - Candidate: "Neural Machine Translation by Jointly Learning to Align and Translate" → **YES** (Direct precursor to Transformer models)
  - Candidate: "A Study on Deep Learning Optimizers" → **NO** (Relevant but not a strict prerequisite)

Only respond with 'Yes' or 'No'.
""")
        ),
        (
            "human",
            (
                f"""
Carefully analyze the research paper '{provided_title}'. To determine its essential prerequisites, consider the following factors:

1. **Fundamental Concepts** – Does the paper '{paper['title']}' introduce key theories or methods that the target paper builds upon?
2. **Direct Citations** – Is this paper explicitly referenced as a critical source in '{provided_title}'?
3. **Methodological Dependence** – Does the target paper rely on experimental techniques, models, or frameworks introduced in '{paper['title']}'?
4. **Field Significance** – Is this paper widely recognized as a foundational work in the same domain?

Based on these criteria, is '{paper['title']}' an essential prerequisite to understand '{provided_title}'?  
Respond **only** with 'Yes' or 'No'. No explanations or extra text.
"""
            )
        
            # was working better (f"Is the paper titled '{paper['title']}' essential as prerequisites to understand '{provided_title}'? Respond with 'Yes' or 'No'.")
        ),
    ]

    ai_msg = llm.invoke(messages)
    response_text = ai_msg.content.strip()

    print(response_text)

    if response_text == "Yes":
        filtered_papers.append(paper)


with open("filtered_papers.json", "w") as file:
    json.dump(filtered_papers, file, indent=4)

with open("filtered_papers.json", "r") as file:
    filtered_papers = json.load(file)

with open("target_paper.json", "r") as file:
    provided_title = json.load(file)["provided_title"]

def clean_title(title):
    return re.sub(r"[^\w\s]", "", title).strip().lower()

clean_provided_title = clean_title(provided_title)

unique_titles = set()
cleaned_papers = []

for paper in filtered_papers:
    cleaned_title = clean_title(paper["title"])

    if cleaned_title == clean_provided_title:
        continue

    if cleaned_title not in unique_titles:
        unique_titles.add(cleaned_title)
        cleaned_papers.append({"title": paper["title"]})


with open("cleaned_filtered_papers.json", "w") as file:
    json.dump(cleaned_papers, file, indent=4)

print(f"{len(cleaned_papers)} papers saved to 'cleaned_filtered_papers.json'.")


def sanitize_filename(title):
    """Sanitize filename to remove or replace problematic characters."""
    return re.sub(r'[<>:"/\\|?*]', '', title.replace(" ", "_"))


def download_arxiv_paper(paper_name, download_dir="reference_research_papers"):
    """Searches for a research paper on arXiv by name and downloads the first exact match if found."""
    client = arxiv.Client()

    search = arxiv.Search(
        query=f"ti:\"{paper_name}\"",
        max_results=5,
        sort_by=arxiv.SortCriterion.Relevance
    )

    os.makedirs(download_dir, exist_ok=True)

    results = list(client.results(search))

    for result in results:
        if result.title.lower() == paper_name.lower():
            paper_title = sanitize_filename(result.title)
            pdf_path = os.path.join(download_dir, f"{paper_title}.pdf")
            print(f"Downloading exact match: {result.title}")
            result.download_pdf(filename=pdf_path)
            print(f"Downloaded to: {pdf_path}")
            return pdf_path

    if results:
        best_match = results[0]
        paper_title = sanitize_filename(best_match.title)
        pdf_path = os.path.join(download_dir, f"{paper_title}.pdf")
        print(f"No exact match. Downloading best match: {best_match.title}")
        best_match.download_pdf(filename=pdf_path)
        print(f"Downloaded to: {pdf_path}")
        return pdf_path

    print(f"No suitable paper found on arXiv for: {paper_name}")
    return None


with open("cleaned_filtered_papers.json", "r") as file:
    papers = json.load(file)

downloaded_papers = []

for paper in papers:
    paper_name = paper["title"]
    pdf_path = download_arxiv_paper(paper_name)
    if pdf_path:
        downloaded_papers.append({"title": paper_name, "pdf_path": pdf_path})

print(f"\nDownloaded {len(downloaded_papers)} papers.")


folder_path = "reference_research_papers"

def extract_text_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load_and_split()
    all_text = "\n".join(doc.page_content for doc in documents)
    return all_text

def evaluate_difficulty(text, file_name):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-pro-exp-02-05",
        temperature=0,
        google_api_key=random.choice(GEMINI_API_KEYS)
    )

    messages = [
        ("system", "You are a research assistant specializing in academic research. Your task is to evaluate the difficulty level of a research paper based on its content. Categorize it as 'Beginner', 'Intermediate', or 'Expert'. Output only the difficulty level without any extra text."),
        ("human", text[:5000])  
    ]

    response = llm.invoke(messages)
    result = response.content.strip()
    print(f"Paper: {file_name} | Difficulty: {result}")
    return result

def main():
    paper_difficulties = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            text = extract_text_from_pdf(file_path)
            difficulty = evaluate_difficulty(text, file_name)
            paper_difficulties.append((file_name, difficulty))

    difficulty_order = {'Beginner': 1, 'Intermediate': 2, 'Expert': 3}
    paper_difficulties.sort(key=lambda x: difficulty_order.get(x[1], 4))

    for index, (file_name, difficulty) in enumerate(paper_difficulties, start=1):
        old_path = os.path.join(folder_path, file_name)
        new_file_name = f"{index}_{file_name}"
        new_path = os.path.join(folder_path, new_file_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {file_name} -> {new_file_name}")

if __name__ == "__main__":
    main()
