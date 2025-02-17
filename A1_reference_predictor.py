from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
import re
import json

file_path = "target_RP/transformer_paper.pdf"

def text_extractor(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load_and_split()
    all_text = "\n".join(doc.page_content for doc in documents)
    return all_text

all_text = text_extractor(file_path=file_path)

text_splitter = CharacterTextSplitter(
    separator=" ",       
    chunk_size=400,      
    chunk_overlap=2     
)

chunks = text_splitter.split_text(all_text)


GEMINI_API_KEY = ""
GEMINI_API_KEYS = ["","","","","","","","","","",""]

# provided_title = "Attention is All you need" 
provided_title = input("Enter the name of the research paper: ")

data = {
    "provided_title": provided_title
}

with open("target_paper.json", "w") as file:
    json.dump(data, file, indent=4)


RESPONSE = []

USING_API_KEY = GEMINI_API_KEYS[0]
SEL_API_KEY = 0

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")

    if i%14 == 0:
        SEL_API_KEY = SEL_API_KEY + 1
        if SEL_API_KEY > len(GEMINI_API_KEYS):
            SEL_API_KEY = 0
        
        USING_API_KEY = GEMINI_API_KEYS[SEL_API_KEY]

    print(f"API NUMBER: {SEL_API_KEY} and API KEY: {USING_API_KEY}")

    llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=1,
    google_api_key=USING_API_KEY,
    max_tokens=None,
    timeout=None
    )

    messages = [
    (
        "system",
        (
            "You are a helpful academic research assistant specializing in identifying prerequisite research papers. "
            "Your task is to determine which additional prerequisites research papers are essential for understanding the study topic, based on the input research paper. "
            "Consider closely related papers if they significantly contribute to understanding the topic. " ##
            "Do not include the research paper provided by the user. Specifically, do not include any paper whose title matches. "
            f"'{provided_title}'. For each relevant paper you identify, output only its title, formatted exactly as follows:\n\n"
            "Select only the most relevant and essential research papers, avoiding any marginally related or redundant works."
            
            "Title: [Paper Title]\n"
            # "DOI: [DOI Number]\n\n"
            
                    )
        
    ),
    (
        # "human", chunks[i]
        "human", f"List only the most essential prerequisite research papers needed to understand {chunks[i]}, excluding the provided paper itself that is {provided_title}"
        ),
    ]

    ai_msg = llm.invoke(messages)
    print(ai_msg.content)
    response_text = ai_msg.content
    paper_entries = re.findall(r"Title:\s*(.*?)(?:\n|$)", response_text)
    
    for title in paper_entries:
        RESPONSE.append({
            # "chunk_id": i,
            "title": title.strip(),
            # "doi": doi.strip()
        })

output_json = json.dumps(RESPONSE, indent=4)
print(output_json)

with open("output.json", "w") as json_file:
    json_file.write(output_json)

print("JSON file saved as 'output.json' in the current working directory.")

import json

with open("output.json", "r") as json_file:
    data = json.load(json_file)

unique_data = []
seen = set()

duplicates_found = 0

for item in data:
    title = item["title"].strip().lower()
    # doi = item["doi"].strip().lower()
    identifier = (title)

    if identifier not in seen:
        seen.add(identifier)
        unique_data.append(item)
    else:
        duplicates_found += 1

print(f"Duplicates found and removed: {duplicates_found}")

with open("Research_Papers_DB.json", "w") as json_file:
    json.dump(unique_data, json_file, indent=4)

print("Duplicates removed and saved as 'deduplicated_output.json'.")

