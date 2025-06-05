import streamlit as st
import requests
# import json

API_BASE_URL = "http://127.0.0.1:8000"

def extract_links(url):
    endpoint = f"{API_BASE_URL}/extract_links"
    payload = {"url": url}
    response = requests.post(endpoint, json=payload)
    if response.status_code == 200:
        return response.json()["unique_links"]
    else:
        raise Exception(f"Failed to extract links: {response.text}")

def extract_text(urls):
    endpoint = f"{API_BASE_URL}/extract_text"
    response = requests.post(endpoint, json=urls)
    if response.status_code == 200:
        return response.json()["file_saved"]
    else:
        raise Exception(f"Failed to extract text: {response.text}")

def perform_rag(file_path, prompt):
    endpoint = f"{API_BASE_URL}/rag"
    payload = {"file_path": file_path, "prompt": prompt}
    response = requests.post(endpoint, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to perform RAG: {response.text}")

def main():
    st.title("Web RAG System")

    url = st.text_input("Enter the URL to start with:")
    prompt = st.text_area("Enter your prompt for RAG:")
    option = st.radio("Choose data source:", ("Data from multiple links", "Only home page data"))

    if st.button("Process"):
        try:
            with st.spinner("Processing..."):
                if option == "Data from multiple links":
                    st.info("Extracting links...")
                    links = extract_links(url)
                    sample_links = links[:5]
                    st.info("Extracting text from links...")
                    file_path = extract_text(sample_links)
                else:
                    st.info("Extracting text from home page...")
                    file_path = extract_text([url])

                st.info("Performing RAG...")
                result = perform_rag(file_path, prompt)

                st.subheader("RAG Result")
                st.write(f"**User Query:** {result['user_query']}")
                st.write(f"**Assistant Response:** {result['assistant_response']}")
                
                st.subheader("Sources")
                st.text(result['sources'])

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()