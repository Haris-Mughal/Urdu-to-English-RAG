## 📄 Retrieval Augmented Generation: Urdu PDF OCR, Translation & Q&A

An intelligent document processing application that extracts text from Urdu PDFs using OCR, translates content to English, and enables Q&A interactions in both Urdu and English. Powered by Retrieval Augmented Generation (RAG) techniques for accurate, context-aware responses.

## Features

- **Urdu PDF OCR**: Extract text directly from Urdu PDF documents with advanced OCR capabilities.
- **Automatic Translation**: Seamlessly translate extracted Urdu content to English for broader accessibility.
- **Bilingual Q&A**: Ask questions in Urdu or English and receive intelligent answers based on document content.
- **RAG-Powered Responses**: Utilizes Retrieval Augmented Generation for accurate, contextually relevant answers.
- **Interactive Interface**: Built with Streamlit for a seamless user experience.

## Installation
1. **Install the required packages**:
   ```
   pip install -r requirements.txt
   ```
2. **Set up your API key**:
   - Obtain your API key from Groq and set it as an environment variable (`GROQ_API_KEY`)
3. **Run the app**:
   ```
   streamlit run app.py
   ```
## Usage

1. **Upload Urdu PDF**: Use the file uploader to select your Urdu PDF document.
2. **Extract Text**: The application automatically performs OCR to extract text from the PDF.
3. **Automatic Translation**: Extracted Urdu text is translated to English for processing.
4. **Ask Questions**: Enter your questions in Urdu or English based on the document content.
5. **Get Answers**: Receive intelligent, context-aware answers powered by RAG technology.

## Acknowledgments

- **Streamlit**: Interactive web framework for building the application
- **Groq LLM**: High-performance language model for Q&A capabilities
- **Sentence Transformers**: For semantic text embeddings and retrieval
- **Pytesseract**: For OCR text extraction from Urdu PDFs
- **Google Translator**: For seamless Urdu to English translation
- **Langchain**: For RAG pipeline implementation

> ~ Muhammad Haris Ahsan