from langchain.document_loaders import PyPDFLoader


def loadPDF(filename):
    loader = PyPDFLoader(filename)
    pages = loader.load_and_split()
    return pages
