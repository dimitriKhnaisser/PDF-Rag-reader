from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str):

    splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )

    return splitter.split_text(text)


# What these values mean
# chunk_size = 500
# each chunk ~500 characters
# keeps context small enough for embeddings
# chunk_overlap = 100
# repeats last 100 characters in next chunk
# prevents losing context between chunks