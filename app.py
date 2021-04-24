from util.match_face import is_match
from util.embedding import get_embeddings

if __name__ == "__main__":
    filenames = ['1.jpg', '2.jpg']
    # get embeddings file filenames
    embeddings = get_embeddings(filenames)
    is_match(embeddings[0], embeddings[1])