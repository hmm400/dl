# === Program 8(a): Word2Vec ===
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")  # Added to resolve the LookupError

# Sample dataset
text = [
    "I love machine learning",
    "Word22Vec is a useful technique",
    "I love deep learning",
    "Natural language processing is interesting",
]

data = [word_tokenize(sentence.lower()) for sentence in text]

model = Word2Vec(
    sentences=data,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
)

print("Vector for 'learning':")
print(model.wv["learning"])

print("\nWords similar to 'learning':")
print(model.wv.most_similar("learning"))


# === Program 8(b): Doc2Vec ===
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

documents = [
    "I like machine learning",
    "Doc2Vec learns document embeddings",
    "Deep learning is powerful",
    "Natural language processing is fun",
]

tagged_data = [
    TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)])
    for i, doc in enumerate(documents)
]

model = Doc2Vec(
    tagged_data,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    epochs=40,
)

vec1 = model.dv['0']
vec2 = model.dv['2']
sim = cosine_similarity([vec1], [vec2])
print("Similarity between Document 0 and 2:", sim[0][0])

new_doc = "I enjoy learning NLP"
new_vec = model.infer_vector(word_tokenize(new_doc.lower()))

sim_new = cosine_similarity([new_vec], [model.dv['0']])
print("Similarity between new document and Document 0:", sim_new[0][0])

print("\nMost similar documents to Document 0:")
print(model.dv.most_similar('0'))