import wikipediaapi
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')

# Initialize Wikipedia API
wiki = wikipediaapi.Wikipedia('en')

# Specify a user agent with language information
user_agent = 'MyWikiBot/1.0 (Language: en; https://github.com/Amit-2013/LearningGPT)'

# Set user agent directly after creating the Wikipedia object
wiki.headers = {'User-Agent': user_agent}

# Function to fetch Wikipedia data
def get_wikipedia_text(topic):
    page = wiki.page(topic)
    if page.exists():
        return page.text
    else:
        return None

# Function to preprocess text
def preprocess_text(text):
    sentences = sent_tokenize(text)
    return ' '.join(sentences)

# Function to generate response to user query
def generate_response(query, corpus, vectorizer):
    query = preprocess_text(query)
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, corpus)
    most_similar_index = similarities.argmax()
    return corpus[most_similar_index].toarray().flatten()  # Convert sparse matrix to array and flatten it

# Main function
def main():
    # User input for topic
    topic = input("Enter a Wikipedia topic: ")

    # Extract text from Wikipedia article
    article_text = get_wikipedia_text(topic)

    if article_text:
        # Preprocess article text
        preprocessed_text = preprocess_text(article_text)

        # Vectorize preprocessed text
        vectorizer = TfidfVectorizer()
        corpus = vectorizer.fit_transform([preprocessed_text])

        # Chat loop
        print("Chatbot: Hello! Ask me anything about", topic)
        while True:
            user_query = input("You: ")
            if user_query.lower() == 'exit':
                print("Chatbot: Goodbye!")
                break
            response = generate_response(user_query, corpus, vectorizer)
            print("Chatbot:", response)
    else:
        print("Topic not found on Wikipedia. Please try again.")

if __name__ == "__main__":
    main()
