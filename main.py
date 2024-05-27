import openai
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set OpenAI API key
openai.api_key = 'your_openai_api_key'

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess text
def preprocess_text(text):
    # Tokenization
    sentences = sent_tokenize(text)
    
    # Lowercasing, removing stopwords, lemmatization
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    preprocessed_sentences = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence.lower())
        filtered_words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
        preprocessed_sentences.append(' '.join(filtered_words))
    
    return ' '.join(preprocessed_sentences)

# Function to generate response to user query
def generate_response(query, corpus, vectorizer):
    query = preprocess_text(query)
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, corpus)
    most_similar_index = similarities.argmax()
    response = corpus[most_similar_index].toarray().flatten()

    # Reshape the response array to be 2D
    response = response.reshape(1, -1)

    # Inverse transform the response to convert it back to text
    text_response = vectorizer.inverse_transform(response)

    # Join the words in the text response
    return ' '.join(text_response[0])

# Function to process user query
def process_user_query(query, c, vectorizer):
    c.execute('''SELECT text FROM learning''')
    texts = c.fetchall()
    preprocessed_texts = [preprocess_text(text[0]) for text in texts]
    corpus = vectorizer.transform(preprocessed_texts)
    response = generate_response(query, corpus, vectorizer)
    return response

# Main function
def main():
    # Create or connect to the database
    with sqlite3.connect("learning_bot.db") as conn:
        c = conn.cursor()

        # Create table if not exists
        c.execute('''CREATE TABLE IF NOT EXISTS learning (id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT, text TEXT)''')

        # Initialize TfidfVectorizer
        vectorizer = TfidfVectorizer()

        # User input loop
        print("Chatbot: I'm ready to learn and answer your questions.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'learn':
                topic = input("Enter the topic you want me to learn about: ")
                # Fetch text from ChatGPT
                response = openai.ChatCompletion.create(
                    model="text-davinci-003",  # You can adjust the model based on your preference
                    messages=[
                        {"role": "system", "content": "You are a chatbot."},
                        {"role": "user", "content": f"Tell me about {topic}."},
                    ],
                )
                text = response.choices[0].message['content']
                # Preprocess and insert the fetched text into the database
                preprocessed_text = preprocess_text(text)
                c.execute('''INSERT INTO learning (topic, text) VALUES (?, ?)''', (topic, preprocessed_text))
                conn.commit()
                print("Chatbot: I have learned about:", topic)
            elif user_input.lower() == 'exit':
                print("Chatbot: Goodbye!")
                break
            else:
                # Fetch all text from the database and fit the vectorizer
                c.execute('''SELECT text FROM learning''')
                texts = c.fetchall()
                preprocessed_texts = [preprocess_text(text[0]) for text in texts]
                vectorizer.fit(preprocessed_texts)
                # Process user's query
                response = process_user_query(user_input, c, vectorizer)
                print("Chatbot:", response)

if __name__ == "__main__":
    main()
