import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


corpus = ["Hello. How can I help you?", 
          "I am a chatbot here to assist you.", 
          "I can help answer basic questions about AI.",
          "AI is the simulation of human intelligence by machines.",
          "Robotics deals with the design and creation of robots.",
          "Gaming is the creation of virtual spaces where people can control characters for fun.", 
          "Gaming uses procedural generation to create content algorithmically", 
          "Machine learning is a section of AI that allows systems to learn from data without being told", 
          "SLAM stands for Simultaneous Localization and Mapping, which robots use to create maps of unknown places.", 
          "Goodbye, take care."
          ]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

def chatbot_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, X)
    best_match_idx = similarities.argmax()
    return corpus[best_match_idx]

print("Chatbot: Hello. Ask me about AI, Robotics, or Gaming (type quit to exit).")
while True:
    user_input = input("You: ").lower()
    if user_input in ["quit", "bye", "exit"]:
        print("Chatbot: Goodbye")
        break
    response = chatbot_response(user_input)
    print("Chatbot: ", response)