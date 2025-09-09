import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Chatbot:
    def __init__(self, corpus, fallback_responses=None, threshold=0.2):
        """
        Initializes the AI with a list of possible reponses.

        Args:
            corpus (list[str]): List of knowledge.
            fallback_responses (list[str]): random responses if no answer is found.
            threshold (float): Minimum similarity score to accept answer.
        """

        self.corpus = corpus
        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(corpus)
        self.threshold = threshold
        self.fallback_responses = fallback_responses or [
            "Hmm, I'm not sure about this",
            "Could you rephrase?",
            "That's interesting, tell me more.", 
            "I don't have an answer for that."
        ]
        self.history = []

    def preprocess(self, text):
        """ Makes the input question lowercase with no punctuation, ect."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "" , text)
        return text.strip()
    
    def get_best_response(self, user_input):
        """Finds the best response for the question asked"""
        user_input = self.preprocess(user_input)
        user_vec = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vec, self.X).flatten()

        best_score = similarities.max()
        best_idx = similarities.argmax()

        if best_score < self.threshold:
            return random.choice(self.fallback_responses)
        return self.corpus[best_idx]
    
    def respond(self, user_input):
        """Creates a response and adds it to history"""
        response = self.get_best_response(user_input)
        self.history.append(("User", user_input))
        self.history.append(("Bot", response))
        return response
    
    def show_history(self):
        """Prints the conversation history"""
        print("\n--- Conversation Log ---")
        for speaker, text in self.history:
            print(f"{speaker}: {text}")
        print("------------------\n")
                                             

if __name__ == "__main__":
    corpus = [
        "Hello. How can I help you?", 
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

    chatbot = Chatbot(corpus, threshold=.25)

    print("Chatbot: Hello. Ask me about AI, Robotics, or Gaming (type quit to exit).")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "bye", "exit"]:
            print("Chatbot: Goodbye")
            chatbot.show_history()
            break
        response = chatbot.respond(user_input)
        print("Chatbot: ", response)