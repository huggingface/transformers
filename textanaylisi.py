from transformers import pipeline

# Load sentiment-analysis pipeline
classifier = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = classifier(text)[0]
    print(f"Sentiment: {result['label']}, Score: {result['score']:.2f}")

if __name__ == "__main__":
    text = input("Enter text: ")
    analyze_sentiment(text)
