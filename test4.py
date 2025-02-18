import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math
import random

# Download necessary NLTK data
nltk.download('punkt')

# Cosine Similarity using SBERT
def cosine_similarity_score(reference, generated):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode([reference, generated])
    cosine_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return cosine_sim

# BLEU score calculation
def evaluate_bleu(reference, generated):
    reference_tokens = [nltk.word_tokenize(reference)]
    generated_tokens = nltk.word_tokenize(generated)
    smoothie = SmoothingFunction().method1  # Smoothing for short sentences
    return sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smoothie)

# ROUGE score calculation
def evaluate_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {key: scores[key].fmeasure for key in scores}

# METEOR score calculation
def evaluate_meteor(reference, generated):
    return meteor_score([reference], generated)

# Perplexity Calculation (for language models)
def calculate_perplexity(text, model):
    # Tokenize the text and calculate the perplexity for each token
    tokens = nltk.word_tokenize(text)
    score = 0.0
    for token in tokens:
        # Dummy probability (for demonstration purposes)
        score += math.log(random.uniform(0.5, 1.5))  # Using random probability as placeholder
    return math.exp(-score / len(tokens))

# Diversity Metrics (Distinct n-grams)
def calculate_diversity(text, n=2):
    tokens = nltk.word_tokenize(text)
    n_grams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    unique_n_grams = set(n_grams)
    return len(unique_n_grams) / len(n_grams) if n_grams else 0

# Example usage for open-ended or generative task evaluation
def evaluate_task(reference_text, generated_texts):
    # Metrics initialization
    results = {
        "BLEU": [],
        "ROUGE": [],
        "METEOR": [],
        "Cosine Similarity": [],
        "Perplexity": [],
        "Diversity (Distinct 1)": [],
        "Diversity (Distinct 2)": []
    }

    # Evaluate each generated text
    for generated_text in generated_texts:
        # BLEU Score
        bleu_score = evaluate_bleu(reference_text, generated_text)
        results["BLEU"].append(bleu_score)

        # ROUGE Score
        rouge_scores = evaluate_rouge(reference_text, generated_text)
        results["ROUGE"].append(rouge_scores)

        # METEOR Score
        meteor = evaluate_meteor(reference_text, generated_text)
        results["METEOR"].append(meteor)

        # Cosine Similarity (using embeddings)
        cosine_sim = cosine_similarity_score(reference_text, generated_text)
        results["Cosine Similarity"].append(cosine_sim)

        # Perplexity (dummy language model)
        perplexity = calculate_perplexity(generated_text, model=None)  # No actual language model here
        results["Perplexity"].append(perplexity)

        # Diversity (Distinct 1-gram and 2-gram)
        diversity_1 = calculate_diversity(generated_text, n=1)
        diversity_2 = calculate_diversity(generated_text, n=2)
        results["Diversity (Distinct 1)"].append(diversity_1)
        results["Diversity (Distinct 2)"].append(diversity_2)

    # Aggregate scores for overall evaluation
    overall_results = {key: np.mean(values) for key, values in results.items()}
    
    return overall_results

# Example usage
reference_text = "The truck will arrive at the warehouse by 10 AM."
generated_texts = [
    "The truck is expected at the warehouse around 10 AM.",
    "The truck will reach the warehouse at approximately 10 AM.",
    "The truck should be at the warehouse by 10 in the morning."
]

# Evaluate the generative task
results = evaluate_task(reference_text, generated_texts)

# Print overall evaluation results
print("Overall Evaluation Results:")
for metric, score in results.items():
    print(f"{metric}: {score:.4f}")
