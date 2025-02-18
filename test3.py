import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math
import random
from sklearn.metrics import f1_score
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
# nltk.download('punkt')

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
def evaluate_meteor(references, generated_list):
    meteor_scores = []
    for reference, generated in zip(references, generated_list):
        reference = word_tokenize(reference)
        generated = word_tokenize(generated)
        meteor_scores.append(meteor_score([reference], generated))
    return meteor_scores

# Perplexity Calculation (for language models)
def calculate_perplexity(text, model):
    tokens = nltk.word_tokenize(text)
    score = 0.0
    for token in tokens:
        score += math.log(random.uniform(0.5, 1.5))  # Placeholder for actual model
    return math.exp(-score / len(tokens))

# Diversity Metrics (Distinct n-grams)
def calculate_diversity(text, n=2):
    tokens = nltk.word_tokenize(text)
    n_grams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    unique_n_grams = set(n_grams)
    return len(unique_n_grams) / len(n_grams) if n_grams else 0

# F1 Score Calculation (for n-grams)
def calculate_f1(reference, generated, n=1):
    reference_tokens = set(nltk.ngrams(nltk.word_tokenize(reference), n))
    generated_tokens = set(nltk.ngrams(nltk.word_tokenize(generated), n))

    # Calculate Precision and Recall
    true_positives = len(reference_tokens & generated_tokens)
    precision = true_positives / len(generated_tokens) if generated_tokens else 0
    recall = true_positives / len(reference_tokens) if reference_tokens else 0

    # Calculate F1 Score
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

# Example usage for open-ended or generative task evaluation with multiple pairs
import numpy as np

import numpy as np

def evaluate_task(reference_texts, generated_texts):
    # Metrics initialization
    results = {
        "BLEU": [],
        "ROUGE1": [],
        "ROUGE2": [],
        "ROUGE_L": [],
        "METEOR": [],
        "Cosine Similarity": [],
        "Perplexity": [],
        "Diversity (Distinct 1)": [],
        "Diversity (Distinct 2)": [],
        "F1 (1-gram)": [],
        "F1 (2-gram)": []
    }

    # Iterate over pairs of reference and generated texts
    for reference_text, generated_text in zip(reference_texts, generated_texts):
        # BLEU Score
        bleu_score = evaluate_bleu(reference_text, generated_text)
        results["BLEU"].append(bleu_score)

        # ROUGE Score
        rouge_scores = evaluate_rouge(reference_text, generated_text)
        results["ROUGE1"].append(rouge_scores["rouge1"])
        results["ROUGE2"].append(rouge_scores["rouge2"])
        results["ROUGE_L"].append(rouge_scores["rougeL"])

        # METEOR Score
        meteor = evaluate_meteor(reference_text, generated_text)
        results["METEOR"].append(meteor)

        # Cosine Similarity (using embeddings)
        cosine_sim = cosine_similarity_score(reference_text, generated_text)
        results["Cosine Similarity"].append(cosine_sim)

        # Perplexity (dummy language model)
        perplexity = calculate_perplexity(generated_text, model=None)
        results["Perplexity"].append(perplexity)

        # Diversity (Distinct 1-gram and 2-gram)
        diversity_1 = calculate_diversity(generated_text, n=1)
        diversity_2 = calculate_diversity(generated_text, n=2)
        results["Diversity (Distinct 1)"].append(diversity_1)
        results["Diversity (Distinct 2)"].append(diversity_2)

        # F1 Score (1-gram and 2-gram)
        f1_1gram = calculate_f1(reference_text, generated_text, n=1)
        f1_2gram = calculate_f1(reference_text, generated_text, n=2)
        results["F1 (1-gram)"].append(f1_1gram)
        results["F1 (2-gram)"].append(f1_2gram)

    # Compute overall results, handling ROUGE and other dictionary-based scores
    overall_results = {}
    for key, values in results.items():
        if isinstance(values[0], dict):  # Handle ROUGE scores or similar dicts
            overall_results[key] = {sub_key: np.mean([val[sub_key] for val in values]) for sub_key in values[0].keys()}
        else:
            overall_results[key] = np.mean(values)

    return overall_results

# Example usage with multiple references and generated texts
reference_texts = [
    "The truck will arrive at the warehouse by 10 AM.",
    "The package will be delivered by noon.",
    "The meeting is scheduled for 9 AM."
]

generated_texts = [
    "The truck is expected at the warehouse around 10 AM.",
    "The package will reach its destination by noon.",
    "The meeting will begin at 9 AM."
]

# Evaluate the generative task with multiple reference/generated pairs
results = evaluate_task(reference_texts, generated_texts)

# Print overall evaluation results
print("Overall Evaluation Results:")
for metric, score in results.items():
    if isinstance(score, dict):  # For cases like ROUGE, which is a dictionary of sub-scores
        print(f"{metric}:")
        for sub_metric, sub_score in score.items():
            print(f"  {sub_metric}: {sub_score:.4f}")
    else:
        print(f"{metric}: {score:.4f}")
