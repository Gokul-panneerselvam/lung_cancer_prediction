import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_bleu(references, generated_list):
    bleu_scores = []
    smoothie = SmoothingFunction().method1  # Smoothing to handle short sentences
    for reference, generated in zip(references, generated_list):
        reference_tokens = [reference.split()]
        generated_tokens = generated.split()
        bleu_scores.append(sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smoothie))
    return bleu_scores

def evaluate_rouge(references, generated_list):
    rouge_scores = []
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    for reference, generated in zip(references, generated_list):
        scores = scorer.score(reference, generated)
        rouge_scores.append({key: scores[key].fmeasure for key in scores})
    return rouge_scores

def evaluate_meteor(references, generated_list):
    meteor_scores = []
    for reference, generated in zip(references, generated_list):
        reference = word_tokenize(reference)
        generated = word_tokenize(generated)
        meteor_scores.append(meteor_score([reference], generated))
    return meteor_scores

def calculate_average(scores):
    return sum(scores) / len(scores)

# Example Usage
references = ["The truck will arrive at the warehouse by 10 AM.", "The driver is expected at the station at 3 PM."]
generated_texts = ["The truck is expected at the warehouse around 10 AM.", "The driver will arrive at the station at 3 PM."]

# bleu_scores = evaluate_bleu(references, generated_texts)
# rouge_scores = evaluate_rouge(references, generated_texts)
# meteor_scores = evaluate_meteor(references, generated_texts)

# avg_bleu = calculate_average(bleu_scores)
# avg_rouge = {key: calculate_average([score[key] for score in rouge_scores]) for key in rouge_scores[0]}  # Calculate average for each ROUGE metric
# avg_meteor = calculate_average(meteor_scores)

# # Print results
# print(f"Overall BLEU Score: {avg_bleu:.4f}")
# print(f"Overall ROUGE Scores: {avg_rouge}")
# print(f"Overall METEOR Score: {avg_meteor:.4f}")

def calculate_overall_score(references, generated_list, bleu_weight=0.3, rouge_weight=0.3, meteor_weight=0.4):
    # Calculate individual scores
    bleu_scores = evaluate_bleu(references, generated_list)
    rouge_scores = evaluate_rouge(references, generated_list)
    meteor_scores = evaluate_meteor(references, generated_list)

    # Calculate average scores for each metric
    avg_bleu = calculate_average(bleu_scores)
    avg_rouge = {key: calculate_average([score[key] for score in rouge_scores]) for key in rouge_scores[0]}  # For ROUGE-1, ROUGE-2, ROUGE-L
    avg_meteor = calculate_average(meteor_scores)

    # Calculate the overall score as a weighted average
    overall_score = (
        bleu_weight * avg_bleu +
        rouge_weight * sum(avg_rouge.values()) / len(avg_rouge) +  # Averaging the ROUGE scores
        meteor_weight * avg_meteor
    )

    return avg_bleu, avg_rouge, avg_meteor, overall_score

# Example Usage
references = ["The truck will arrive at the warehouse by 10 AM.", "The driver is expected at the station at 3 PM."]
generated_texts = ["The truck is expected at the warehouse around 10 AM.", "The driver will arrive at the station at 3 PM."]

# Weights for BLEU, ROUGE, and METEOR scores (adjust as needed)
bleu_weight = 0.3
rouge_weight = 0.3
meteor_weight = 0.4

# Calculate scores
avg_bleu, avg_rouge, avg_meteor, overall_score = calculate_overall_score(
    references, generated_texts, bleu_weight, rouge_weight, meteor_weight
)

# Print results
print(f"Average BLEU Score: {avg_bleu:.4f}")
print(f"Average ROUGE Scores: {avg_rouge}")
print(f"Average METEOR Score: {avg_meteor:.4f}")
print(f"Overall Score: {overall_score:.4f}")