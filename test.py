import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('wordnet')
def evaluate_bleu(reference, generated):
    reference_tokens = [reference.split()]
    generated_tokens = generated.split()
    smoothie = SmoothingFunction().method1  # Smoothing to handle short sentences
    return sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smoothie)

def evaluate_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {key: scores[key].fmeasure for key in scores}

def evaluate_meteor(reference_text, generated_text):
    # Tokenize the input texts
    reference = word_tokenize(reference_text)
    generated = word_tokenize(generated_text)
    
    # Calculate METEOR score
    return meteor_score([reference], generated)

# Example Usage
reference_text = "The truck will arrive at the warehouse by 10 AM."
generated_text = "The truck is expected at the warehouse around 10 AM."

bleu_score = evaluate_bleu(reference_text, generated_text)
rouge_scores = evaluate_rouge(reference_text, generated_text)
meteor = evaluate_meteor(reference_text, generated_text)

print(f"BLEU Score: {bleu_score:.4f}")
print(f"ROUGE Scores: {rouge_scores}")
print(f"METEOR Score: {meteor:.4f}")
