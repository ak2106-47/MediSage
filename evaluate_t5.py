import json
from t5_rewriter import T5Rewriter
import evaluate
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import nltk

nltk.download('punkt')

# === Load QA Paraphrase Data ===
with open("qa_paraphrase.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# === Init models ===
rewriter = T5Rewriter()
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === Store for BLEU and Cosine ===
original_questions = []
rewritten_questions = []
cosine_scores = []

# === Evaluation loop ===
print("🔍 Evaluating T5 rewrites...\n")
for item in tqdm(qa_data):
    original = item["original_question"]
    rewritten = item["paraphrased_question"]  # <-- use already paraphrased question

    original_questions.append(original)
    rewritten_questions.append(rewritten)

    # Cosine similarity
    emb1 = embedder.encode(original, convert_to_tensor=True)
    emb2 = embedder.encode(rewritten, convert_to_tensor=True)
    cosine = util.pytorch_cos_sim(emb1, emb2).item()
    cosine_scores.append(cosine)

# === Compute metrics ===
rouge_result = rouge.compute(
    predictions=rewritten_questions,
    references=original_questions,
    use_stemmer=True
)
bleu_result = bleu.compute(
    predictions=rewritten_questions,
    references=[[ref] for ref in original_questions]
)

# === Show results ===
print("=== T5 Rewrite Evaluation ===")
print(f"ROUGE-1 F1       : {rouge_result['rouge1']:.4f}")
print(f"ROUGE-2 F1       : {rouge_result['rouge2']:.4f}")
print(f"ROUGE-L F1       : {rouge_result['rougeL']:.4f}")
print(f"BLEU Score       : {bleu_result['bleu']:.4f}")
print(f"Avg Cosine Sim.  : {sum(cosine_scores) / len(cosine_scores):.4f}")
print(f"Total Examples   : {len(original_questions)}")
