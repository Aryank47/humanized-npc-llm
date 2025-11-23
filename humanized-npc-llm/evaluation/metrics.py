"""
Enhanced Metrics Module for NPC-LLM Evaluation
Includes: Schema validation, persona faithfulness, hallucination detection, 
diversity metrics, and statistical analysis.
"""

import json
import re
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import jsonschema
import numpy as np
import nltk
from nltk.util import ngrams
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import logging
from scipy import stats
import warnings

# Suppress specific scipy warnings about invalid values in CI calculations
warnings.filterwarnings('ignore', message='invalid value encountered in multiply')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.stats')

# Setup logging
log = logging.getLogger(__name__)

# --- Constants ---
CONTENT_WORD_MIN_LENGTH = 3
NLI_CONTRADICTION_THRESHOLD = 0.75
MAX_UTTERANCE_TOKENS = 60
BANLIST = ["lol", "tbh", "omg", "btw", "afk", "brb"]

# Common conversational words (not considered hallucinations)
CONVERSATIONAL_WORDS = {
    "yes", "no", "maybe", "perhaps", "well", "indeed", "sure", "okay", "alright",
    "hello", "hi", "hey", "greetings", "goodbye", "bye", "farewell",
    "please", "thanks", "thank", "sorry", "excuse", "pardon",
    "very", "quite", "rather", "really", "truly", "definitely", "certainly",
    "think", "know", "believe", "suppose", "guess", "feel", "see", "hear",
    "could", "would", "should", "might", "must", "can", "will", "shall",
    "the", "and", "but", "for", "with", "from", "about", "into", "through",
    "this", "that", "these", "those", "what", "which", "who", "when", "where", "why", "how"
}

# --- 1. Schema & Constraint Metrics ---

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "utterance": {"type": "string"},
        "mood": {"type": "string"}  # Removed enum to be more flexible
    },
    "required": ["utterance", "mood"],
    "additionalProperties": False
}

# Expected moods for validation (soft check)
EXPECTED_MOODS = {
    "calm", "angry", "wary", "relieved", "happy", "sad", "confused", 
    "curious", "friendly", "grumpy", "helpful", "annoyed", "neutral",
    "excited", "fearful", "suspicious", "amused", "bored", "nervous",
    "guarded", "cautious", "casual", "insistent", "serious", "aggressive",
    "determined", "welcoming", "sympathetic", "interested", "thoughtful",
    "relaxed", "tense", "playful", "melancholic", "proud", "humble",
    "confident", "shy", "eager", "reluctant", "patient", "impatient"
}


def validate_schema(response: str) -> Tuple[bool, bool, Dict[str, Any], Optional[str]]:
    """
    Checks if the model output is valid JSON matching the defined schema.
    
    Returns:
        (is_valid_json, is_valid_schema, parsed_json, error_message)
    """
    try:
        # Find the first { and last } to clean up potential extra text
        start = response.find('{')
        end = response.rfind('}')
        
        if start == -1 or end == -1:
            return False, False, {"error": "No JSON object found", "raw": response}, "No JSON brackets found"
        
        json_str = response[start:end+1]
        parsed = json.loads(json_str)
        
        # Validate schema
        jsonschema.validate(instance=parsed, schema=OUTPUT_SCHEMA)
        
        # Check if mood is reasonable (soft check - log warning but don't fail)
        mood = parsed.get("mood", "").lower()
        if mood not in EXPECTED_MOODS:
            log.warning(f"Unexpected mood value: '{mood}' (expected one of {EXPECTED_MOODS})")
        
        return True, True, parsed, None
        
    except json.JSONDecodeError as e:
        return False, False, {"error": f"Invalid JSON: {str(e)}", "raw": response}, str(e)
        
    except jsonschema.exceptions.ValidationError as e:
        return True, False, {"error": f"Schema validation failed: {e.message}", "raw": response}, e.message


def check_constraints(
    parsed_json: Dict[str, Any], 
    max_tokens: int = MAX_UTTERANCE_TOKENS
) -> Dict[str, bool]:
    """
    Checks for token limit and banlist violations.
    
    Returns:
        Dict with constraint check results
    """
    utterance = parsed_json.get("utterance", "")
    
    # 1. Token count (approximate)
    token_count = len(utterance.split())
    is_brief = token_count <= max_tokens
    
    # 2. Check banlist
    utterance_lower = utterance.lower()
    banned_words_found = [word for word in BANLIST if word in utterance_lower]
    has_no_banned_words = len(banned_words_found) == 0
    
    # 3. Check for empty utterance
    is_non_empty = len(utterance.strip()) > 0
    
    return {
        "is_brief": is_brief,
        "token_count": token_count,
        "has_no_banned_words": has_no_banned_words,
        "banned_words": banned_words_found,
        "is_non_empty": is_non_empty
    }


# --- 2. Diversity Metrics ---

def calculate_distinct_n(texts: List[str], n: int) -> float:
    """
    Calculates Distinct-n score (Li et al., 2016).
    Measures the ratio of unique n-grams to total n-grams.
    
    Improved version with proper tokenization.
    """
    if not texts:
        return 0.0
        
    all_ngrams = []
    total_ngrams_count = 0
    
    for text in texts:
        # Proper tokenization - handle punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        if len(tokens) < n:
            continue
            
        current_ngrams = list(ngrams(tokens, n))
        all_ngrams.extend(current_ngrams)
        total_ngrams_count += len(current_ngrams)
        
    if total_ngrams_count == 0:
        return 0.0
        
    unique_ngrams_count = len(set(all_ngrams))
    return unique_ngrams_count / total_ngrams_count


def calculate_entropy(texts: List[str]) -> float:
    """
    Calculates the entropy of the word distribution across all texts.
    Higher entropy = more diverse vocabulary.
    """
    if not texts:
        return 0.0
    
    all_words = []
    for text in texts:
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend(words)
    
    if not all_words:
        return 0.0
    
    word_counts = Counter(all_words)
    total_words = len(all_words)
    
    entropy = 0.0
    for count in word_counts.values():
        probability = count / total_words
        entropy -= probability * np.log2(probability)
    
    return entropy


# --- 3. Hallucination / Grounding Metrics ---

# Pre-compile regex
CONTENT_WORD_RE = re.compile(r'\b[a-zA-Z0-9]{3,}\b')


def get_content_words(text: str) -> set:
    """Extracts a set of unique lowercase content words from a text."""
    return set(CONTENT_WORD_RE.findall(text.lower()))


def calculate_hallucination_metrics(
    utterance: str, 
    grounding_context: set
) -> Tuple[float, float, int, int]:
    """
    Calculates UCR (Unsupported Content Rate) and NEP (Named Entity Precision).
    
    Enhanced version that excludes common conversational words from hallucination detection.
    
    Args:
        utterance: The generated text from the model.
        grounding_context: A set of allowed lowercase words from
                          (persona + player_query + world_facts).
                           
    Returns:
        (UCR, NEP, unsupported_count, total_content_words)
    """
    if not utterance:
        return 0.0, 1.0, 0, 0
        
    generated_words = get_content_words(utterance)
    
    if not generated_words:
        return 0.0, 1.0, 0, 0

    # Combine grounding context with conversational words
    allowed_words = grounding_context | CONVERSATIONAL_WORDS
    
    # Find words NOT in the allowed set
    unsupported_words = generated_words - allowed_words
    
    # UCR: Fraction of content words that are unsupported
    ucr = len(unsupported_words) / len(generated_words)
    
    # NEP: Fraction of content words that are supported
    nep = len(generated_words & allowed_words) / len(generated_words)
    
    return ucr, nep, len(unsupported_words), len(generated_words)


# --- 4. Persona Faithfulness Metrics (NLI & Embeddings) ---

class PersonaMetrics:
    """
    Stateful class for expensive models (NLI, Embeddings).
    Computes persona-related metrics.
    """
    
    def __init__(
        self, 
        nli_model_name: str, 
        embedding_model_name: str, 
        device: str, 
        max_nli_length: int = 128,
        nli_batch_size: int = 8
    ):
        log.info(f"Loading NLI model: {nli_model_name}")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)
        
        device_id = 0 if device == "cuda" else -1
        self.nli_pipeline = pipeline(
            "text-classification",
            model=self.nli_model,
            tokenizer=self.nli_tokenizer,
            device=device_id
        )
        
        self.max_nli_length = max_nli_length
        self.nli_batch_size = nli_batch_size

        log.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)
        self.device = device
        
        log.info("PersonaMetrics models loaded successfully.")

    def calculate_persona_contradiction_rate(
        self,
        utterance: str,
        persona_statements: List[str],
        threshold: float = NLI_CONTRADICTION_THRESHOLD
    ) -> Tuple[bool, float, List[Dict]]:
        """
        Checks if utterance contradicts any persona statement using NLI.
        
        Returns:
            (has_contradiction, max_contradiction_score, all_results)
        """
        if not utterance or not persona_statements:
            return False, 0.0, []

        try:
            # NLI format: (premise, hypothesis)
            # Premise = persona statement (what we know to be true)
            # Hypothesis = utterance (what we're testing)
            pairs = [(stmt, utterance) for stmt in persona_statements]
            
            results = self.nli_pipeline(
                pairs,
                batch_size=self.nli_batch_size,
                truncation=True,
                padding=True,
                max_length=self.max_nli_length
            )

            max_contradiction_score = 0.0
            contradiction_found = False
            detailed_results = []
            
            for persona_stmt, res in zip(persona_statements, results):
                label = res['label'].upper()
                score = res['score']
                
                detailed_results.append({
                    "persona_statement": persona_stmt,
                    "label": label,
                    "score": score
                })
                
                if label == 'CONTRADICTION' and score > threshold:
                    contradiction_found = True
                    max_contradiction_score = max(max_contradiction_score, score)
            
            return contradiction_found, max_contradiction_score, detailed_results

        except Exception as e:
            log.warning(f"NLI pipeline failed: {e}. Utterance: '{utterance[:100]}'")
            return False, 0.0, []

    def calculate_persona_similarity(
        self,
        utterance: str,
        persona_statements: List[str]
    ) -> Tuple[float, float, List[float]]:
        """
        Calculates cosine similarity between utterance and persona statements.
        
        Returns:
            (max_similarity, mean_similarity, all_similarities)
        """
        if not utterance or not persona_statements:
            return 0.0, 0.0, []
            
        try:
            utterance_emb = self.embedding_model.encode(
                utterance, 
                convert_to_tensor=True,
                device=self.device
            )
            
            persona_emb = self.embedding_model.encode(
                persona_statements, 
                convert_to_tensor=True, 
                device=self.device
            )
            
            cos_scores = util.cos_sim(utterance_emb, persona_emb)[0].cpu().numpy()
            
            max_sim = float(np.max(cos_scores))
            mean_sim = float(np.mean(cos_scores))
            all_sims = [float(s) for s in cos_scores]
            
            return max_sim, mean_sim, all_sims
            
        except Exception as e:
            log.warning(f"Embedding similarity failed: {e}")
            return 0.0, 0.0, []

    def batch_encode_utterances(
        self,
        utterances: List[str],
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Batch encode multiple utterances for efficiency.
        """
        try:
            embeddings = self.embedding_model.encode(
                utterances,
                convert_to_tensor=True,
                device=self.device,
                batch_size=batch_size,
                show_progress_bar=True
            )
            return embeddings
        except Exception as e:
            log.error(f"Batch encoding failed: {e}")
            return None


# --- 5. Main Metrics Bundle ---

class EvaluationMetrics:
    """Wrapper class to compute all metrics for a single generation."""
    
    def __init__(
        self, 
        persona_metrics_computer: PersonaMetrics, 
        sim_threshold: float,
        store_detailed_results: bool = True
    ):
        self.persona_computer = persona_metrics_computer
        self.sim_threshold = sim_threshold
        self.store_detailed_results = store_detailed_results
        
        try:
            nltk.download('punkt', quiet=True)
        except Exception as e:
            log.warning(f"Could not download nltk 'punkt': {e}")

    def compute_all(
        self, 
        item: Dict[str, Any], 
        response: str
    ) -> Dict[str, Any]:
        """
        Computes all metrics for a single test item and its response.
        
        Args:
            item: The original test.jsonl record
            response: Raw string generated by the model
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {
            "record_id": item.get("id"),
            "source_dataset": item.get("source")
        }
        
        # 1. Schema & Constraint Metrics
        is_valid_json, is_valid_schema, parsed, error_msg = validate_schema(response)
        
        metrics["is_valid_json"] = is_valid_json
        metrics["is_valid_schema"] = is_valid_schema
        
        if error_msg and self.store_detailed_results:
            metrics["parse_error"] = error_msg
        
        utterance = ""
        if is_valid_json and is_valid_schema:
            utterance = parsed.get("utterance", "")
            mood = parsed.get("mood", "")
            
            constraints = check_constraints(parsed)
            metrics.update({
                "is_brief": constraints["is_brief"],
                "token_count": constraints["token_count"],
                "is_clean": constraints["has_no_banned_words"],
                "is_non_empty": constraints["is_non_empty"],
                "mood": mood
            })
            
            if self.store_detailed_results and constraints["banned_words"]:
                metrics["banned_words_found"] = constraints["banned_words"]
        else:
            metrics.update({
                "is_brief": False,
                "token_count": 0,
                "is_clean": False,
                "is_non_empty": False,
                "mood": None
            })
            
        metrics["utterance"] = utterance
        metrics["raw_response"] = response

        # 2. Persona Faithfulness Metrics
        persona = item.get("persona", [])
        
        if utterance and persona:
            # Contradiction check
            has_contradiction, contradiction_score, nli_details = \
                self.persona_computer.calculate_persona_contradiction_rate(utterance, persona)
            
            metrics["persona_contradiction"] = has_contradiction
            metrics["contradiction_score"] = contradiction_score
            
            if self.store_detailed_results:
                metrics["nli_details"] = nli_details
            
            # Similarity check
            max_sim, mean_sim, all_sims = \
                self.persona_computer.calculate_persona_similarity(utterance, persona)
            
            metrics["persona_similarity_max"] = max_sim
            metrics["persona_similarity_mean"] = mean_sim
            metrics["is_in_persona"] = max_sim > self.sim_threshold
            
            if self.store_detailed_results:
                metrics["all_similarities"] = all_sims
        else:
            metrics.update({
                "persona_contradiction": None,
                "contradiction_score": None,
                "persona_similarity_max": None,
                "persona_similarity_mean": None,
                "is_in_persona": None
            })

        # 3. Hallucination / Grounding Metrics
        grounding_text = " ".join(persona)
        grounding_text += " " + " ".join(item.get("world_facts", []))
        
        if item.get("dialog") and len(item["dialog"]) > 0:
            grounding_text += " " + item["dialog"][0].get("text", "")
            
        grounding_context = get_content_words(grounding_text)
        
        if utterance:
            ucr, nep, unsupported_count, total_words = \
                calculate_hallucination_metrics(utterance, grounding_context)
            
            metrics["ucr"] = ucr
            metrics["nep"] = nep
            metrics["unsupported_word_count"] = unsupported_count
            metrics["total_content_words"] = total_words
        else:
            metrics.update({
                "ucr": None,
                "nep": None,
                "unsupported_word_count": None,
                "total_content_words": None
            })
        
        return metrics


# --- 6. Aggregation & Statistical Analysis ---

def safe_confidence_interval(
    values: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Safely calculate confidence interval, handling edge cases.
    
    Returns:
        (lower_bound, upper_bound) or (mean, mean) if calculation fails
    """
    if len(values) <= 1:
        mean_val = float(np.mean(values))
        return mean_val, mean_val
    
    mean_val = np.mean(values)
    sem_val = stats.sem(values)
    
    # Check for zero variance or invalid values
    if sem_val < 1e-10 or np.isnan(sem_val) or np.isinf(sem_val):
        # Zero variance: return mean as both bounds
        return float(mean_val), float(mean_val)
    
    try:
        ci = stats.t.interval(
            confidence_level,
            len(values) - 1,
            loc=mean_val,
            scale=sem_val
        )
        
        # Validate CI results
        if np.isnan(ci[0]) or np.isnan(ci[1]) or np.isinf(ci[0]) or np.isinf(ci[1]):
            # Fallback: use mean ± 2*std (approximately 95% CI for normal dist)
            std_val = np.std(values)
            return float(mean_val - 1.96 * std_val), float(mean_val + 1.96 * std_val)
        
        return float(ci[0]), float(ci[1])
        
    except Exception as e:
        log.warning(f"CI calculation failed: {e}. Using mean ± std as fallback.")
        std_val = np.std(values)
        return float(mean_val - std_val), float(mean_val + std_val)


def aggregate_metrics(
    results: List[Dict[str, Any]], 
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Aggregates metrics from individual results with confidence intervals.
    """
    if not results:
        return {"error": "No results to aggregate."}
        
    num_samples = len(results)
    agg = {"total_samples": num_samples}
    
    # Metrics to average
    numeric_keys = [
        "is_valid_json", "is_valid_schema", "is_brief", "is_clean", 
        "is_non_empty", "persona_contradiction", "contradiction_score",
        "persona_similarity_max", "persona_similarity_mean", "is_in_persona",
        "ucr", "nep", "token_count", "unsupported_word_count", "total_content_words"
    ]
    
    for key in numeric_keys:
        values = [r[key] for r in results if r.get(key) is not None]
        
        if values:
            # Convert to float array (handles booleans properly)
            values_arr = np.array(values, dtype=np.float64)
            
            mean_val = np.mean(values_arr)
            std_val = np.std(values_arr)
            
            agg[f"avg_{key}"] = float(mean_val)
            agg[f"std_{key}"] = float(std_val)
            
            # Confidence interval (only if we have variance)
            if len(values_arr) > 1:
                sem_val = stats.sem(values_arr)
                
                # Only calculate CI if we have non-zero variance
                if sem_val > 1e-10 and not np.isnan(sem_val) and not np.isinf(sem_val):
                    try:
                        ci = stats.t.interval(
                            confidence_level,
                            len(values_arr) - 1,
                            loc=mean_val,
                            scale=sem_val
                        )
                        agg[f"ci_{key}_lower"] = float(ci[0])
                        agg[f"ci_{key}_upper"] = float(ci[1])
                    except (ValueError, RuntimeWarning):
                        # If CI calculation fails, use mean ± std as fallback
                        agg[f"ci_{key}_lower"] = float(mean_val - std_val)
                        agg[f"ci_{key}_upper"] = float(mean_val + std_val)
                else:
                    # Zero variance: CI is just the mean
                    agg[f"ci_{key}_lower"] = float(mean_val)
                    agg[f"ci_{key}_upper"] = float(mean_val)
        else:
            agg[f"avg_{key}"] = None
    
    # Diversity Metrics
    valid_utterances = [
        r["utterance"] for r in results 
        if r.get("utterance") and r.get("is_valid_schema")
    ]
    
    agg["valid_utterance_count"] = len(valid_utterances)
    
    if valid_utterances:
        agg["diversity_distinct_1"] = calculate_distinct_n(valid_utterances, n=1)
        agg["diversity_distinct_2"] = calculate_distinct_n(valid_utterances, n=2)
        agg["diversity_distinct_3"] = calculate_distinct_n(valid_utterances, n=3)
        agg["diversity_entropy"] = calculate_entropy(valid_utterances)
    else:
        agg["diversity_distinct_1"] = 0.0
        agg["diversity_distinct_2"] = 0.0
        agg["diversity_distinct_3"] = 0.0
        agg["diversity_entropy"] = 0.0
    
    # Mood distribution
    moods = [r.get("mood") for r in results if r.get("mood")]
    if moods:
        mood_counts = Counter(moods)
        agg["mood_distribution"] = dict(mood_counts)
        agg["unique_moods"] = len(mood_counts)
    
    log.info(f"Aggregated metrics for {num_samples} samples.")
    return agg


def compare_models(
    baseline_results: List[Dict[str, Any]], 
    tuned_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Statistical comparison between baseline and fine-tuned models.
    Performs paired t-tests and calculates effect sizes.
    """
    if len(baseline_results) != len(tuned_results):
        log.error("Result lists must have the same length for paired comparison")
        return {"error": "Mismatched result lengths"}
    
    comparison = {}
    
    keys_to_compare = [
        "is_valid_schema", "persona_contradiction", "persona_similarity_max",
        "ucr", "nep", "token_count", "contradiction_score"
    ]
    
    for key in keys_to_compare:
        baseline_vals = [r[key] for r in baseline_results if r.get(key) is not None]
        tuned_vals = [r[key] for r in tuned_results if r.get(key) is not None]
        
        if len(baseline_vals) != len(tuned_vals) or len(baseline_vals) == 0:
            continue
        
        # Convert to float arrays (handles boolean values properly)
        baseline_arr = np.array(baseline_vals, dtype=np.float64)
        tuned_arr = np.array(tuned_vals, dtype=np.float64)
        
        # Check for valid data
        if np.any(np.isnan(baseline_arr)) or np.any(np.isnan(tuned_arr)):
            log.warning(f"Skipping {key}: contains NaN values")
            continue
        
        # Paired t-test
        try:
            t_stat, p_value = stats.ttest_rel(baseline_arr, tuned_arr)
            
            # Handle edge cases in t-test results
            if np.isnan(t_stat) or np.isnan(p_value):
                log.warning(f"Skipping {key}: t-test returned NaN")
                continue
                
        except Exception as e:
            log.warning(f"t-test failed for {key}: {e}")
            continue
        
        # Effect size (Cohen's d for paired samples)
        diff = tuned_arr - baseline_arr
        diff_std = np.std(diff)
        
        # Only calculate Cohen's d if we have variance
        if diff_std > 1e-10:
            cohens_d = np.mean(diff) / diff_std
        else:
            # No variance in differences = no effect
            cohens_d = 0.0
        
        # Improvement metrics
        baseline_mean = float(np.mean(baseline_arr))
        baseline_std = float(np.std(baseline_arr))
        tuned_mean = float(np.mean(tuned_arr))
        tuned_std = float(np.std(tuned_arr))
        improvement = tuned_mean - baseline_mean
        
        if baseline_mean != 0:
            improvement_pct = (improvement / abs(baseline_mean)) * 100
        else:
            improvement_pct = 0.0
        
        comparison[key] = {
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "tuned_mean": tuned_mean,
            "tuned_std": tuned_std,
            "improvement": float(improvement),
            "improvement_pct": float(improvement_pct),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "is_significant": bool(p_value < 0.05),
            "cohens_d": float(cohens_d),
            "effect_size": (
                "large" if abs(cohens_d) > 0.8 else 
                "medium" if abs(cohens_d) > 0.5 else 
                "small"
            )
        }
    
    log.info("Model comparison complete.")
    return comparison


def get_best_and_worst_examples(
    results: List[Dict[str, Any]],
    metric_key: str = "persona_similarity_max",
    n: int = 3
) -> Dict[str, List[Dict]]:
    """
    Extracts best and worst examples based on a metric.
    """
    valid_results = [r for r in results if r.get(metric_key) is not None]
    
    if not valid_results:
        return {"best": [], "worst": []}
    
    sorted_results = sorted(valid_results, key=lambda x: x[metric_key], reverse=True)
    
    best = sorted_results[:n]
    worst = sorted_results[-n:]
    
    return {
        "best": best,
        "worst": worst
    }