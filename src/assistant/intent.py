from dataclasses import dataclass
import traceback

from transformers import pipeline

from app_logger import AppLogger
from utils import CROSS_ENCODER_NLI_DISTILROBERTA_BASE

GENERIC_INTENTS = [
    "unknown intent",
    "general request",
    "explain request",
    "summarize request",
    "translate request",
]


@dataclass
class DetectionResult:
    label: str
    score: float


class Intent:
    def __init__(self, labels=None, model_name=CROSS_ENCODER_NLI_DISTILROBERTA_BASE):
        """
        Initializes the intent detector with the specified model.
        """
        self.classifier = pipeline("zero-shot-classification", model=model_name)
        self.labels = labels if labels else []
        self.logger = AppLogger()

    def calculate_intent_scores(self, sentence, candidate_labels=None):
        """
        Detects the intent of a given sentence based on candidate labels.
        Args:
            sentence (str): The input sentence for intent detection.
            candidate_labels (list of str): A list of possible intent labels.
        Returns:
            dict: A dictionary containing labels and their associated scores.
        """
        if not candidate_labels:
            if not self.labels:
                raise ValueError("Candidate labels must be provided or set during initialization.")
            candidate_labels = self.labels
        try:
            # Perform zero-shot classification and format the results
            results = self.classifier(sentence, candidate_labels)
            return self._format_intent_results(results)
        except Exception as e:
            self.logger.error(f"Error: {e}, {traceback.format_exc()}")
            return {}

    @staticmethod
    def _format_intent_results(results):
        """
        Formats the results of zero-shot classification.
        Args:
            results (dict): Raw results from the classifier.
        Returns:
            dict: A dictionary of intent labels and their scores (rounded and percentual).
        """
        return {label: round(score * 100, 2) for label, score in zip(results["labels"], results["scores"])}

    def detect(self, sentence, candidate_labels=None):
        """
        Retrieves the top intent label and its score.
        Args:
            sentence (str): The input sentence for intent detection.
            candidate_labels (list of str): A list of possible intent labels.
        Returns:
            DetectionResult: An object containing the top intent label and its score.
        """
        # Get intent results by calling calculate_intent_scores
        confidence_results = self.calculate_intent_scores(sentence, candidate_labels)
        if not confidence_results:
            return DetectionResult(label=None, score=0)
        # Get the top intent (highest score)
        top_label = max(confidence_results, key=confidence_results.get)
        return DetectionResult(label=top_label, score=confidence_results[top_label])


# Example usage
if __name__ == "__main__":
    intent = Intent()
    sent = "What is the current date and time?"
    labels = [
        "request for time",
        "request for date and time",
        "request for date",
        "general request",
        "request for information",
        "weather",
        "politics",
        "sport",
        "general",
    ]
    # Get intent classification results
    intent_scores = intent.calculate_intent_scores(sent, labels)
    # Print all results
    print(f"The phrase: `{sent}` label classification:")
    for label, score in intent_scores.items():
        print(f"{label}: {score}%")
    # Get the top intent
    result = intent.detect(sent, labels)
    print(f"\nTop intent: `{result.label}` with confidence: {result.score}%")
