import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from utils import ALL_MINI_LM_L6_V2


class Similarity:
    def __init__(self, model_name=ALL_MINI_LM_L6_V2):
        """
        Initialize the SentenceSimilarity class by loading the model and tokenizer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        """
        Mean Pooling to aggregate token embeddings into a single sentence embedding.
        """
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @staticmethod
    def cosine_similarity(embeddings):
        """
        Compute the cosine similarity matrix for the given embeddings.
        """
        return torch.mm(embeddings, embeddings.T)

    def embed_sentences(self, sentences):
        """
        Generate embeddings for the input sentences.
        """
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return F.normalize(sentence_embeddings, p=2, dim=1)

    def get_similarity_matrix(self, sentences: list):
        """
        Compare multiple sentences and return all pairwise similarity scores as a dictionary.
        """
        embeddings = self.embed_sentences(sentences)
        similarity_matrix = self.cosine_similarity(embeddings)
        return similarity_matrix

    def compare_multiple(self, sentences):
        """
        Compare multiple sentences, return all pairwise similarity scores,
        and calculate the overall similarity score.
        """
        similarity_matrix = self.get_similarity_matrix(sentences)
        all_similarity = 0
        num_comparisons = 0
        results = []
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:  # Skip self-comparisons
                    all_similarity += similarity_matrix[i][j].item()
                    num_comparisons += 1  # Count the number of valid comparisons
                    results.append({
                        "first": sentences[i],
                        "second": sentences[j],
                        "similarity": similarity_matrix[i][j].item()
                    })
        overall_similarity = all_similarity / num_comparisons if num_comparisons > 0 else 0
        return {
            "pairwise": results,  # Details of individual pairs
            "overall_similarity": overall_similarity,  # Average similarity
        }

    def get_score(self, sentences, digits=2):
        result = self.compare_multiple(sentences)
        return round(result["overall_similarity"], digits)

if __name__ == "__main__":

    # Instantiate the similarity model
    similarity_model = Similarity()

    # Pairwise comparison (two sentences)
    sentences = [
        "What can you tell me about Kangaroos?",
        "Tell me more about marsupials.",
        "Can you tell me more about them.",
        "Is Tasmania good for lifestyle of marsupials?.",
        "What about rocket science?."
    ]

    result_two = similarity_model.get_score(sentences[:2])
    print(f"Pairwise Result: {result_two}")

    results = similarity_model.compare_multiple(sentences)


    print("\nPairwise Similarities:")
    for pair in results["pairwise"]:
        print(pair)
    print("Overall Similarity:", results["overall_similarity"])

    result = similarity_model.get_score(sentences)
    print(f"Result: {result}")
