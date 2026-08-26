from typing import Tuple, List, Dict, Any

from retriever import LegalRetriever
from augmentor import LegalAugmentor
from generator import LegalGenerator


class LegalRAGPipeline:

    def __init__(
        self,
        retriever: LegalRetriever,
        augmentor: LegalAugmentor,
        generator: LegalGenerator
    ):

        self.retriever = retriever
        self.augmentor = augmentor
        self.generator = generator

    def generate_answer(
        self,
        query: str,
        top_k: int = 3
    ) -> Tuple[str, List[Dict[str, Any]]]:

        # 1. Retrieval
        retrieved_docs = self.retriever.retrieve(
            query,
            top_k=top_k
        )

        # 2. Augmentation
        messages = self.augmentor.augment(
            query,
            retrieved_docs
        )

        # 3. Generation
        answer = self.generator.generate(
            messages
        )

        return answer, retrieved_docs