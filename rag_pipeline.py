"""
Complete RAG Pipeline for Arabic Legal Assistant.
Orchestrates Retrieval, Augmentation, and Generation.
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from augmentor import LegalAugmentor
from generator import LegalGenerator
from retriever import LegalRetriever


class LegalRAGPipeline:

    def __init__(
        self,
        retriever: LegalRetriever,
        augmentor: Optional[LegalAugmentor] = None,
        generator: Optional[LegalGenerator] = None,
    ):
        self.retriever = retriever
        self.augmentor = augmentor or LegalAugmentor()
        self.generator = generator or LegalGenerator()

    def generate_answer(
        self,
        query: str,
        top_k: int = 4,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Retrieve relevant clauses and generate complete answer."""
        # 1. Retrieval
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)

        # 2. Augmentation
        messages = self.augmentor.augment(query, retrieved_docs)

        # 3. Generation
        answer = self.generator.generate(messages)

        return answer, retrieved_docs

    def generate_answer_stream(
        self,
        query: str,
        top_k: int = 4,
    ) -> Tuple[Iterator[str], List[Dict[str, Any]]]:
        """Retrieve relevant clauses and stream generated answer tokens."""
        # 1. Retrieval
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)

        # 2. Augmentation
        messages = self.augmentor.augment(query, retrieved_docs)

        # 3. Stream Generation
        stream_iter = self.generator.generate_stream(messages)

        return stream_iter, retrieved_docs

    def generate_contract_analysis(
        self,
        contract_title: str = "العقد المرفوع",
        top_k: int = 6,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Perform a full contract audit and risk analysis."""
        # Retrieve broad context for all aspects of contract
        retrieved_docs = self.retriever.retrieve("أطراف العقد والالتزامات والمدة والفسخ والشروط الجزائية", top_k=top_k)
        messages = self.augmentor.build_contract_summary_prompt(retrieved_docs, contract_title=contract_title)
        report = self.generator.generate(messages)
        return report, retrieved_docs