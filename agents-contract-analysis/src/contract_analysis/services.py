import datetime
import hashlib
import json
import logging
import os
import uuid
import numpy as np
from typing import List, Dict
import re

import openai
from crewai.llm import LLM
from markitdown import MarkItDown

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sklearn.metrics.pairwise import cosine_similarity

from src.contract_analysis.models import ContractClassification

# Setup logging
logger = logging.getLogger(__name__)


class ContractsService:
    """
    Service for processing, classifying, and storing contract documents in a vector database.
    Handles document conversion, chunking, classification, and embedding.
    """

    def __init__(self):
        # Load all environment variables at initialization
        self.contracts_dir = os.getenv("CONTRACTS_DIR", "knowledge/contracts")
        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        self.qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME", "")
        self.qdrant_url = os.getenv("QDRANT_URL", "")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.vector_size = int(os.getenv("VECTOR_SIZE", "1536"))

        # Validate critical configuration
        self._validate_configuration()

        # Initialize clients
        self.vector_client = QdrantClient(
            url=self.qdrant_url, api_key=self.qdrant_api_key
        )
        self.openai_client = openai.Client(api_key=self.openai_key)
        # self.chunker = HybridChunker()
        self.doc_converter = MarkItDown(enable_builtins=True)

    def _validate_configuration(self) -> None:
        """Validate that all required configuration is present."""
        if not self.contracts_dir or not os.path.isdir(self.contracts_dir):
            raise ValueError(f"Invalid contracts directory: {self.contracts_dir}")
        if not self.openai_key:
            raise ValueError("OpenAI API key is required")
        if (
            not self.qdrant_url
            or not self.qdrant_api_key
            or not self.qdrant_collection_name
        ):
            raise ValueError("Qdrant configuration is incomplete")

    @property
    def llm(self) -> LLM:
        """Get the LLM instance for contract classification."""
        return LLM(
            model="o3-mini",
            api_key=self.openai_key,
            response_format=ContractClassification,
        )

    def load_and_classify_contracts(self) -> None:
        """Main method to create collection and process all contracts."""
        try:
            self._create_collection()
            self._populate_collection()
        except Exception as e:
            logger.error(f"Error processing contracts: {str(e)}")
            raise

    def _create_collection(self) -> None:
        """Create the vector collection if it doesn't exist already."""
        try:
            if self.vector_client.collection_exists(self.qdrant_collection_name):
                logger.info(
                    f"Collection {self.qdrant_collection_name} already exists... skipping creation"
                )
            else:
                self.vector_client.create_collection(
                    collection_name=self.qdrant_collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size, distance=Distance.COSINE
                    ),
                )
                logger.info(
                    f"Collection {self.qdrant_collection_name} created successfully"
                )
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise

    def _populate_collection(self) -> None:
        """Process and embed all contracts found in the contracts directory."""
        points = []
        files_processed = 0
        files_skipped = 0

        # Get list of files to process
        try:
            contract_files = os.listdir(self.contracts_dir)
        except Exception as e:
            logger.error(f"Error reading contracts directory: {str(e)}")
            raise

        for filename in contract_files:
            contract_full_path = os.path.join(self.contracts_dir, filename)
            try:
                # Calculate file hash
                with open(contract_full_path, "rb") as file:
                    md5 = hashlib.md5(file.read()).hexdigest()

                # Skip if already processed
                if self._contract_already_embedded(md5):
                    logger.info(f"{filename} already exists in collection... skipping")
                    files_skipped += 1
                    continue

                logger.info(f"Processing {filename}")

                # Process the new contract
                points.extend(self._process_contract(contract_full_path, filename, md5))
                files_processed += 1

            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                # Continue with next file instead of failing completely
                continue

        # Upsert all points in a single batch
        if points:
            try:
                self.vector_client.upsert(
                    collection_name=self.qdrant_collection_name, points=points
                )
                logger.info(
                    f"Upserted {len(points)} points from {files_processed} files"
                )
            except Exception as e:
                logger.error(f"Error upserting points to Qdrant: {str(e)}")
                raise
        else:
            logger.info("No new documents to process")

        logger.info(
            f"Processing complete: {files_processed} files processed, {files_skipped} files skipped"
        )

    def _combine_sentences(
        self, sentences: List[Dict], buffer_size: int = 3
    ) -> List[Dict]:
        """Combines sentences with surrounding context buffer."""
        for i in range(len(sentences)):
            combined_sentence = ""

            # Add preceding sentences
            for j in range(i - buffer_size, i):
                if j >= 0:
                    combined_sentence += sentences[j]["sentence"] + " "

            # Add current sentence
            combined_sentence += sentences[i]["sentence"]

            # Add following sentences
            for j in range(i + 1, i + 1 + buffer_size):
                if j < len(sentences):
                    combined_sentence += " " + sentences[j]["sentence"]

            sentences[i]["combined_sentence"] = combined_sentence.strip()

        return sentences

    def _calculate_cosine_distances(
        self, sentences: List[Dict]
    ) -> tuple[List[float], List[Dict]]:
        """Calculate cosine distances between consecutive sentences."""
        distances = []
        for i in range(len(sentences) - 1):
            embedding_current = sentences[i]["combined_sentence_embedding"]
            embedding_next = sentences[i + 1]["combined_sentence_embedding"]

            similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
            distance = 1 - similarity

            distances.append(distance)
            sentences[i]["distance_to_next"] = distance

        return distances, sentences

    def _semantic_chunker(self, text: str) -> List[str]:
        """Chunk text based on semantic similarity."""
        # Split into sentences
        single_sentences_list = re.split(r"(?<=[.?!])\s+", text)
        sentences = [
            {"sentence": x, "index": i} for i, x in enumerate(single_sentences_list)
        ]

        # Combine sentences with context
        sentences = self._combine_sentences(sentences)

        # Get embeddings
        embeddings = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=[x["combined_sentence"] for x in sentences],
        )

        for i, sentence in enumerate(sentences):
            sentence["combined_sentence_embedding"] = embeddings.data[i].embedding

        # Calculate distances and find breakpoints
        distances, sentences = self._calculate_cosine_distances(sentences)
        breakpoint_threshold = np.percentile(distances, 95)
        breakpoint_indices = [
            i for i, x in enumerate(distances) if x > breakpoint_threshold
        ]

        # Create chunks
        chunks = []
        start_index = 0
        for index in breakpoint_indices:
            group = sentences[start_index : index + 1]
            chunks.append(" ".join([d["sentence"] for d in group]))
            start_index = index + 1

        # Add final chunk
        if start_index < len(sentences):
            chunks.append(" ".join([d["sentence"] for d in sentences[start_index:]]))

        return chunks

    def _process_contract(
        self, file_path: str, filename: str, md5: str
    ) -> List[PointStruct]:
        """Process a single contract file and return points for embedding."""
        points = []

        # Convert document
        result = self.doc_converter.convert(file_path)

        # Classify contract
        classification = self._classify_contract(result.markdown)

        # Create chunks using semantic chunking
        chunks = self._semantic_chunker(result.markdown)

        # Process each chunk
        for i, chunk in enumerate(chunks):
            try:
                # Create embedding
                embedding_result = self.openai_client.embeddings.create(
                    input=chunk, model=self.embedding_model
                )
                vector = embedding_result.data[0].embedding

                # Create metadata
                metadata = {
                    "md5": md5,
                    "filename": filename,
                    "contract_classification": classification,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_length": len(chunk),
                    "processed_date": datetime.datetime.now().isoformat(),
                    "title": result.title,
                }

                # Create point
                point_id = str(uuid.uuid4())
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "text": chunk,
                            "metadata": metadata,
                        },
                    )
                )
            except Exception as e:
                logger.error(f"Error processing chunk {i} of {filename}: {str(e)}")
                continue

        return points

    def _contract_already_embedded(self, md5: str) -> bool:
        existing_points = self.vector_client.scroll(
            collection_name=self.qdrant_collection_name,
            scroll_filter={"must": [{"key": "metadata.md5", "match": {"text": md5}}]},  # type: ignore
        )[0]
        return len(existing_points) > 0

    def _classify_contract(self, document: str) -> dict:
        prompt = f"""
        You are a contract classifier expert.
        You are given a contract.
        You need to classify the contract into one of the following categories:
        - Affiliate Agreement
        - Co-Branding
        - Development
        - Distributor
        - Endorsement
        - Franchise
        - Hosting
        - IP
        - Joint Venture
        - License Agreement
        - Maintenance
        - Manufacturing
        - Marketing
        - Non-Compete Non-Solicit
        - Outsourcing
        - Promotion
        - Reseller
        - Service
        - Sponsorship
        - Strategic Alliance
        - Supply
        - Transportation

        It's imperative that:
        1. You only return the category name
        2. One category needs to be selected out of this list
        3. The reasoning provided has to be thorough yet concise

        This is the contract's content:
        {document}
        """
        result = self.llm.call(prompt)
        result_dict = json.loads(result)
        print(f">>>> Document classified as {result_dict['category']}")
        return result_dict


if __name__ == "__main__":
    contracts_service = ContractsService()
    contracts_service.load_and_classify_contracts()
