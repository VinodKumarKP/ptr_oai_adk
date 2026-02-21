"""Query analyzer for breaking down and optimizing user queries."""

import logging
from typing import List, Any, Optional

class PromptAnalyzer:
    """Analyzes and transforms user queries for better retrieval."""

    def __init__(self, llm: Any, logger: Optional[logging.Logger] = None):
        """Initialize the query analyzer.

        Args:
            llm: The LLM instance to use for analysis
            logger: Optional logger instance
        """
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)

    def analyze(self, query: str) -> List[str]:
        """Break down user input into atomic questions.

        Args:
            query: The user's input query

        Returns:
            List of atomic questions
        """
        if not self.llm:
            self.logger.warning("No LLM provided for query analysis. Returning original query.")
            return [query]

        try:
            prompt = f"""You are an expert at optimizing queries for vector database retrieval.

            Your goal is to process the user's query into a list of standalone search queries.

            Rules:
            1. **Single Intent**: If the query asks about one specific thing (e.g., "what is architect fees?"), return it as a single query. Do NOT split it into sub-concepts like "architect" and "fees".
            2. **Compound Queries**: If the query asks multiple distinct questions (e.g., "what is X and how does Y work?"), split it into separate, self-contained queries.
            3. **Optimization**: Ensure each output query is optimized for vector similarity search (semantically complete, removing unnecessary conversational filler).
            4. **Strictness**: Do NOT generate synonyms, related questions, or sub-topics that were not explicitly asked.

            Return ONLY the comma-separated list of queries.

            Query: {query}"""

            # Handle different LLM interfaces (LangChain vs Strands/LiteLLM)
            if hasattr(self.llm, 'invoke'):
                response = self.llm.invoke(prompt)
                content = response.content if hasattr(response, 'content') else str(response)
            else:
                # Fallback for other LLM types
                from litellm import completion
                # Check if llm has config attribute (Strands model wrapper)
                if hasattr(self.llm, 'config'):
                    model_id = self.llm.config['model_id']
                elif hasattr(self.llm, 'model'):
                    model_id = self.llm.model
                elif hasattr(self.llm, 'model_id'):
                    model_id = self.llm.model_id
                else:
                    model_id = self.llm
                
                response = completion(model=model_id, messages=[{ "content": prompt, "role": "user"}])
                content = response['choices'][0]['message']['content']


            questions = [q.strip() for q in content.split(',') if q.strip()]
            
            self.logger.debug(f"Analyzed query '{query}' into: {questions}")
            return questions

        except Exception as e:
            self.logger.error(f"Error analyzing query: {e}")
            return [query]
