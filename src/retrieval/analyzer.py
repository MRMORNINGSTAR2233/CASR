"""
Query Analyzer

LLM-based query analysis for intent detection, entity extraction,
and retrieval optimization.
"""

import json
import time
from typing import Any, Optional

from anthropic import Anthropic
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from src.models.documents import SecurityClassification
from src.models.queries import (
    AnalyzedQuery,
    DataTypeRequirement,
    QueryContext,
    QueryIntent,
    SearchQuery,
)


class QueryAnalyzer:
    """
    LLM-based query analyzer.
    
    Analyzes user queries to extract:
    - Intent (factual, analytical, procedural, etc.)
    - Entities and keywords
    - Domain relevance
    - Data type requirements
    - Ambiguity detection
    - Security requirements
    """
    
    ANALYSIS_PROMPT = """You are a query analyzer for an enterprise search system. Analyze the following query and user context to optimize retrieval.

User Context:
{user_context}

Query: {query}

Session Context: {session_context}

Analyze this query and respond with a JSON object containing:

{{
    "intent": "<factual|analytical|procedural|definitional|exploratory|troubleshooting|summarization|unknown>",
    "intent_confidence": <0.0-1.0>,
    "reformulated_query": "<improved version of the query for better retrieval>",
    "expanded_queries": ["<alternative phrasings or related queries>"],
    "entities": [{{"text": "<entity>", "type": "<person|organization|product|concept|location|date|other>"}}],
    "keywords": ["<important keywords>"],
    "detected_domains": ["<relevant business domains>"],
    "primary_domain": "<most likely domain or null>",
    "data_type_required": "<structured|unstructured|semi-structured|mixed|any>",
    "is_ambiguous": <true|false>,
    "ambiguity_type": "<lexical|referential|contextual|null>",
    "disambiguation_options": ["<possible interpretations if ambiguous>"],
    "minimum_clearance_needed": "<public|internal|confidential|secret|top_secret|null>",
    "recommended_top_k": <10-50>,
    "use_hybrid_search": <true|false>,
    "metadata_filters": {{"<filter_key>": "<filter_value>"}}
}}

Be thorough but concise. Focus on optimizing retrieval effectiveness."""

    DISAMBIGUATION_PROMPT = """The following query is ambiguous. Help disambiguate it.

Query: {query}
Possible interpretations: {options}
User context: {user_context}

Which interpretation is most likely given the context? Respond with just the selected interpretation."""

    def __init__(
        self,
        provider: str = "anthropic",
        model: Optional[str] = None,
    ):
        """
        Initialize the query analyzer.
        
        Args:
            provider: LLM provider ("anthropic", "openai", "gemini", or "groq")
            model: Model name
        """
        self.provider = provider
        settings = get_settings()
        
        if provider == "anthropic":
            self.model = model or "claude-3-haiku-20240307"
            self._client = Anthropic(api_key=settings.anthropic_api_key)
        elif provider == "openai":
            self.model = model or "gpt-4o-mini"
            self._client = OpenAI(api_key=settings.openai_api_key)
        elif provider == "gemini":
            self.model = model or "gemini-2.0-flash"
            import google.generativeai as genai
            genai.configure(api_key=settings.gemini_api_key)
            self._genai = genai
            self._gemini_model = genai.GenerativeModel(self.model)
        elif provider == "groq":
            self.model = model or "llama-3.3-70b-specdec"
            from groq import Groq
            self._client = Groq(api_key=settings.groq_api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini API."""
        response = self._gemini_model.generate_content(
            prompt + "\n\nRespond with valid JSON only.",
            generation_config={
                "max_output_tokens": 1000,
                "temperature": 0.3,
                "response_mime_type": "application/json",
            }
        )
        return response.text
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _call_groq(self, prompt: str) -> str:
        """Call Groq API."""
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt + "\n\nRespond with valid JSON only."}],
        )
        return response.choices[0].message.content
    
    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM."""
        if self.provider == "anthropic":
            return self._call_anthropic(prompt)
        elif self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "gemini":
            return self._call_gemini(prompt)
        elif self.provider == "groq":
            return self._call_groq(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _parse_analysis_response(self, response: str) -> dict[str, Any]:
        """Parse the LLM response into a structured dict."""
        try:
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            return json.loads(response)
        except json.JSONDecodeError:
            # Return minimal valid structure
            return {
                "intent": "unknown",
                "intent_confidence": 0.5,
                "reformulated_query": "",
                "expanded_queries": [],
                "entities": [],
                "keywords": [],
                "detected_domains": [],
                "primary_domain": None,
                "data_type_required": "any",
                "is_ambiguous": False,
                "ambiguity_type": None,
                "disambiguation_options": [],
                "minimum_clearance_needed": None,
                "recommended_top_k": 20,
                "use_hybrid_search": True,
                "metadata_filters": {},
            }
    
    def analyze(self, query: SearchQuery) -> AnalyzedQuery:
        """
        Analyze a search query.
        
        Args:
            query: The search query to analyze
            
        Returns:
            Analyzed query with extracted information
        """
        start_time = time.time()
        
        # Build context strings
        user_context = query.context.to_prompt_context()
        session_context = (
            f"Recent queries: {', '.join(query.context.session_history[-3:])}"
            if query.context.session_history
            else "No session history"
        )
        
        # Build prompt
        prompt = self.ANALYSIS_PROMPT.format(
            user_context=user_context,
            query=query.text,
            session_context=session_context,
        )
        
        # Call LLM
        response = self._call_llm(prompt)
        
        # Parse response
        analysis = self._parse_analysis_response(response)
        
        # Map to AnalyzedQuery
        try:
            intent = QueryIntent(analysis.get("intent", "unknown"))
        except ValueError:
            intent = QueryIntent.UNKNOWN
        
        try:
            data_type = DataTypeRequirement(analysis.get("data_type_required", "any"))
        except ValueError:
            data_type = DataTypeRequirement.ANY
        
        clearance = None
        if analysis.get("minimum_clearance_needed"):
            try:
                clearance = SecurityClassification(analysis["minimum_clearance_needed"])
            except ValueError:
                clearance = None
        
        analysis_time = (time.time() - start_time) * 1000
        
        return AnalyzedQuery(
            original_query=query.text,
            intent=intent,
            intent_confidence=float(analysis.get("intent_confidence", 0.5)),
            reformulated_query=analysis.get("reformulated_query") or query.text,
            expanded_queries=analysis.get("expanded_queries", []),
            entities=analysis.get("entities", []),
            keywords=analysis.get("keywords", []),
            detected_domains=analysis.get("detected_domains", []),
            primary_domain=analysis.get("primary_domain"),
            data_type_required=data_type,
            is_ambiguous=analysis.get("is_ambiguous", False),
            ambiguity_type=analysis.get("ambiguity_type"),
            disambiguation_options=analysis.get("disambiguation_options", []),
            minimum_clearance_needed=clearance,
            recommended_top_k=int(analysis.get("recommended_top_k", 20)),
            use_hybrid_search=analysis.get("use_hybrid_search", True),
            metadata_filters=analysis.get("metadata_filters", {}),
            analysis_time_ms=analysis_time,
        )
    
    def disambiguate(
        self,
        query: str,
        options: list[str],
        context: QueryContext
    ) -> str:
        """
        Disambiguate an ambiguous query.
        
        Args:
            query: The ambiguous query
            options: Possible interpretations
            context: User context
            
        Returns:
            Most likely interpretation
        """
        prompt = self.DISAMBIGUATION_PROMPT.format(
            query=query,
            options=", ".join(options),
            user_context=context.to_prompt_context(),
        )
        
        response = self._call_llm(prompt)
        return response.strip()
    
    def quick_analyze(self, query_text: str) -> AnalyzedQuery:
        """
        Quick analysis without full context (for testing/simple cases).
        
        Args:
            query_text: The query text
            
        Returns:
            Basic analyzed query
        """
        # Simple keyword-based analysis
        query_lower = query_text.lower()
        
        # Detect intent
        if any(w in query_lower for w in ["how to", "how do", "steps to", "process"]):
            intent = QueryIntent.PROCEDURAL
        elif any(w in query_lower for w in ["what is", "define", "meaning of"]):
            intent = QueryIntent.DEFINITIONAL
        elif any(w in query_lower for w in ["compare", "difference", "versus", "vs"]):
            intent = QueryIntent.ANALYTICAL
        elif any(w in query_lower for w in ["why", "error", "fix", "problem", "issue"]):
            intent = QueryIntent.TROUBLESHOOTING
        elif any(w in query_lower for w in ["summarize", "summary", "overview"]):
            intent = QueryIntent.SUMMARIZATION
        elif "?" in query_text:
            intent = QueryIntent.FACTUAL
        else:
            intent = QueryIntent.EXPLORATORY
        
        # Extract simple keywords
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how", "why"}
        words = query_lower.split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return AnalyzedQuery(
            original_query=query_text,
            intent=intent,
            intent_confidence=0.7,
            reformulated_query=query_text,
            expanded_queries=[],
            entities=[],
            keywords=keywords[:10],
            detected_domains=[],
            primary_domain=None,
            data_type_required=DataTypeRequirement.ANY,
            is_ambiguous=False,
            ambiguity_type=None,
            disambiguation_options=[],
            minimum_clearance_needed=None,
            recommended_top_k=20,
            use_hybrid_search=True,
            metadata_filters={},
            analysis_time_ms=0.0,
        )
