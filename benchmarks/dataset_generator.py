"""
Generate realistic multi-domain test datasets for CASR evaluation.

Creates documents across domains (finance, healthcare, legal, tech) with:
- Realistic content and metadata
- Security classifications (PUBLIC to TOP_SECRET)
- Ground truth query-document pairs
- RBAC/ABAC attributes
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import random
import json
from datetime import datetime, timedelta
from pathlib import Path

from src.models.documents import Document, SecurityClassification, DocumentMetadata


# Domain-specific document templates
DOMAIN_TEMPLATES = {
    "finance": {
        "topics": [
            "quarterly earnings", "merger acquisition", "market analysis",
            "risk assessment", "compliance report", "investment strategy",
            "financial projections", "audit findings", "shareholder meeting",
            "regulatory filing", "credit analysis", "portfolio management"
        ],
        "entities": [
            "Goldman Sachs", "JPMorgan Chase", "BlackRock", "Vanguard",
            "Federal Reserve", "SEC", "NYSE", "NASDAQ", "S&P 500"
        ],
        "templates": [
            "Q{quarter} {year} Earnings Report for {company}: Revenue of ${revenue}M, {metric} growth of {percent}%. Key highlights include {detail}.",
            "{company} announces {event} valued at ${value}M. Expected completion in {timeframe}. Impact on {metric}: {impact}.",
            "Market Analysis: {sector} sector shows {trend}. Key drivers: {factors}. Recommendation: {recommendation}.",
            "Risk Assessment Report: {risk_type} risk level {level}. Mitigation strategy: {strategy}. Expected impact: {impact}.",
            "Compliance Report {date}: {regulation} compliance status {status}. Issues identified: {count}. Action items: {actions}."
        ]
    },
    "healthcare": {
        "topics": [
            "clinical trial", "patient outcomes", "medical device", "pharmaceutical",
            "treatment protocol", "diagnosis criteria", "research findings",
            "public health", "epidemiology", "medical imaging", "genomics"
        ],
        "entities": [
            "Pfizer", "Johnson & Johnson", "Mayo Clinic", "CDC", "FDA",
            "NIH", "WHO", "Kaiser Permanente", "Cleveland Clinic"
        ],
        "templates": [
            "Clinical Trial {trial_id}: {drug_name} for {condition}. Phase {phase} results show {efficacy}% efficacy with {side_effects}.",
            "Patient Study: {sample_size} participants with {condition}. Treatment: {treatment}. Outcomes: {outcome}. Statistical significance: p={pvalue}.",
            "{device_name} medical device received {approval} approval for {indication}. Clinical benefits: {benefits}.",
            "Research Findings: {study_topic} shows {finding}. Implications for {population}. Published in {journal}.",
            "Public Health Alert: {disease} outbreak in {location}. {count} confirmed cases. Recommended actions: {actions}."
        ]
    },
    "legal": {
        "topics": [
            "case law", "contract review", "intellectual property", "litigation",
            "regulatory compliance", "corporate governance", "employment law",
            "data privacy", "antitrust", "patent filing", "trademark"
        ],
        "entities": [
            "Supreme Court", "District Court", "SEC", "FTC", "DOJ",
            "USPTO", "Copyright Office", "ACLU", "Legal Aid"
        ],
        "templates": [
            "Case {case_number}: {plaintiff} v. {defendant}. Ruling: {ruling}. Precedent set for {legal_area}. Judge: {judge}.",
            "Contract Review: {contract_type} between {party1} and {party2}. Key terms: {terms}. Risks identified: {risks}.",
            "Patent Application {patent_id}: {invention} filed by {inventor}. Claims: {claims}. Prior art: {prior_art}.",
            "Regulatory Compliance: {regulation} requirements for {industry}. Compliance deadline: {deadline}. Penalties for non-compliance: {penalties}.",
            "Litigation Update: {case_name} proceeding to {stage}. Key arguments: {arguments}. Expected resolution: {timeline}."
        ]
    },
    "technology": {
        "topics": [
            "software architecture", "cloud infrastructure", "cybersecurity",
            "machine learning", "data engineering", "API design",
            "system performance", "scalability", "DevOps", "microservices"
        ],
        "entities": [
            "AWS", "Google Cloud", "Microsoft Azure", "OpenAI", "Meta",
            "NVIDIA", "Intel", "Kubernetes", "Docker", "GitHub"
        ],
        "templates": [
            "Architecture Design: {system_name} using {technology}. Components: {components}. Scalability: {scalability}. Expected throughput: {throughput}.",
            "Security Assessment: {vulnerability} discovered in {system}. Severity: {severity}. Mitigation: {mitigation}. Patch available: {patch_status}.",
            "ML Model Performance: {model_name} achieves {metric} of {score} on {dataset}. Training time: {time}. Inference latency: {latency}ms.",
            "Infrastructure Update: Migration to {cloud_provider} completed. Cost savings: {savings}%. Performance improvement: {improvement}.",
            "API Documentation: {endpoint} endpoint accepts {params}. Returns: {response}. Rate limit: {rate_limit} req/min."
        ]
    }
}

# Security classification distribution
CLASSIFICATION_WEIGHTS = {
    SecurityClassification.PUBLIC: 0.4,
    SecurityClassification.INTERNAL: 0.3,
    SecurityClassification.CONFIDENTIAL: 0.2,
    SecurityClassification.SECRET: 0.08,
    SecurityClassification.TOP_SECRET: 0.02
}

# RBAC roles
ROLES = [
    "public_user", "employee", "manager", "director",
    "executive", "admin", "security_officer"
]


@dataclass
class GroundTruthQuery:
    """Ground truth query with expected results."""
    query: str
    relevant_doc_ids: List[str]
    relevance_scores: Dict[str, float]  # doc_id -> score (0.0 to 1.0)
    domain: str
    required_role: str
    required_classification: SecurityClassification
    expected_access: bool
    attributes: Dict[str, Any]


class DatasetGenerator:
    """Generate realistic test datasets for CASR evaluation."""
    
    def __init__(self, seed: int = 42):
        """Initialize dataset generator with random seed for reproducibility."""
        random.seed(seed)
        self.documents: List[Document] = []
        self.queries: List[GroundTruthQuery] = []
    
    def _generate_content(self, domain: str) -> Tuple[str, str]:
        """Generate realistic document content and title."""
        template_data = DOMAIN_TEMPLATES[domain]
        template = random.choice(template_data["templates"])
        
        # Fill template with realistic values
        content = template
        title_parts = []
        
        # Replace placeholders with realistic values
        replacements = {
            "quarter": random.choice(["Q1", "Q2", "Q3", "Q4"]),
            "year": random.choice(["2023", "2024", "2025"]),
            "company": random.choice(template_data["entities"]),
            "revenue": random.randint(100, 10000),
            "metric": random.choice(["revenue", "profit", "growth", "market share"]),
            "percent": random.randint(5, 50),
            "detail": random.choice(template_data["topics"]),
            "event": random.choice(["merger", "acquisition", "partnership", "expansion"]),
            "value": random.randint(100, 5000),
            "timeframe": random.choice(["Q2 2025", "6 months", "fiscal year 2025"]),
            "impact": random.choice(["positive", "neutral", "significant"]),
            "sector": random.choice(["technology", "healthcare", "finance", "energy"]),
            "trend": random.choice(["upward momentum", "consolidation", "volatility"]),
            "factors": ", ".join(random.sample(template_data["topics"], 2)),
            "recommendation": random.choice(["buy", "hold", "sell", "outperform"]),
            "risk_type": random.choice(["credit", "market", "operational", "regulatory"]),
            "level": random.choice(["low", "medium", "high", "critical"]),
            "strategy": random.choice(template_data["topics"]),
            "regulation": random.choice(["SOX", "GDPR", "HIPAA", "CCPA"]),
            "status": random.choice(["compliant", "partial", "non-compliant"]),
            "count": random.randint(1, 20),
            "actions": random.choice(template_data["topics"]),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "trial_id": f"NCT{random.randint(10000, 99999)}",
            "drug_name": f"Drug-{random.randint(100, 999)}",
            "condition": random.choice(["diabetes", "hypertension", "cancer", "arthritis"]),
            "phase": random.choice(["I", "II", "III", "IV"]),
            "efficacy": random.randint(60, 95),
            "side_effects": random.choice(["mild side effects", "minimal adverse events"]),
            "sample_size": random.randint(100, 10000),
            "treatment": random.choice(template_data["topics"]),
            "outcome": random.choice(["improved symptoms", "reduced progression"]),
            "pvalue": f"0.{random.randint(1, 49):02d}",
            "device_name": f"Device-{random.randint(100, 999)}",
            "approval": random.choice(["FDA", "CE", "regulatory"]),
            "indication": random.choice(template_data["topics"]),
            "benefits": random.choice(template_data["topics"]),
            "study_topic": random.choice(template_data["topics"]),
            "finding": random.choice(["significant correlation", "causal relationship"]),
            "population": random.choice(["elderly patients", "pediatric population"]),
            "journal": random.choice(["Nature", "JAMA", "Lancet", "NEJM"]),
            "disease": random.choice(["influenza", "COVID-19", "measles"]),
            "location": random.choice(["Northeast", "California", "Texas"]),
            "case_number": f"{random.randint(20, 25)}-{random.randint(1000, 9999)}",
            "plaintiff": f"Party A Corp",
            "defendant": f"Party B Inc",
            "ruling": random.choice(["in favor of plaintiff", "dismissed", "settled"]),
            "legal_area": random.choice(template_data["topics"]),
            "judge": f"Judge {random.choice(['Smith', 'Johnson', 'Williams'])}",
            "contract_type": random.choice(["NDA", "Service Agreement", "License"]),
            "party1": random.choice(template_data["entities"]),
            "party2": f"Company {random.randint(1, 100)}",
            "terms": random.choice(template_data["topics"]),
            "risks": random.choice(["ambiguous clauses", "liability exposure"]),
            "patent_id": f"US{random.randint(10000000, 99999999)}",
            "invention": random.choice(template_data["topics"]),
            "inventor": f"Dr. {random.choice(['Lee', 'Chen', 'Kumar'])}",
            "claims": random.randint(5, 30),
            "prior_art": random.choice(["identified", "none found"]),
            "industry": domain,
            "deadline": f"{random.randint(1, 12)}/30/{random.randint(2025, 2026)}",
            "penalties": f"${random.randint(10, 500)}K",
            "case_name": random.choice(template_data["topics"]),
            "stage": random.choice(["discovery", "trial", "appeal"]),
            "arguments": random.choice(template_data["topics"]),
            "timeline": f"{random.randint(3, 18)} months",
            "system_name": f"System-{random.randint(1, 100)}",
            "technology": random.choice(["Kubernetes", "microservices", "serverless"]),
            "components": ", ".join(random.sample(template_data["topics"], 2)),
            "scalability": random.choice(["horizontal", "vertical", "auto-scaling"]),
            "throughput": f"{random.randint(1000, 100000)} req/s",
            "vulnerability": f"CVE-2024-{random.randint(10000, 99999)}",
            "system": random.choice(template_data["entities"]),
            "severity": random.choice(["low", "medium", "high", "critical"]),
            "mitigation": random.choice(template_data["topics"]),
            "patch_status": random.choice(["yes", "pending", "no"]),
            "model_name": random.choice(["GPT-4", "BERT", "ResNet", "YOLO"]),
            "score": f"0.{random.randint(85, 99)}",
            "dataset": random.choice(["ImageNet", "COCO", "SQuAD"]),
            "time": f"{random.randint(1, 48)} hours",
            "latency": random.randint(10, 500),
            "cloud_provider": random.choice(template_data["entities"]),
            "savings": random.randint(10, 60),
            "improvement": f"{random.randint(20, 80)}%",
            "endpoint": f"/api/{random.choice(['users', 'data', 'metrics'])}",
            "params": random.choice(["user_id", "date_range", "filters"]),
            "response": random.choice(["JSON", "XML", "CSV"]),
            "rate_limit": random.randint(100, 10000)
        }
        
        for key, value in replacements.items():
            content = content.replace(f"{{{key}}}", str(value))
            if key in ["company", "event", "drug_name", "case_name", "system_name"]:
                title_parts.append(str(value))
        
        # Generate title from topic
        topic = random.choice(template_data["topics"])
        title = f"{topic.title()}: {' - '.join(title_parts[:2]) if title_parts else content[:50]}"
        
        return title, content
    
    def _generate_metadata(self, domain: str) -> DocumentMetadata:
        """Generate realistic document metadata."""
        template_data = DOMAIN_TEMPLATES[domain]
        
        return DocumentMetadata(
            source=f"{domain}_system",
            created_at=datetime.now() - timedelta(days=random.randint(1, 365)),
            updated_at=datetime.now() - timedelta(days=random.randint(0, 30)),
            author=f"{random.choice(['John', 'Jane', 'Alex', 'Sam'])} {random.choice(['Smith', 'Doe', 'Johnson'])}",
            department=domain.title(),
            tags=random.sample(template_data["topics"], k=random.randint(2, 4)),
            language="en",
            file_type="text",
            custom_fields={
                "domain": domain,
                "entity": random.choice(template_data["entities"]),
                "reviewed": random.choice([True, False]),
                "version": f"{random.randint(1, 5)}.{random.randint(0, 9)}"
            }
        )
    
    def generate_documents(
        self,
        num_docs_per_domain: int = 100,
        domains: List[str] = None
    ) -> List[Document]:
        """
        Generate realistic multi-domain documents.
        
        Args:
            num_docs_per_domain: Number of documents to generate per domain
            domains: List of domains to generate (default: all)
            
        Returns:
            List of generated documents
        """
        if domains is None:
            domains = list(DOMAIN_TEMPLATES.keys())
        
        documents = []
        doc_id_counter = 1
        
        for domain in domains:
            for _ in range(num_docs_per_domain):
                title, content = self._generate_content(domain)
                metadata = self._generate_metadata(domain)
                
                # Assign security classification
                classification = random.choices(
                    list(CLASSIFICATION_WEIGHTS.keys()),
                    weights=list(CLASSIFICATION_WEIGHTS.values())
                )[0]
                
                # ABAC attributes based on classification and domain
                attributes = {
                    "domain": domain,
                    "sensitivity": classification.value,
                    "region": random.choice(["US", "EU", "APAC", "GLOBAL"]),
                    "project": f"Project-{random.randint(1, 50)}",
                    "cost_center": f"CC-{random.randint(1000, 9999)}"
                }
                
                # Higher classification = more restricted access
                if classification in [SecurityClassification.SECRET, SecurityClassification.TOP_SECRET]:
                    attributes["clearance_required"] = True
                    attributes["need_to_know"] = random.choice([True, False])
                
                doc = Document(
                    id=f"doc_{domain}_{doc_id_counter:04d}",
                    content=content,
                    metadata=metadata,
                    security_classification=classification,
                    allowed_roles=self._get_allowed_roles(classification),
                    attributes=attributes,
                    embeddings=None  # Will be generated during indexing
                )
                
                documents.append(doc)
                doc_id_counter += 1
        
        self.documents = documents
        return documents
    
    def _get_allowed_roles(self, classification: SecurityClassification) -> List[str]:
        """Get allowed roles for a security classification."""
        role_hierarchy = {
            SecurityClassification.PUBLIC: ROLES,
            SecurityClassification.INTERNAL: ROLES[1:],  # Exclude public_user
            SecurityClassification.CONFIDENTIAL: ROLES[2:],  # Manager+
            SecurityClassification.SECRET: ROLES[4:],  # Executive+
            SecurityClassification.TOP_SECRET: ROLES[5:]  # Admin+
        }
        return role_hierarchy.get(classification, [])
    
    def generate_queries(
        self,
        num_queries_per_domain: int = 50,
        domains: List[str] = None
    ) -> List[GroundTruthQuery]:
        """
        Generate ground truth queries with expected results.
        
        Args:
            num_queries_per_domain: Number of queries per domain
            domains: List of domains (default: all)
            
        Returns:
            List of ground truth queries
        """
        if not self.documents:
            raise ValueError("Must generate documents first")
        
        if domains is None:
            domains = list(DOMAIN_TEMPLATES.keys())
        
        queries = []
        
        for domain in domains:
            domain_docs = [d for d in self.documents if d.attributes.get("domain") == domain]
            template_data = DOMAIN_TEMPLATES[domain]
            
            for _ in range(num_queries_per_domain):
                # Select 1-5 relevant documents
                num_relevant = random.randint(1, 5)
                relevant_docs = random.sample(domain_docs, min(num_relevant, len(domain_docs)))
                
                # Generate query based on document topics
                query_terms = []
                for doc in relevant_docs[:2]:  # Use up to 2 docs for query
                    # Extract key terms from content
                    words = doc.content.split()
                    query_terms.extend(random.sample(words, min(3, len(words))))
                
                # Add domain-specific terms
                query_terms.extend(random.sample(template_data["topics"], 2))
                query = " ".join(query_terms[:6])  # Keep query reasonable length
                
                # Assign relevance scores (decreasing)
                relevance_scores = {}
                for i, doc in enumerate(relevant_docs):
                    score = 1.0 - (i * 0.15)  # 1.0, 0.85, 0.70, ...
                    relevance_scores[doc.id] = max(score, 0.3)
                
                # Determine access requirements
                max_classification = max(
                    [d.security_classification for d in relevant_docs],
                    key=lambda x: list(SecurityClassification).index(x)
                )
                required_role = random.choice(self._get_allowed_roles(max_classification))
                
                # Random role for testing (may or may not have access)
                test_role = random.choice(ROLES)
                expected_access = test_role in self._get_allowed_roles(max_classification)
                
                query_obj = GroundTruthQuery(
                    query=query,
                    relevant_doc_ids=[d.id for d in relevant_docs],
                    relevance_scores=relevance_scores,
                    domain=domain,
                    required_role=required_role,
                    required_classification=max_classification,
                    expected_access=expected_access,
                    attributes={
                        "test_role": test_role,
                        "num_relevant": num_relevant,
                        "difficulty": random.choice(["easy", "medium", "hard"])
                    }
                )
                
                queries.append(query_obj)
        
        self.queries = queries
        return queries
    
    def save_dataset(self, output_dir: Path):
        """Save generated dataset to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save documents
        docs_file = output_dir / "documents.jsonl"
        with open(docs_file, 'w') as f:
            for doc in self.documents:
                f.write(json.dumps(doc.dict()) + "\n")
        
        # Save queries
        queries_file = output_dir / "queries.jsonl"
        with open(queries_file, 'w') as f:
            for query in self.queries:
                f.write(json.dumps(asdict(query)) + "\n")
        
        # Save statistics
        stats = {
            "total_documents": len(self.documents),
            "total_queries": len(self.queries),
            "documents_by_domain": {},
            "documents_by_classification": {},
            "queries_by_domain": {}
        }
        
        for doc in self.documents:
            domain = doc.attributes.get("domain", "unknown")
            stats["documents_by_domain"][domain] = stats["documents_by_domain"].get(domain, 0) + 1
            
            classification = doc.security_classification.value
            stats["documents_by_classification"][classification] = \
                stats["documents_by_classification"].get(classification, 0) + 1
        
        for query in self.queries:
            stats["queries_by_domain"][query.domain] = \
                stats["queries_by_domain"].get(query.domain, 0) + 1
        
        stats_file = output_dir / "dataset_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Dataset saved to {output_dir}")
        print(f"Documents: {len(self.documents)}")
        print(f"Queries: {len(self.queries)}")
        print(f"Files: documents.jsonl, queries.jsonl, dataset_stats.json")
    
    @staticmethod
    def load_dataset(dataset_dir: Path) -> Tuple[List[Document], List[GroundTruthQuery]]:
        """Load dataset from files."""
        docs_file = dataset_dir / "documents.jsonl"
        queries_file = dataset_dir / "queries.jsonl"
        
        documents = []
        with open(docs_file, 'r') as f:
            for line in f:
                doc_dict = json.loads(line)
                documents.append(Document(**doc_dict))
        
        queries = []
        with open(queries_file, 'r') as f:
            for line in f:
                query_dict = json.loads(line)
                # Convert classification string to enum
                query_dict["required_classification"] = SecurityClassification(
                    query_dict["required_classification"]
                )
                queries.append(GroundTruthQuery(**query_dict))
        
        return documents, queries


if __name__ == "__main__":
    # Generate test dataset
    generator = DatasetGenerator(seed=42)
    
    print("Generating documents...")
    documents = generator.generate_documents(num_docs_per_domain=100)
    
    print("Generating queries...")
    queries = generator.generate_queries(num_queries_per_domain=50)
    
    print("Saving dataset...")
    output_dir = Path(__file__).parent / "data" / "test_dataset"
    generator.save_dataset(output_dir)
    
    print("Done!")
