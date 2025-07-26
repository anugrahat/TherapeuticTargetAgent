#!/usr/bin/env python3
"""
Therapeutic Target Agent

A multi-tool agent that queries biological databases to find therapeutic targets.
Integrates with PubMed, ChEMBL, and PDB to provide comprehensive target information.
"""

import os
import json
import requests
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import xmltodict
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field


# Load environment variables
load_dotenv()

@dataclass
class TargetHit:
    """Represents a therapeutic target with associated data."""
    gene_symbol: str
    protein_name: str
    inhibitors: List[Dict[str, Any]]
    pdb_ids: List[str]
    pubmed_evidence: List[str]
    score: float = 0.0

class PubMedTool:
    """Tool for querying PubMed database."""
    
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.api_key = os.getenv("NCBI_API_KEY")
        
    def search(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Search PubMed for articles matching the query."""
        # Search for PMIDs
        search_url = f"{self.base_url}esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json"
        }
        if self.api_key:
            search_params["api_key"] = self.api_key
            
        try:
            response = requests.get(search_url, params=search_params, timeout=30)
            response.raise_for_status()
            search_data = response.json()
            
            pmids = search_data.get("esearchresult", {}).get("idlist", [])
            if not pmids:
                return []
            
            # Fetch article details
            fetch_url = f"{self.base_url}efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml"
            }
            if self.api_key:
                fetch_params["api_key"] = self.api_key
                
            fetch_response = requests.get(fetch_url, params=fetch_params, timeout=30)
            fetch_response.raise_for_status()
            
            # Parse XML response
            xml_data = xmltodict.parse(fetch_response.text)
            articles = []
            
            pubmed_articles = xml_data.get("PubmedArticleSet", {}).get("PubmedArticle", [])
            if not isinstance(pubmed_articles, list):
                pubmed_articles = [pubmed_articles]
                
            for article in pubmed_articles:
                try:
                    medline = article.get("MedlineCitation", {})
                    pmid = medline.get("PMID", {}).get("#text", "")
                    
                    article_data = medline.get("Article", {})
                    title = article_data.get("ArticleTitle", "")
                    
                    abstract_data = article_data.get("Abstract", {})
                    abstract = ""
                    if abstract_data:
                        abstract_text = abstract_data.get("AbstractText", "")
                        if isinstance(abstract_text, list):
                            abstract = " ".join([text.get("#text", text) if isinstance(text, dict) else str(text) for text in abstract_text])
                        elif isinstance(abstract_text, dict):
                            abstract = abstract_text.get("#text", "")
                        else:
                            abstract = str(abstract_text)
                    
                    articles.append({
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract
                    })
                except Exception as e:
                    print(f"Error parsing article: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            print(f"PubMed search error: {e}")
            # Fallback to Europe PMC
            print("Trying Europe PMC fallback...")
            return self._search_europe_pmc(query, max_results)
    
    def _search_europe_pmc(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Fallback search using Europe PMC when NCBI is down."""
        try:
            url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            params = {
                "query": query,
                "format": "json", 
                "pageSize": min(max_results, 25),
                "resultType": "core"
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            result_list = data.get("resultList", {}).get("result", [])
            
            for item in result_list:
                try:
                    # Map Europe PMC format to our expected format
                    pmid = item.get("pmid", item.get("id", ""))
                    title = item.get("title", "")
                    abstract = item.get("abstractText", "")
                    
                    articles.append({
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract
                    })
                except Exception as e:
                    print(f"Error parsing Europe PMC article: {e}")
                    continue
            
            print(f"Found {len(articles)} articles from Europe PMC")
            return articles
            
        except Exception as e:
            print(f"Europe PMC search error: {e}")
            return []

class ChEMBLTool:
    """Tool for querying ChEMBL database for drug/inhibitor information."""
    
    def __init__(self):
        self.base_url = "https://www.ebi.ac.uk/chembl/api/data"
        
    def get_inhibitors_for_target(self, gene_symbol: str, max_ic50_nm: float = None, min_ic50_nm: float = None) -> List[Dict[str, Any]]:
        """Find inhibitors for a given gene/target."""
        try:
            # First, search for target by gene symbol
            target_url = f"{self.base_url}/target.json"
            target_params = {
                "target_synonym__icontains": gene_symbol,
                "format": "json",
                "limit": 10
            }
            
            response = requests.get(target_url, params=target_params, timeout=30)
            response.raise_for_status()
            target_data = response.json()
            
            targets = target_data.get("targets", [])
            if not targets:
                return []
            
            # Get the first matching target
            target = targets[0]
            chembl_id = target.get("target_chembl_id")
            
            if not chembl_id:
                return []
            
            # Search for activities (inhibitors) for this target
            activity_url = f"{self.base_url}/activity.json"
            activity_params = {
                "target_chembl_id": chembl_id,
                "standard_type__in": "IC50,Ki,Kd",
                "standard_relation": "=",
                "format": "json",
                "limit": 100  # Balanced limit for performance vs coverage
            }
            
            # Add IC50 upper bound filter if threshold specified
            if max_ic50_nm:
                activity_params["standard_value__lte"] = max_ic50_nm
                activity_params["standard_units"] = "nM"
            
            activity_response = requests.get(activity_url, params=activity_params, timeout=30)
            activity_response.raise_for_status()
            activity_data = activity_response.json()
            
            activities = activity_data.get("activities", [])
            inhibitors = []
            
            for activity in activities:
                try:
                    compound_chembl_id = activity.get("molecule_chembl_id")
                    if not compound_chembl_id:
                        continue
                    
                    # Get compound details
                    compound_url = f"{self.base_url}/molecule/{compound_chembl_id}.json"
                    compound_response = requests.get(compound_url, timeout=30)
                    compound_response.raise_for_status()
                    compound_data = compound_response.json()
                    
                    ic50_value = activity.get("standard_value")
                    if not ic50_value:
                        continue
                    
                    # Convert IC50 to float for comparison
                    try:
                        ic50_float = float(ic50_value)
                    except (ValueError, TypeError):
                        continue
                    
                    # Apply range filtering if specified
                    if max_ic50_nm and ic50_float > max_ic50_nm:
                        continue
                    if min_ic50_nm and ic50_float < min_ic50_nm:
                        continue
                    
                    inhibitor = {
                        "name": compound_data.get("pref_name") or "Unknown",
                        "chembl_id": compound_chembl_id,
                        "ic50_nm": ic50_value,
                        "assay_type": activity.get("standard_type"),
                        "activity_id": activity.get("activity_id")
                    }
                    
                    inhibitors.append(inhibitor)
                        
                except Exception as e:
                    print(f"Error processing compound: {e}")
                    continue
            
            # Sort by IC50 (lower is better)
            inhibitors.sort(key=lambda x: float(x["ic50_nm"]) if x["ic50_nm"] else float('inf'))
            
            # INTELLIGENT SAMPLING: Don't just take top 10, sample across potency spectrum
            if len(inhibitors) > 10:
                # Sample every n-th compound to get diversity across IC50 range
                step = max(1, len(inhibitors) // 10)
                sampled = inhibitors[::step][:10]
                print(f"Sampled {len(sampled)} compounds from {len(inhibitors)} total (every {step}th compound)")
                return sampled
            else:
                return inhibitors[:10]  # Return all if ≤10 found
            
        except Exception as e:
            print(f"ChEMBL search error: {e}")
            return []

class PDBTool:
    """Tool for querying Protein Data Bank - REAL v2 API ONLY."""
    
    def __init__(self):
        self.search_url = "https://search.rcsb.org/rcsbsearch/v2/query"  # FIXED: v2 not v1!
        self.data_url = "https://data.rcsb.org/rest/v1/core/entry"
    
    def search_structures_for_gene(self, gene_symbol: str) -> List[str]:
        """Search for PDB structures containing the gene/protein."""
        return self._search_by_gene(gene_symbol)
    
    def search_structures_for_ligand(self, chembl_id: str) -> List[str]:
        """Search for PDB structures containing a specific ChEMBL ligand."""
        return self._search_by_ligand(chembl_id)
    
    def search_structures(self, gene_symbol: str) -> List[str]:
        """Search for PDB structures using the WORKING v2 API."""
        try:
            # Use the correct v2 API payload - full text search
            payload = {
                "query": {
                    "type": "terminal",
                    "service": "full_text",
                    "parameters": {"value": gene_symbol}
                },
                "return_type": "entry",
                "request_options": {"paginate": {"start": 0, "rows": 20}}
            }
            
            response = requests.post(self.search_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                pdb_ids = [r["identifier"] for r in data.get("result_set", [])]
                return pdb_ids[:10]  # Return top 10
            else:
                print(f"PDB v2 API error: {response.status_code} - {response.text[:200]}")
                return self._try_alternate_search(gene_symbol)
                
        except Exception as e:
            print(f"PDB search error: {e}")
            return self._try_alternate_search(gene_symbol)
    
    def _search_by_gene(self, gene_symbol: str) -> List[str]:
        """Search PDB by gene symbol."""
        try:
            # Use the correct v2 API payload - full text search
            payload = {
                "query": {
                    "type": "terminal",
                    "service": "full_text",
                    "parameters": {"value": gene_symbol}
                },
                "return_type": "entry",
                "request_options": {"paginate": {"start": 0, "rows": 20}}
            }
            
            response = requests.post(self.search_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                pdb_ids = [r["identifier"] for r in data.get("result_set", [])]
                return pdb_ids[:10]  # Return top 10
            else:
                return []
                
        except Exception as e:
            print(f"PDB gene search error: {e}")
            return []
    
    def _search_by_ligand(self, chembl_id: str) -> List[str]:
        """Search PDB for structures containing a specific ChEMBL ligand."""
        try:
            # Search for ChEMBL ID in PDB ligand data
            payload = {
                "query": {
                    "type": "terminal",
                    "service": "full_text",
                    "parameters": {"value": chembl_id}
                },
                "return_type": "entry",
                "request_options": {"paginate": {"start": 0, "rows": 10}}
            }
            
            response = requests.post(self.search_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                pdb_ids = [r["identifier"] for r in data.get("result_set", [])]
                return pdb_ids[:5]  # Return top 5 for ligand matches
            else:
                return []
                
        except Exception as e:
            print(f"PDB ligand search error: {e}")
            return []
    
    def _try_alternate_search(self, gene_symbol: str) -> List[str]:
        """Try alternate search methods when main search fails."""
        return self._search_by_gene(gene_symbol)

class TherapeuticTargetAgent:
    """Main agent that orchestrates database queries and analysis."""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.pubmed_tool = PubMedTool()
        self.chembl_tool = ChEMBLTool()
        self.pdb_tool = PDBTool()
        
        # Store tool results for structured output
        self.tool_results = {
            "pubmed_queries": [],
            "targets_found": {},  # gene_symbol -> {inhibitors, pdb_ids, papers}
            "all_papers": []
        }
        
        # Create tools for the LangChain agent
        self.tools = [
            Tool(
                name="search_pubmed",
                description="Search PubMed for scientific articles. Input should be a search query.",
                func=self._search_pubmed
            ),
            Tool(
                name="find_inhibitors",
                description="Find inhibitors for a gene/protein target from ChEMBL. Input should be a gene symbol.",
                func=self._find_inhibitors
            ),
            Tool(
                name="search_pdb",
                description="Search PDB for protein structures. Input should be a gene symbol.",
                func=self._search_pdb
            )
        ]
        
        # Create the agent
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a therapeutic target discovery agent. Your goal is to help researchers find potential therapeutic targets by querying biological databases.

For EVERY query, you MUST call ALL THREE tools in this exact order:
1. FIRST: Call 'search_pubmed' to find relevant literature
2. SECOND: Call 'find_inhibitors' to get inhibitor data from ChEMBL
3. THIRD: Call 'search_pdb' to find protein structures

Do NOT skip any tools. Always call all three tools for each target gene you identify.

Extract gene symbols from the query (like BACE1, APP, PIK3CG) and run all three tools for each gene.

After calling all tools, synthesize the results and answer the user's specific filtering requests (like "only below 1 nM")."""),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        self.agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
    
    def _search_pubmed(self, query: str) -> str:
        """Search PubMed and return formatted results."""
        articles = self.pubmed_tool.search(query)
        if not articles:
            return f"No PubMed articles found for query: {query}"
        
        # Store results for structured output
        self.tool_results["pubmed_queries"].append({
            "query": query,
            "articles": articles
        })
        self.tool_results["all_papers"].extend(articles)
        
        result = f"Found {len(articles)} PubMed articles for '{query}':\n\n"
        for article in articles[:5]:  # Show top 5
            result += f"PMID: {article['pmid']}\n"
            result += f"Title: {article['title']}\n"
            if article['abstract']:
                result += f"Abstract: {article['abstract'][:200]}...\n"
            result += "\n"
        
        return result
    
    def _find_inhibitors(self, gene_symbol: str) -> str:
        """Find inhibitors and cross-validate with PDB structures."""
        # Parse range from current query
        min_nm, max_nm = self._parse_ic50_range(self.current_query)
        inhibitors = self.chembl_tool.get_inhibitors_for_target(gene_symbol, max_nm, min_nm)
        if not inhibitors:
            return f"No inhibitors found for {gene_symbol}"
        
        # Store results for structured output
        if gene_symbol not in self.tool_results["targets_found"]:
            self.tool_results["targets_found"][gene_symbol] = {
                "gene_symbol": gene_symbol,
                "protein_name": f"{gene_symbol} protein",  # Could be enhanced
                "inhibitors": [],
                "pdb_ids": [],
                "pubmed_evidence": []
            }
        
        # Cross-validate each inhibitor with PDB structures
        validated_inhibitors = []
        for inhibitor in inhibitors:
            chembl_id = inhibitor.get("chembl_id", "")
            if chembl_id:
                # Search for PDB structures containing this specific ligand
                pdb_structures = self.pdb_tool.search_structures_for_ligand(chembl_id)
                inhibitor["pdb_structures"] = pdb_structures
            else:
                inhibitor["pdb_structures"] = []
            validated_inhibitors.append(inhibitor)
        
        self.tool_results["targets_found"][gene_symbol]["inhibitors"] = validated_inhibitors
        
        result = f"Found {len(inhibitors)} inhibitors for {gene_symbol}:\n\n"
        for inhibitor in validated_inhibitors[:5]:  # Show top 5
            result += f"Name: {inhibitor['name']}\n"
            result += f"ChEMBL ID: {inhibitor['chembl_id']}\n"
            result += f"IC50: {inhibitor['ic50_nm']} nM\n"
            result += f"Assay Type: {inhibitor['assay_type']}\n"
            
            # Show which PDB structures contain this inhibitor
            pdb_structures = inhibitor.get("pdb_structures", [])
            if pdb_structures:
                result += f"PDB Structures: {', '.join(pdb_structures)}\n"
            else:
                result += f"PDB Structures: None found\n"
            result += "\n"
        
        return result
    
    def _search_pdb(self, gene_symbol: str) -> str:
        """Search PDB and return formatted results."""
        pdb_ids = self.pdb_tool.search_structures(gene_symbol)
        if not pdb_ids:
            return f"No PDB structures found for {gene_symbol}"
        
        # Store results for structured output
        if gene_symbol not in self.tool_results["targets_found"]:
            self.tool_results["targets_found"][gene_symbol] = {
                "gene_symbol": gene_symbol,
                "protein_name": f"{gene_symbol} protein",
                "inhibitors": [],
                "pdb_ids": [],
                "pubmed_evidence": []
            }
        
        self.tool_results["targets_found"][gene_symbol]["pdb_ids"] = pdb_ids
        
        return f"Found {len(pdb_ids)} PDB structures for {gene_symbol}: {', '.join(pdb_ids)}"
    
    def query(self, user_query: str) -> Dict[str, Any]:
        """Process a user query and return structured results."""
        try:
            # Store the original query
            self.current_query = user_query
            
            # Reset tool results for new query
            self.tool_results = {
                "targets_found": {},
                "all_papers": [],
                "pubmed_queries": []
            }
            
            # Run the agent
            result = self.agent_executor.invoke({"input": user_query})
            
            # Extract and structure the information
            structured_result = self._extract_structured_data(result)
            return structured_result
            
        except Exception as e:
            return {"error": str(e), "raw_output": ""}

    def _extract_structured_data(self, agent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured data from agent output."""
        output = agent_result.get("output", "")
        
        # Build targets list from tool results
        targets = []
        for gene_symbol, target_data in self.tool_results["targets_found"].items():
            # Add paper evidence from all papers mentioning this gene
            papers = []
            for paper in self.tool_results["all_papers"]:
                try:
                    if isinstance(paper, dict):
                        title = str(paper.get("title", "") or "")
                        abstract = str(paper.get("abstract", "") or "")
                        if gene_symbol.lower() in title.lower() or gene_symbol.lower() in abstract.lower():
                            papers.append(paper)
                except Exception as e:
                    continue  # Skip problematic papers
            
            target_data["pubmed_evidence"] = [p.get("pmid", "") for p in papers[:5] if isinstance(p, dict)]  # Top 5 papers
            target_data["papers"] = papers[:3]  # Top 3 for display
            
            # Calculate score
            target_data["score"] = self._calculate_score(target_data)
            
            targets.append(target_data)

        # Sort by score (descending)
        targets.sort(key=lambda x: x.get("score", 0), reverse=True)

        return {
            "raw_output": output,
            "targets": targets,
            "query": self.current_query,  # Add the original query
            "timestamp": "2025-01-25",
            "total_papers": len(self.tool_results["all_papers"]),
            "queries_performed": len(self.tool_results["pubmed_queries"])
        }

    def _parse_ic50_range(self, query: str):
        """Parse IC50 range from natural language query. Returns (min_nm, max_nm)."""
        import re
        
        # Convert query to lowercase for easier matching
        query_lower = query.lower()
        
        # First, look for "between X and Y" patterns
        between_patterns = [
            r'between\s+(\d*\.?\d+)\s*n?m\s+and\s+(\d*\.?\d+)\s*n?m',
            r'between\s+(\d*\.?\d+)\s*and\s+(\d*\.?\d+)\s*n?m',
            r'between\s+(\d*\.?\d+)\s*n?m\s+and\s+(\d*\.?\d+)\s*μ?m',
            r'between\s+(\d*\.?\d+)\s*and\s+(\d*\.?\d+)\s*μ?m',
            r'between\s+(\d*\.?\d+)\s*and\s+(\d*\.?\d+)\s*micromolar',
            r'from\s+(\d*\.?\d+)\s*to\s+(\d*\.?\d+)\s*n?m',
            r'(\d*\.?\d+)\s*-\s*(\d*\.?\d+)\s*n?m'
        ]
        
        for pattern in between_patterns:
            match = re.search(pattern, query_lower)
            if match:
                min_val, max_val = map(float, match.groups())
                
                # Convert to nM if in μM/micromolar
                if 'μm' in pattern or 'micromolar' in pattern:
                    if 'n?m' not in pattern:  # Only convert if both are μM
                        min_val *= 1000
                        max_val *= 1000
                    elif pattern.endswith('μ?m'):  # Only max is μM
                        max_val *= 1000
                        
                print(f"Parsed IC50 range: {min_val}-{max_val} nM from query: '{query}'")
                return (min_val, max_val)
        
        # Fallback to single upper bound patterns
        upper_patterns = [
            r'under\s+(\d*\.?\d+)\s*n?m',
            r'below\s+(\d*\.?\d+)\s*n?m', 
            r'less\s+than\s+(\d*\.?\d+)\s*n?m',
            r'<\s*(\d*\.?\d+)\s*n?m',
            r'under\s+(\d*\.?\d+)\s*μ?m',
            r'below\s+(\d*\.?\d+)\s*μ?m',
            r'less\s+than\s+(\d*\.?\d+)\s*μ?m',
            r'<\s*(\d*\.?\d+)\s*μ?m',
            r'under\s+(\d*\.?\d+)\s*micromolar',
            r'below\s+(\d*\.?\d+)\s*micromolar'
        ]
        
        for pattern in upper_patterns:
            match = re.search(pattern, query_lower)
            if match:
                max_val = float(match.group(1))
                
                # Convert to nM if in μM/micromolar
                if 'μm' in pattern or 'micromolar' in pattern:
                    max_val *= 1000  # Convert μM to nM
                    
                print(f"Parsed IC50 upper bound: ≤{max_val} nM from query: '{query}'")
                return (None, max_val)
                
        # If no threshold found, return None (no filtering)
        print(f"No IC50 range detected in query: '{query}'")
        return (None, None)
    
    def _calculate_score(self, target_data: Dict[str, Any]) -> float:
        """Calculate a comprehensive score for a target."""
        score = 0.0

        # Score based on inhibitor potency (lower IC50 = higher score)
        inhibitors = target_data.get("inhibitors", [])
        if inhibitors:
            best_ic50 = min([float(inh.get("ic50_nm", float('inf'))) for inh in inhibitors if inh.get("ic50_nm")])
            if best_ic50 < float('inf'):
                # Inverse relationship: better inhibitors (lower IC50) get higher scores
                if best_ic50 <= 10:  # Very potent
                    score += 2.0
                elif best_ic50 <= 100:  # Good
                    score += 1.0
                elif best_ic50 <= 1000:  # Moderate
                    score += 0.5
                else:  # Weak
                    score += 0.1
                    
                # Bonus for having multiple inhibitors
                score += min(len(inhibitors) * 0.1, 0.5)
        
        # Score based on structural data availability
        pdb_ids = target_data.get("pdb_ids", [])
        if pdb_ids:
            score += min(len(pdb_ids) * 0.2, 1.0)  # Max 1.0 for structures
        
        # Score based on literature evidence
        evidence = target_data.get("pubmed_evidence", [])
        if evidence:
            score += min(len(evidence) * 0.1, 0.5)  # Max 0.5 for papers
        
        return round(score, 3)

def _simple_score(hit: Dict[str, Any]) -> float:
    """Calculate a simple relevance score for a target hit."""
    score = 0.0
    
    # Score based on inhibitor potency (lower IC50 = higher score)
    inhibitors = hit.get("inhibitors", [])
    if inhibitors:
        best_ic50 = min([float(inh.get("ic50_nm", float('inf'))) for inh in inhibitors if inh.get("ic50_nm")])
        if best_ic50 < float('inf'):
            score += 1.0 / (best_ic50 + 1)  # Inverse relationship with IC50
    
    # Score based on structural data availability
    pdb_ids = hit.get("pdb_ids", [])
    score += len(pdb_ids) * 0.1
    
    # Score based on literature evidence
    evidence = hit.get("pubmed_evidence", [])
    score += len(evidence) * 0.05
    
    return round(score, 3)

def rank_and_print(results: Dict[str, Any]) -> None:
    """Print ranked results as a nicely formatted table."""
    targets = results.get("targets", [])
    
    if not targets:
        print("\n❌ No targets found.")
        return
    
    print(f"\nQuery: {results.get('query', 'N/A')}")
    print("=" * 50)
    
    # Separately show PDB structures to avoid confusion
    print(f"\n🧬 **AVAILABLE PDB STRUCTURES:**")
    print("=" * 40)
    for target in targets:
        gene = target.get("gene_symbol", "N/A")
        pdb_ids = target.get("pdb_ids", [])
        
        if pdb_ids:
            print(f"\n🎯 **{gene}**: {len(pdb_ids)} structures found")
            # Show in groups of 10 per line
            for i in range(0, len(pdb_ids), 10):
                pdb_group = pdb_ids[i:i+10]
                print(f"   {', '.join(pdb_group)}")
        else:
            print(f"\n🎯 **{gene}**: No structures found")
    
    # Show top papers
    print(f"\n📚 **TOP PAPERS FROM PUBMED:**")
    print("=" * 60)
    for target in targets:
        gene = target.get("gene_symbol", "N/A")
        protein_name = target.get("protein_name", "")
        papers = target.get("papers", [])
        
        if papers:
            print(f"\n🎯 **{gene}** ({protein_name})")
            print("-" * 50)
            for i, paper in enumerate(papers, 1):
                print(f"{i}. {paper.get('title', 'No title')}")
                print(f"   PMID: {paper.get('pmid', 'N/A')}")
    
    # Export to JSON
    with open("ranked_hits.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Saved to ranked_hits.json")
    print(f"✅ Found {len(targets)} real targets from intelligent agent!")

def main():
    parser = argparse.ArgumentParser(description="Therapeutic Target Discovery Agent")
    parser.add_argument("query", help="Search query for therapeutic targets")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    
    args = parser.parse_args()
    
    # Initialize the agent
    agent = TherapeuticTargetAgent(model_name=args.model)
    
    # Process the query
    print(f"Searching for: {args.query}")
    print("=" * 50)
    
    result = agent.query(args.query)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    # Print raw LLM output first
    print(result["raw_output"])
    print("\n" + "=" * 60)
    
    # Now show the structured, scored results
    rank_and_print(result)

if __name__ == "__main__":
    main()
