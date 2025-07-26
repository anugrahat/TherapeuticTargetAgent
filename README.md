# Therapeutic Target Agent

A multi-tool agent that queries biological databases to find therapeutic targets. Integrates with PubMed, ChEMBL, and PDB to provide comprehensive target information.

## Quick Start

1. **Set up environment:**
   ```bash
   cd /home/agent
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure API keys:**
   - Edit `.env` file with your OpenAI API key (required)
   - Add NCBI API key for higher PubMed rate limits (optional)

3. **Run a query:**
   ```bash
   python therapeutic_target_agent.py "Find human proteins that bind HIV-1 gp120 and have sub-100 nM inhibitors"
   ```

## Usage Examples

### Single Target Queries
```bash
# Find HIV host targets
python therapeutic_target_agent.py "Find host proteins that interact with HIV-1 gp120"

# Find ultra-potent EGFR inhibitors
python therapeutic_target_agent.py "Find EGFR inhibitors under 10 nM"

# Range-based potency filtering
python therapeutic_target_agent.py "Find SARS-CoV-2 targets with inhibitors between 1 nM and 100 nM"
```

### Multi-Target Queries
```bash
# Cancer kinase targets
python therapeutic_target_agent.py "Show inhibitors for EGFR, JAK2 and CDK9 under 100 nM"

# Neurological targets
python therapeutic_target_agent.py "Find inhibitors for BACE1, MAOB and AChE between 5 nM and 50 nM"
```

### Advanced Options
```bash
# Use different LLM models
python therapeutic_target_agent.py "your query" --model gpt-4o           # Most capable
python therapeutic_target_agent.py "your query" --model gpt-4o-mini     # Default, fast
python therapeutic_target_agent.py "your query" --model gpt-3.5-turbo   # Cost-effective
```

## Output

The agent provides:
- **Console table**: Ranked targets with inhibitors, IC₅₀ values, PDB IDs, and evidence counts
- **JSON file**: Machine-readable results saved as `ranked_hits.json`

## Architecture

- **PubMedTool**: Searches scientific literature via NCBI E-utilities
- **ChEMBLTool**: Finds drug/inhibitor data from ChEMBL database
- **PDBTool**: Searches protein structures from RCSB PDB
- **TherapeuticTargetAgent**: LangChain agent orchestrating all tools

## Next Steps Roadmap

| Tier         | Feature                    | Implementation                                    |
|--------------|----------------------------|---------------------------------------------------|
| Nice-to-have | SQLite/Redis caching       | Wrap API calls with `cachecontrol` decorator     |
|              | Auto-download PDB files    | Loop over PDB IDs with `requests.get()`          |
|              | Export to CSV/Excel        | `pandas.DataFrame(hits).to_csv()`                |
| Level-up     | Neo4j ingestion           | `pip install neo4j`, Cypher `UNWIND $hits AS h` |
|              | Web UI dashboard          | FastAPI or Streamlit wrapper                     |
|              | Docker packaging          | Dockerfile with `pip install -r requirements.txt`|

## Troubleshooting

| Symptom                              | Likely Cause             | Fix                                |
|--------------------------------------|-------------------------|------------------------------------|
| `403` from PubMed                    | Over quota              | Get NCBI API key                   |
| Empty ChEMBL hits                    | Gene ↔ target mismatch  | Add UniProt→ChEMBL mapping step    |
| LLM "function call parsing" error    | Model hallucinated      | Keep `temperature=0.0`             |
