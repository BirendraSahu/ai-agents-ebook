"""
Streamlit App: Knowledge Graphs for Agentic Reasoning Demo
Demonstrates KG concepts, ontology, multi-hop reasoning, and GraphRAG using local graph DB.
"""

import streamlit as st
import os
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from openai import OpenAI
import pandas as pd
import networkx as nx
from collections import defaultdict
import re

# Try to import plotly for graph visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Try to import RDFLib for SPARQL-like queries
try:
    from rdflib import Graph, Namespace, Literal, URIRef
    from rdflib.namespace import RDF, RDFS, OWL
    RDFLIB_AVAILABLE = True
except ImportError:
    RDFLIB_AVAILABLE = False
    st.warning("⚠️ RDFLib not installed. Install with: `pip install rdflib` for full SPARQL support")

# Page config
st.set_page_config(
    page_title="Knowledge Graph Demo",
    page_icon="🕸️",
    layout="wide"
)

# Initialize session state
if "openai_client" not in st.session_state:
    st.session_state.openai_client = None
if "knowledge_graph" not in st.session_state:
    st.session_state.knowledge_graph = defaultdict(list)  # {subject: [(predicate, object), ...]}
if "ontology" not in st.session_state:
    st.session_state.ontology = {
        "classes": set(),
        "properties": set(),
        "relationships": {}
    }
if "graph_visualization" not in st.session_state:
    st.session_state.graph_visualization = None

# ============================================================================
# Graph Database Implementation (Simple In-Memory)
# ============================================================================

class SimpleGraphDB:
    """Simple in-memory Knowledge Graph database."""
    
    def __init__(self):
        self.triples = []  # List of (subject, predicate, object) tuples
        self.entities = {}  # Entity metadata
        self.ontology = {
            "classes": set(),
            "object_properties": set(),
            "data_properties": set(),
            "relationships": {}
        }
    
    def add_triple(self, subject: str, predicate: str, obj: str, context: str = None):
        """Add a triple to the graph."""
        triple = (subject, predicate, obj)
        if triple not in self.triples:
            self.triples.append(triple)
            # Track entities
            if subject not in self.entities:
                self.entities[subject] = {"type": None, "properties": {}}
            if obj not in self.entities and not self._is_literal(obj):
                self.entities[obj] = {"type": None, "properties": {}}
    
    def _is_literal(self, value: str) -> bool:
        """Check if value is a literal (string, number) vs entity."""
        # Simple heuristic: if it starts with number or is quoted, it's likely a literal
        return value.startswith('"') or value.replace('.', '').replace('-', '').isdigit()
    
    def query(self, subject: str = None, predicate: str = None, obj: str = None) -> List[Tuple]:
        """Query triples (simple pattern matching)."""
        results = []
        for s, p, o in self.triples:
            if (subject is None or s == subject) and \
               (predicate is None or p == predicate) and \
               (obj is None or o == obj):
                results.append((s, p, o))
        return results
    
    def multi_hop_query(self, start_entity: str, path: List[str], max_hops: int = 3) -> List[Dict]:
        """Perform multi-hop reasoning along a path of predicates."""
        results = []
        
        def traverse(current: str, remaining_path: List[str], visited: Set[str], depth: int):
            if depth > max_hops or not remaining_path:
                return
            
            if current in visited:
                return
            
            visited.add(current)
            predicate = remaining_path[0]
            next_path = remaining_path[1:]
            
            # Find all objects connected via this predicate
            matches = self.query(subject=current, predicate=predicate)
            
            for s, p, o in matches:
                if not next_path:
                    # End of path
                    results.append({
                        "path": [current] + [o],
                        "predicates": [predicate],
                        "end_entity": o
                    })
                else:
                    # Continue traversal
                    traverse(o, next_path, visited.copy(), depth + 1)
        
        traverse(start_entity, path, set(), 0)
        return results
    
    def get_entity_neighbors(self, entity: str) -> List[Dict]:
        """Get all neighbors of an entity."""
        neighbors = []
        for s, p, o in self.triples:
            if s == entity:
                neighbors.append({"relationship": p, "target": o, "direction": "outgoing"})
            elif o == entity:
                neighbors.append({"relationship": p, "target": s, "direction": "incoming"})
        return neighbors
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX graph for visualization."""
        G = nx.DiGraph()
        for s, p, o in self.triples:
            if not self._is_literal(o):
                G.add_edge(s, o, label=p, relationship=p)
            else:
                # For literals, create a node with the value
                literal_node = f"{s}_{p}_literal"
                G.add_edge(s, literal_node, label=f"{p}: {o}", relationship=p)
        return G
    
    def get_stats(self) -> Dict:
        """Get graph statistics."""
        entities = set()
        for s, p, o in self.triples:
            entities.add(s)
            if not self._is_literal(o):
                entities.add(o)
        
        return {
            "total_triples": len(self.triples),
            "total_entities": len(entities),
            "unique_predicates": len(set(p for _, p, _ in self.triples)),
            "classes": len(self.ontology["classes"]),
            "properties": len(self.ontology["object_properties"]) + len(self.ontology["data_properties"])
        }

# Initialize graph DB
if "graph_db" not in st.session_state:
    st.session_state.graph_db = SimpleGraphDB()

# ============================================================================
# Helper Functions
# ============================================================================

def create_sample_ontology():
    """Create a sample corporate project management ontology."""
    ontology = {
        "classes": {
            "Employee": "A person working in the organization",
            "Project": "A work initiative or project",
            "Skill": "A technical or professional skill",
            "Department": "An organizational unit",
            "Program": "A collection of related projects",
            "Portfolio": "A collection of programs"
        },
        "object_properties": {
            "manages": ("Employee", "Project"),
            "worksIn": ("Employee", "Department"),
            "possesses": ("Employee", "Skill"),
            "requires": ("Project", "Skill"),
            "isPartOf": ("Project", "Program"),
            "isPartOf": ("Program", "Portfolio"),
            "dependsOn": ("Project", "Project"),
            "reportsTo": ("Employee", "Employee")
        },
        "data_properties": {
            "hasName": ("Employee", "string"),
            "hasBudget": ("Project", "decimal"),
            "hasStartDate": ("Project", "date"),
            "hasEndDate": ("Project", "date"),
            "hasLevel": ("Skill", "string")
        },
        "rules": {
            "inverse": {
                "manages": "isManagedBy",
                "reportsTo": "manages"  # Inverse: if A reportsTo B, then B manages A
            },
            "transitive": {
                "isPartOf": True,  # If A isPartOf B and B isPartOf C, then A isPartOf C
                "reportsTo": True  # If A reportsTo B and B reportsTo C, then A reportsTo C
            }
        }
    }
    return ontology

def populate_sample_graph(graph_db: SimpleGraphDB):
    """Populate graph with sample corporate data."""
    # Employees
    graph_db.add_triple("Employee:Alice", "hasName", '"Alice Johnson"')
    graph_db.add_triple("Employee:Alice", "worksIn", "Department:Engineering")
    graph_db.add_triple("Employee:Alice", "possesses", "Skill:Python")
    graph_db.add_triple("Employee:Alice", "possesses", "Skill:Cloud_Architecture")
    graph_db.add_triple("Employee:Alice", "manages", "Project:Payment_Service")
    graph_db.add_triple("Employee:Alice", "reportsTo", "Employee:Bob")
    
    graph_db.add_triple("Employee:Bob", "hasName", '"Bob Smith"')
    graph_db.add_triple("Employee:Bob", "worksIn", "Department:Engineering")
    graph_db.add_triple("Employee:Bob", "possesses", "Skill:Leadership")
    graph_db.add_triple("Employee:Bob", "manages", "Project:Data_Pipeline")
    graph_db.add_triple("Employee:Bob", "reportsTo", "Employee:Carol")
    
    graph_db.add_triple("Employee:Carol", "hasName", '"Carol Williams"')
    graph_db.add_triple("Employee:Carol", "worksIn", "Department:Engineering")
    graph_db.add_triple("Employee:Carol", "manages", "Program:Digital_Transformation")
    
    graph_db.add_triple("Employee:John", "hasName", '"John Doe"')
    graph_db.add_triple("Employee:John", "worksIn", "Department:Engineering")
    graph_db.add_triple("Employee:John", "possesses", "Skill:Python")
    graph_db.add_triple("Employee:John", "possesses", "Skill:Cloud_Architecture")
    graph_db.add_triple("Employee:John", "manages", "Project:Project_X")
    
    # Projects
    graph_db.add_triple("Project:Payment_Service", "hasName", '"Payment Service"')
    graph_db.add_triple("Project:Payment_Service", "hasBudget", "500000")
    graph_db.add_triple("Project:Payment_Service", "requires", "Skill:Python")
    graph_db.add_triple("Project:Payment_Service", "requires", "Skill:Cloud_Architecture")
    graph_db.add_triple("Project:Payment_Service", "dependsOn", "Project:Database_Service")
    graph_db.add_triple("Project:Payment_Service", "isPartOf", "Program:Digital_Transformation")
    
    graph_db.add_triple("Project:Data_Pipeline", "hasName", '"Data Pipeline"')
    graph_db.add_triple("Project:Data_Pipeline", "hasBudget", "300000")
    graph_db.add_triple("Project:Data_Pipeline", "requires", "Skill:Python")
    graph_db.add_triple("Project:Data_Pipeline", "isPartOf", "Program:Digital_Transformation")
    
    graph_db.add_triple("Project:Project_X", "hasName", '"Project X"')
    graph_db.add_triple("Project:Project_X", "requires", "Skill:Python")
    graph_db.add_triple("Project:Project_X", "requires", "Skill:Cloud_Architecture")
    
    graph_db.add_triple("Project:Database_Service", "hasName", '"Database Service"')
    graph_db.add_triple("Project:Database_Service", "requires", "Skill:Database_Design")
    
    # Programs and Portfolio
    graph_db.add_triple("Program:Digital_Transformation", "hasName", '"Digital Transformation Program"')
    graph_db.add_triple("Program:Digital_Transformation", "isPartOf", "Portfolio:Strategic_Initiatives")
    
    graph_db.add_triple("Portfolio:Strategic_Initiatives", "hasName", '"Strategic Initiatives Portfolio"')
    
    # Skills
    graph_db.add_triple("Skill:Python", "hasLevel", '"Advanced"')
    graph_db.add_triple("Skill:Cloud_Architecture", "hasLevel", '"Expert"')
    graph_db.add_triple("Skill:Leadership", "hasLevel", '"Senior"')
    graph_db.add_triple("Skill:Database_Design", "hasLevel", '"Intermediate"')
    
    # Departments
    graph_db.add_triple("Department:Engineering", "hasName", '"Engineering Department"')

def query_graph_natural_language(client: OpenAI, query: str, graph_db: SimpleGraphDB) -> Dict:
    """Convert natural language query to graph traversal and execute."""
    try:
        # Get all entity names from graph for context
        entity_names = []
        for s, p, o in graph_db.triples:
            if p == "hasName":
                entity_names.append(o.strip('"'))
        
        # Step 1: Extract entities and relationships from query
        prompt = f"""Analyze this query and extract entities and relationships for a Knowledge Graph query.

Query: "{query}"

Available entities in the graph: {', '.join(entity_names[:20])}

Return JSON with:
{{
  "entities": ["list of entity names mentioned (use exact names from available entities)"],
  "relationships": ["list of relationship types like manages, requires, possesses, etc."],
  "query_type": "single_hop|multi_hop|aggregation",
  "sparql_like": "suggested graph traversal pattern"
}}

Important: Match entity names exactly as they appear in the graph (e.g., "Alice" should map to entities with name containing "Alice")."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a Knowledge Graph query analyzer. Extract entities and relationships. Match entity names to available entities in the graph."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        analysis = json.loads(response.choices[0].message.content)
        
        # Step 2: Execute graph query based on analysis
        entities = analysis.get("entities", [])
        relationships = analysis.get("relationships", [])
        query_type = analysis.get("query_type", "single_hop")
        query_lower = query.lower()
        
        results = []
        
        # Detect reverse queries (e.g., "Who has Python skill?" = find who possesses Skill:Python)
        is_reverse_query = "who" in query_lower or ("what" in query_lower and "who" not in query_lower)
        skill_mentioned = any("python" in e.lower() for e in entities) or "python" in query_lower or "skill" in query_lower
        
        # Handle reverse queries first (e.g., "Who has Python skill?")
        if is_reverse_query and skill_mentioned:
            # Find skill entity
            skill_entity = None
            if "python" in query_lower:
                skill_entity = "Skill:Python"
            else:
                # Try to find skill from entities
                for e in entities:
                    if "python" in e.lower():
                        skill_entity = find_entity_by_name(graph_db, "Python")
                        if not skill_entity:
                            skill_entity = "Skill:Python"
                        break
            
            if skill_entity:
                # Reverse query: find who possesses this skill
                possessors = graph_db.query(predicate="possesses", obj=skill_entity)
                for s, p, o in possessors:
                    results.append({
                        "path": [s, skill_entity],
                        "predicates": [p],
                        "end_entity": s
                    })
        
        # Handle multi-hop reverse queries (e.g., "Who manages projects that require Python?")
        elif is_reverse_query and "manages" in query_lower and "require" in query_lower and skill_mentioned:
            # Step 1: Find projects that require Python
            python_skill = "Skill:Python"
            projects_with_python = graph_db.query(predicate="requires", obj=python_skill)
            
            # Step 2: Find who manages those projects
            for s, p, project in projects_with_python:
                managers = graph_db.query(predicate="manages", obj=project)
                for manager_s, manager_p, manager_o in managers:
                    results.append({
                        "path": [manager_s, project, python_skill],
                        "predicates": ["manages", "requires"],
                        "end_entity": manager_s
                    })
        
        # Handle forward queries (starting from an entity)
        elif entities:
            start_entity = find_entity_by_name(graph_db, entities[0])
            if start_entity:
                # Check if query needs multi-hop (e.g., "skills needed for project X manages")
                needs_multi_hop = (
                    ("skill" in query_lower or "require" in query_lower) and 
                    ("manage" in query_lower or "project" in query_lower)
                ) or (query_type == "multi_hop" and len(relationships) >= 2)
                
                if needs_multi_hop and len(relationships) >= 2:
                    # Multi-hop query: entity -> manages -> project -> requires -> skills
                    path = ["manages", "requires"]
                    multi_hop_results = graph_db.multi_hop_query(start_entity, path, max_hops=3)
                    results.extend(multi_hop_results)
                    
                    # Also try explicit path: entity -> manages -> project, then project -> requires -> skill
                    managed_projects = graph_db.query(subject=start_entity, predicate="manages")
                    for s, p, o in managed_projects:
                        project = o
                        required_skills = graph_db.query(subject=project, predicate="requires")
                        for s2, p2, o2 in required_skills:
                            results.append({
                                "path": [start_entity, project, o2],
                                "predicates": ["manages", "requires"],
                                "end_entity": o2
                            })
                elif relationships:
                    # Single-hop query with specific relationship
                    matches = graph_db.query(subject=start_entity, predicate=relationships[0])
                    results.extend([{"path": [start_entity, o], "predicates": [p], "end_entity": o} for s, p, o in matches])
                    
                    # Also try reverse direction
                    reverse_matches = graph_db.query(predicate=relationships[0], obj=start_entity)
                    results.extend([{"path": [s, start_entity], "predicates": [p], "end_entity": s} for s, p, o in reverse_matches])
                else:
                    # Get all neighbors
                    neighbors = graph_db.get_entity_neighbors(start_entity)
                    results = [{"path": [start_entity, n["target"]], "predicates": [n["relationship"]], "end_entity": n["target"]} for n in neighbors]
            else:
                # If entity not found, try to find by partial match
                for s, p, o in graph_db.triples:
                    if p == "hasName" and any(e.lower() in o.lower() for e in entities):
                        entity = s
                        if relationships:
                            matches = graph_db.query(subject=entity, predicate=relationships[0])
                            results.extend([{"path": [entity, o], "predicates": [p], "end_entity": o} for s, p, o in matches])
                        else:
                            neighbors = graph_db.get_entity_neighbors(entity)
                            results.extend([{"path": [entity, n["target"]], "predicates": [n["relationship"]], "end_entity": n["target"]} for n in neighbors])
                        break
        
        # Remove duplicates
        seen_paths = set()
        unique_results = []
        for r in results:
            path_key = tuple(r.get("path", []))
            if path_key not in seen_paths:
                seen_paths.add(path_key)
                unique_results.append(r)
        
        return {
            "analysis": analysis,
            "results": unique_results,
            "executed": True
        }
    
    except Exception as e:
        return {
            "analysis": {},
            "results": [],
            "executed": False,
            "error": str(e)
        }

def find_entity_by_name(graph_db: SimpleGraphDB, name: str) -> Optional[str]:
    """Find entity IRI by name (fuzzy matching)."""
    # Search for entities with matching name
    name_lower = name.lower()
    for s, p, o in graph_db.triples:
        if p == "hasName" and name_lower in o.lower():
            return s
        # Also check if name matches entity ID
        if name_lower in s.lower():
            return s
    return None

def generate_sparql_query(client: OpenAI, natural_query: str, ontology: Dict) -> str:
    """Generate SPARQL-like query from natural language."""
    try:
        ontology_desc = json.dumps(ontology, indent=2)
        
        prompt = f"""Convert this natural language query to a SPARQL-like query pattern.

Natural Language Query: "{natural_query}"

Available Ontology:
{ontology_desc}

Return a SPARQL query pattern (simplified syntax). Example:
SELECT ?result WHERE {{
  ?entity hasName "Alice" .
  ?entity manages ?project .
  ?project requires ?skill .
  ?skill hasName ?result .
}}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a SPARQL query generator. Generate valid SPARQL patterns."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error generating SPARQL: {str(e)}"

def hybrid_rag_kg_query(
    client: OpenAI,
    query: str,
    graph_db: SimpleGraphDB,
    vector_context: List[str] = None
) -> Dict:
    """Hybrid query combining KG structural retrieval with RAG semantic retrieval."""
    
    # Step 1: KG structural retrieval
    kg_results = query_graph_natural_language(client, query, graph_db)
    
    # Step 2: Format KG results
    kg_context = []
    for result in kg_results.get("results", [])[:5]:  # Top 5 results
        path_str = " -> ".join(result.get("path", []))
        kg_context.append(f"Graph Path: {path_str}")
    
    # Step 3: Combine with vector context (if provided)
    all_context = kg_context
    if vector_context:
        all_context.extend([f"Document: {doc}" for doc in vector_context[:3]])
    
    # Step 4: Query LLM with combined context
    context_text = "\n".join(all_context) if all_context else "No context available"
    
    prompt = f"""Answer the user query using the following Knowledge Graph and document context.

Knowledge Graph Context (Structural Facts):
{context_text}

User Query: {query}

Provide a comprehensive answer that combines:
1. Structural facts from the Knowledge Graph
2. Semantic information from documents (if available)
3. Logical reasoning based on relationships"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at combining Knowledge Graph facts with semantic document information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return {
            "kg_results": kg_results,
            "kg_context": kg_context,
            "vector_context": vector_context,
            "combined_response": response.choices[0].message.content.strip()
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "kg_results": kg_results
        }

# ============================================================================
# Main App
# ============================================================================

st.title("🕸️ Knowledge Graphs for Agentic Reasoning")
st.markdown("Demonstrating KG concepts, ontology, multi-hop reasoning, and GraphRAG")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    openai_api_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password"
    )
    
    if st.button("🔌 Connect to OpenAI"):
        if openai_api_key:
            try:
                st.session_state.openai_client = OpenAI(api_key=openai_api_key)
                st.success("✅ Connected to OpenAI")
            except Exception as e:
                st.error(f"❌ Failed to connect: {str(e)}")
        else:
            st.error("❌ Please provide OpenAI API Key")
    
    st.divider()
    st.subheader("Graph Database")
    
    st.info("""
    **Database:** SimpleGraphDB (In-Memory)
    - Lightweight Python implementation
    - Stores triples (Subject-Predicate-Object)
    - No external setup required
    - Perfect for demos and prototyping
    """)
    
    if st.button("🔄 Initialize Sample Graph"):
        populate_sample_graph(st.session_state.graph_db)
        st.success("✅ Sample graph populated")
        st.rerun()
    
    if st.button("🗑️ Clear Graph"):
        st.session_state.graph_db = SimpleGraphDB()
        st.success("✅ Graph cleared")
        st.rerun()
    
    # Graph stats
    stats = st.session_state.graph_db.get_stats()
    st.metric("Triples", stats["total_triples"])
    st.metric("Entities", stats["total_entities"])
    st.metric("Predicates", stats["unique_predicates"])
    
    # Quick links to other tabs
    st.divider()
    st.subheader("🔗 Quick Links")
    st.markdown("""
    - **Build Graph**: Add triples manually
    - **Query Graph**: Natural language queries
    - **Multi-Hop**: Complex reasoning
    - **GraphRAG**: Hybrid KG + RAG
    - **Lab**: Hands-on exercises
    """)

# Main Content Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📚 Concepts",
    "🏗️ Build Graph",
    "🔍 Query Graph",
    "🔄 Multi-Hop Reasoning",
    "🔗 GraphRAG Hybrid",
    "🧪 Hands-on Lab"
])

# Tab 1: Concepts
with tab1:
    st.header("Knowledge Graph Fundamentals")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("When RAG is Not Enough")
        st.markdown("""
        **RAG Limitations:**
        - ❌ Connectivity Gap: Can't connect dots across documents
        - ❌ Multi-Hop Failures: Struggles with indirect relationships
        - ❌ Exact Logic: "Close enough" not acceptable in regulated industries
        
        **KG Advantages:**
        - ✅ Structural Relationships: Explicit connections
        - ✅ Multi-Hop Reasoning: Traverse multiple relationships
        - ✅ Deterministic Logic: Exact, verifiable facts
        """)
        
        st.subheader("Knowledge Graph vs RAG")
        st.markdown("""
        | Feature | RAG | Knowledge Graph |
        |---------|-----|-----------------|
        | Retrieval | Semantic similarity | Structural traversal |
        | Accuracy | "Vibe check" | Source of truth |
        | Multi-hop | Limited | Excellent |
        | Logic | Statistical | Deterministic |
        """)
    
    with col2:
        st.subheader("The Triple (S-P-O)")
        st.markdown("""
        **Basic Unit of Knowledge:**
        - **Subject**: The entity (node)
        - **Predicate**: The relationship (edge)
        - **Object**: Target entity or literal value
        
        **Example:**
        ```
        [Alice] --(manages)--> [Project_X]
        Subject: Alice
        Predicate: manages
        Object: Project_X
        ```
        """)
        
        st.subheader("Ontology (The Schema)")
        st.markdown("""
        **Purpose:**
        - Defines classes of entities
        - Defines allowed relationships
        - Enforces logical constraints
        - Enables automated reasoning
        
        **Example:**
        - Classes: Employee, Project, Skill
        - Properties: manages, requires, possesses
        - Rules: Inverse, Transitive
        """)
    
    st.divider()
    
    st.subheader("Comparison: KG vs Metadata Graph vs Ontology")
    
    comparison_df = pd.DataFrame({
        "Feature": ["Purpose", "Example", "Reasoning"],
        "Ontology": [
            "Defines rules and logic",
            "Every 'Employee' must have an 'ID'",
            "Deductive logic"
        ],
        "Metadata Graph": [
            "Organizes files and attributes",
            "File_X was created by User_Y",
            "Structural (Where is the file?)"
        ],
        "Knowledge Graph": [
            "Maps real-world entities and facts",
            "Employee_101 manages Project_Alpha",
            "Inductive/Relational (Connectivity)"
        ]
    })
    st.dataframe(comparison_df, use_container_width=True)
    
    st.divider()
    
    st.subheader("Sample Ontology: Corporate Project Management")
    
    if st.button("📋 View Sample Ontology"):
        ontology = create_sample_ontology()
        
        st.markdown("**Classes (Entity Types):**")
        classes_df = pd.DataFrame([
            {"Class": cls, "Description": desc}
            for cls, desc in ontology["classes"].items()
        ])
        st.dataframe(classes_df, use_container_width=True)
        
        st.markdown("**Object Properties (Relationships):**")
        props_list = []
        for prop, (domain, range_type) in ontology["object_properties"].items():
            props_list.append({
                "Property": prop,
                "Domain": domain,
                "Range": range_type
            })
        props_df = pd.DataFrame(props_list)
        st.dataframe(props_df, use_container_width=True)
        
        st.markdown("**Data Properties (Attributes):**")
        data_props_list = []
        for prop, (domain, data_type) in ontology["data_properties"].items():
            data_props_list.append({
                "Property": prop,
                "Domain": domain,
                "Data Type": data_type
            })
        data_props_df = pd.DataFrame(data_props_list)
        st.dataframe(data_props_df, use_container_width=True)
        
        st.markdown("**Rules:**")
        st.json(ontology["rules"])

# Tab 2: Build Graph
with tab2:
    st.header("Build Knowledge Graph")
    
    st.subheader("Current Graph State")
    stats = st.session_state.graph_db.get_stats()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Triples", stats["total_triples"])
    with col2:
        st.metric("Entities", stats["total_entities"])
    with col3:
        st.metric("Predicates", stats["unique_predicates"])
    with col4:
        st.metric("Classes", stats["classes"])
    
    st.divider()
    
    st.subheader("Add Triples Manually")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        subject = st.text_input("Subject (Entity)", placeholder="Employee:Alice", key="add_subject")
    with col2:
        predicate = st.text_input("Predicate (Relationship)", placeholder="manages", key="add_predicate")
    with col3:
        obj = st.text_input("Object (Target)", placeholder="Project:Payment_Service", key="add_object")
    
    if st.button("➕ Add Triple"):
        if subject and predicate and obj:
            st.session_state.graph_db.add_triple(subject, predicate, obj)
            st.success(f"✅ Added triple: ({subject}, {predicate}, {obj})")
            st.rerun()
        else:
            st.error("❌ Please fill all fields")
    
    st.divider()
    
    st.subheader("View All Triples")
    if st.session_state.graph_db.triples:
        triples_df = pd.DataFrame(
            st.session_state.graph_db.triples,
            columns=["Subject", "Predicate", "Object"]
        )
        st.dataframe(triples_df, use_container_width=True, height=400)
    else:
        st.info("ℹ️ No triples in graph. Add triples above or initialize sample graph.")
    
    st.divider()
    
    st.subheader("Graph Visualization")
    
    # Always generate fresh graph from current state
    try:
        G = st.session_state.graph_db.to_networkx()
        
        if len(G.nodes()) == 0:
            st.warning("⚠️ Graph is empty. Please add triples or initialize sample graph.")
        else:
            st.success(f"✅ Graph has {len(G.nodes())} nodes and {len(G.edges())} edges")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📋 Show Edge Table"):
                    if len(G.edges()) > 0:
                        edges = [(u, v, G[u][v].get('label', G[u][v].get('relationship', ''))) for u, v in G.edges()]
                        edges_df = pd.DataFrame(edges, columns=["From", "To", "Relationship"])
                        st.dataframe(edges_df, use_container_width=True)
                    else:
                        st.info("ℹ️ No edges in graph")
            
            with col2:
                st.metric("Nodes", len(G.nodes()))
                st.metric("Edges", len(G.edges()))
    except Exception as e:
        st.error(f"❌ Graph generation failed: {str(e)}")
        G = None
    
    # Display graph visualization
    if G is not None and isinstance(G, nx.DiGraph) and len(G.nodes()) > 0:
        if PLOTLY_AVAILABLE:
            # Create plotly network graph
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            edge_trace = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace.append(go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=2, color='#888'),
                    hoverinfo='none',
                    showlegend=False
                ))
            
            node_trace = go.Scatter(
                x=[pos[node][0] for node in G.nodes()],
                y=[pos[node][1] for node in G.nodes()],
                mode='markers+text',
                text=[node[:20] + '...' if len(node) > 20 else node for node in G.nodes()],
                textposition="middle center",
                hovertext=[node for node in G.nodes()],
                marker=dict(
                    size=20,
                    color='lightblue',
                    line=dict(width=2, color='darkblue')
                ),
                showlegend=False
            )
            
            fig = go.Figure(data=edge_trace + [node_trace],
                          layout=go.Layout(
                              title='Knowledge Graph Visualization',
                              titlefont_size=16,
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20, l=5, r=5, t=40),
                              annotations=[dict(
                                  text="Hover over nodes to see entity names",
                                  showarrow=False,
                                  xref="paper", yref="paper",
                                  x=0.005, y=-0.002,
                                  xanchor="left", yanchor="bottom",
                                  font=dict(color="gray", size=10)
                              )],
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                          ))
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback: Show networkx info
            st.info("💡 Install plotly for interactive graph visualization: `pip install plotly`")
            if isinstance(G, nx.DiGraph):
                st.markdown("**Graph Structure:**")
                nodes_list = list(G.nodes())[:10]
                edges_list = list(G.edges())[:10]
                st.code(f"Nodes: {nodes_list}{'...' if len(G.nodes()) > 10 else ''}")
                st.code(f"Edges: {edges_list}{'...' if len(G.edges()) > 10 else ''}")
            else:
                st.warning("⚠️ Graph object is not a NetworkX graph")

# Tab 3: Query Graph
with tab3:
    st.header("Query Knowledge Graph")
    
    if not st.session_state.openai_client:
        st.warning("⚠️ Please connect to OpenAI in the sidebar first")
    else:
        st.subheader("Natural Language Query")
        
        # Example queries with pre-populated results
        example_queries = [
            "What skills are needed for the project Alice manages?",
            "Who manages projects that require Python?",
            "What projects does Alice manage?",
            "Who has Python skill?",
            "What is the budget of Payment Service?"
        ]
        
        selected_example = st.selectbox(
            "Or select an example query:",
            ["Custom Query"] + example_queries,
            key="query_example_selector"
        )
        
        if selected_example != "Custom Query":
            query = selected_example
        else:
            query = st.text_input(
                "Enter your query",
                value="What skills are needed for the project Alice manages?",
                help="Ask questions about entities and relationships in the graph"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Execute Query", type="primary"):
                with st.spinner("Querying graph..."):
                    client = st.session_state.openai_client
                    result = query_graph_natural_language(client, query, st.session_state.graph_db)
                    st.session_state.query_result = result
                    st.session_state.last_query = query
        
        with col2:
            if st.button("📝 Generate SPARQL"):
                with st.spinner("Generating SPARQL query..."):
                    client = st.session_state.openai_client
                    ontology = create_sample_ontology()
                    sparql = generate_sparql_query(client, query, ontology)
                    st.session_state.sparql_query = sparql
        
        # Show example results if graph is populated
        if st.session_state.graph_db.get_stats()["total_triples"] > 0:
            with st.expander("📋 Example Query Results (Pre-computed)"):
                st.markdown("**Try these queries to see results:**")
                for i, eq in enumerate(example_queries[:3], 1):
                    st.markdown(f"{i}. **{eq}**")
                    # Pre-compute example results
                    if "Employee:Alice" in [s for s, _, _ in st.session_state.graph_db.triples]:
                        if "manages" in eq.lower() and "alice" in eq.lower() and "skill" in eq.lower():
                            # Multi-hop: Alice manages Project -> Project requires Skills
                            alice_projects = st.session_state.graph_db.query(subject="Employee:Alice", predicate="manages")
                            all_skills = []
                            for s, p, project in alice_projects:
                                skills = st.session_state.graph_db.query(subject=project, predicate="requires")
                                for s2, p2, skill in skills:
                                    if skill not in all_skills:
                                        all_skills.append(skill)
                            if all_skills:
                                st.code(f"Expected Results: {', '.join(all_skills)}")
                                st.caption("💡 This is a multi-hop query: Alice → manages → Project → requires → Skills")
                        elif "manages" in eq.lower() and "alice" in eq.lower():
                            # Single-hop: What projects does Alice manage?
                            alice_projects = st.session_state.graph_db.query(subject="Employee:Alice", predicate="manages")
                            if alice_projects:
                                project_list = [o for _, _, o in alice_projects]
                                st.code(f"Expected Results: {', '.join(project_list)}")
                        elif "skill" in eq.lower() and "python" in eq.lower():
                            # Who has Python skill?
                            python_holders = st.session_state.graph_db.query(predicate="possesses", obj="Skill:Python")
                            if python_holders:
                                holders = [s for s, _, _ in python_holders]
                                st.code(f"Expected Results: {', '.join(holders)}")
                                st.caption("💡 Reverse query: Finding who possesses Skill:Python")
                        elif "manages" in eq.lower() and "require" in eq.lower() and "python" in eq.lower():
                            # Who manages projects that require Python?
                            # Step 1: Find projects requiring Python
                            projects_with_python = st.session_state.graph_db.query(predicate="requires", obj="Skill:Python")
                            managers_list = []
                            for s, p, project in projects_with_python:
                                # Step 2: Find who manages those projects
                                managers = st.session_state.graph_db.query(predicate="manages", obj=project)
                                for manager_s, manager_p, manager_o in managers:
                                    if manager_s not in managers_list:
                                        managers_list.append(manager_s)
                            if managers_list:
                                st.code(f"Expected Results: {', '.join(managers_list)}")
                                st.caption("💡 Multi-hop reverse query: Projects requiring Python → Who manages them")
        
        # Display query analysis
        if "query_result" in st.session_state:
            result = st.session_state.query_result
            
            st.divider()
            st.subheader("Query Analysis")
            if result.get("analysis"):
                analysis = result["analysis"]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Entities Found:**")
                    st.write(analysis.get("entities", []))
                with col2:
                    st.markdown("**Relationships:**")
                    st.write(analysis.get("relationships", []))
                with col3:
                    st.markdown("**Query Type:**")
                    st.write(analysis.get("query_type", "unknown"))
            
            st.divider()
            st.subheader("Query Results")
            if result.get("results"):
                results_df = pd.DataFrame([
                    {
                        "Path": " -> ".join(r.get("path", [])),
                        "Predicates": ", ".join(r.get("predicates", [])),
                        "End Entity": r.get("end_entity", "")
                    }
                    for r in result["results"]
                ])
                st.dataframe(results_df, use_container_width=True)
                
                # Store results for use in other tabs
                st.session_state.last_query_results = result["results"]
                
                # Link to other tabs
                st.info("💡 **Next Steps:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Multi-Hop Reasoning** → Use these results as starting points")
                with col2:
                    st.markdown("**GraphRAG** → Combine with document context")
                with col3:
                    st.markdown("**Visualization** → See graph structure in Build Graph tab")
            else:
                if result.get("error"):
                    st.error(f"❌ Error: {result.get('error')}")
                else:
                    st.info("ℹ️ No results found. Try a different query or ensure graph is populated.")
                    st.markdown("**Tips:**")
                    st.markdown("- Make sure graph is initialized (sidebar)")
                    st.markdown("- Use entity names from the graph (e.g., 'Alice', 'Payment Service')")
                    st.markdown("- Try simpler queries first (e.g., 'What projects does Alice manage?')")
        
        # Display SPARQL
        if "sparql_query" in st.session_state:
            st.divider()
            st.subheader("Generated SPARQL Query")
            st.code(st.session_state.sparql_query, language="sparql")
        
        st.divider()
        
        st.subheader("Manual Triple Query")
        col1, col2, col3 = st.columns(3)
        with col1:
            query_subject = st.text_input("Subject", placeholder="Employee:Alice", key="query_subject")
        with col2:
            query_predicate = st.text_input("Predicate", placeholder="manages", key="query_predicate")
        with col3:
            query_object = st.text_input("Object", placeholder="", key="query_object")
        
        if st.button("🔍 Query Triples"):
            results = st.session_state.graph_db.query(
                subject=query_subject if query_subject else None,
                predicate=query_predicate if query_predicate else None,
                obj=query_object if query_object else None
            )
            
            if results:
                results_df = pd.DataFrame(results, columns=["Subject", "Predicate", "Object"])
                st.dataframe(results_df, use_container_width=True)
            else:
                st.info("ℹ️ No matching triples found")

# Tab 4: Multi-Hop Reasoning
with tab4:
    st.header("Multi-Hop Reasoning")
    
    st.markdown("""
    **Multi-Hop Reasoning** allows traversing multiple relationships to find answers.
    
    **Example:** "Who manages the project that requires Python?"
    - Hop 1: Find projects that require Python
    - Hop 2: Find employees who manage those projects
    """)
    
    st.subheader("Multi-Hop Query Builder")
    
    start_entity = st.text_input(
        "Start Entity",
        value="Employee:Alice",
        help="Starting entity for traversal"
    )
    
    st.markdown("**Traversal Path (Relationships):**")
    path_input = st.text_input(
        "Path (comma-separated)",
        value="manages, requires",
        help="List of predicates to follow, e.g., 'manages, requires'"
    )
    
    max_hops = st.number_input("Max Hops", min_value=1, max_value=5, value=3)
    
    if st.button("🔄 Execute Multi-Hop Query", type="primary"):
        if start_entity and path_input:
            path = [p.strip() for p in path_input.split(",")]
            results = st.session_state.graph_db.multi_hop_query(start_entity, path, max_hops=max_hops)
            
            st.session_state.multi_hop_results = results
    
    if "multi_hop_results" in st.session_state:
        st.divider()
        st.subheader("Multi-Hop Results")
        
        if st.session_state.multi_hop_results:
            results_df = pd.DataFrame([
                {
                    "Traversal Path": " -> ".join(r.get("path", [])),
                    "Relationships": " -> ".join(r.get("predicates", [])),
                    "End Entity": r.get("end_entity", "")
                }
                for r in st.session_state.multi_hop_results
            ])
            st.dataframe(results_df, use_container_width=True)
            
            # Show reasoning path
            st.subheader("Reasoning Path Explanation")
            for i, result in enumerate(st.session_state.multi_hop_results[:3], 1):
                path = result.get("path", [])
                predicates = result.get("predicates", [])
                st.markdown(f"**Path {i}:**")
                path_str = " → ".join([f"{path[j]} --[{predicates[j] if j < len(predicates) else '?'}]--> {path[j+1]}" for j in range(len(path)-1)])
                st.code(path_str)
        else:
            st.info("ℹ️ No paths found. Check entity names and relationships.")
    
    st.divider()
    
    st.subheader("Example Multi-Hop Queries")
    
    example_queries = [
        {
            "query": "Who should lead Project Y if it requires Python and Cloud Architecture?",
            "start": "Skill:Python",
            "path": "possesses, manages"
        },
        {
            "query": "Find all dependencies of Payment Service",
            "start": "Project:Payment_Service",
            "path": "dependsOn"
        },
        {
            "query": "What projects are part of the Digital Transformation Program?",
            "start": "Program:Digital_Transformation",
            "path": "isPartOf"
        }
    ]
    
    for i, example in enumerate(example_queries):
        with st.expander(f"Example {i+1}: {example['query']}"):
            st.markdown(f"**Start Entity:** {example['start']}")
            st.markdown(f"**Path:** {example['path']}")
            if st.button(f"Run Example {i+1}", key=f"example_{i}"):
                path = [p.strip() for p in example['path'].split(",")]
                results = st.session_state.graph_db.multi_hop_query(example['start'], path, max_hops=3)
                if results:
                    st.dataframe(pd.DataFrame([
                        {"Path": " -> ".join(r.get("path", [])), "End": r.get("end_entity", "")}
                        for r in results
                    ]))

# Tab 5: GraphRAG Hybrid
with tab5:
    st.header("GraphRAG: Hybrid KG + RAG Architecture")
    
    if not st.session_state.openai_client:
        st.warning("⚠️ Please connect to OpenAI in the sidebar first")
    else:
        st.markdown("""
        **GraphRAG combines:**
        - **Semantic Retrieval (RAG)**: Finds broad context and similar situations
        - **Structural Retrieval (KG)**: Finds exact relationships and multi-step connections
        - **Synthesis**: LLM combines both for accurate, contextual responses
        """)
        
        st.subheader("Hybrid Query")
        
        hybrid_query = st.text_input(
            "Enter your query",
            value="Who should lead a new project that requires Python and Cloud Architecture?",
            key="hybrid_query"
        )
        
        st.markdown("**Optional: Add Vector/RAG Context**")
        vector_context = st.text_area(
            "Vector Context (one per line)",
            value="Project Y is a critical initiative for Q2 2024.\nIt requires advanced Python skills and cloud expertise.",
            help="Simulated RAG/document context",
            height=100
        )
        
        if st.button("🔗 Execute GraphRAG Query", type="primary"):
            with st.spinner("Executing hybrid query..."):
                client = st.session_state.openai_client
                vector_list = [line.strip() for line in vector_context.split("\n") if line.strip()]
                
                result = hybrid_rag_kg_query(
                    client,
                    hybrid_query,
                    st.session_state.graph_db,
                    vector_context=vector_list if vector_list else None
                )
                
                st.session_state.hybrid_result = result
        
        if "hybrid_result" in st.session_state:
            result = st.session_state.hybrid_result
            
            if "error" not in result:
                st.divider()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Knowledge Graph Results")
                    if result.get("kg_context"):
                        st.markdown("**Structural Facts:**")
                        for ctx in result["kg_context"]:
                            st.markdown(f"- {ctx}")
                    else:
                        st.info("No KG results")
                
                with col2:
                    st.subheader("📄 Vector/RAG Context")
                    if result.get("vector_context"):
                        st.markdown("**Semantic Information:**")
                        for ctx in result["vector_context"]:
                            st.markdown(f"- {ctx}")
                    else:
                        st.info("No vector context provided")
                
                st.divider()
                st.subheader("🎯 Combined Response")
                st.markdown(result.get("combined_response", "No response generated"))
                
                # Comparison
                st.divider()
                st.subheader("Comparison: KG Only vs GraphRAG")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### KG Only (Structural)")
                    st.info("""
                    **Strengths:**
                    - Exact relationships
                    - Multi-hop reasoning
                    - Deterministic facts
                    
                    **Limitations:**
                    - No semantic context
                    - Limited to graph data
                    """)
                
                with col2:
                    st.markdown("### GraphRAG (Hybrid)")
                    st.success("""
                    **Strengths:**
                    - Combines structure + semantics
                    - Richer context
                    - Better for complex queries
                    
                    **Best For:**
                    - Enterprise decision-making
                    - Complex reasoning tasks
                    """)
            else:
                st.error(f"❌ Error: {result.get('error')}")

# Tab 6: Hands-on Lab
with tab6:
    st.header("🧪 Hands-on Lab: Building and Querying Knowledge Graph")
    
    st.subheader("Lab 1: Build the Graph")
    
    st.markdown("""
    **Objective:** Create a Knowledge Graph with entities and relationships.
    
    **Steps:**
    1. Define Classes: Person, Project, Skill
    2. Insert Triples
    3. Query the graph
    """)
    
    st.info("💡 **Graph Database Used:** Simple In-Memory Graph Database (SimpleGraphDB) - A lightweight Python implementation that stores triples (Subject-Predicate-Object) in memory. No external database setup required!")
    
    with st.expander("📋 Lab Instructions"):
        st.markdown("""
        **Exercise A: Build the Graph**
        
        1. Add these triples:
        - [John] --(Has_Skill)--> [Python]
        - [John] --(Leads)--> [Project_X]
        - [Project_X] --(Requires)--> [Cloud_Architecture]
        
        2. Add more entities:
        - [Alice] --(Has_Skill)--> [Python]
        - [Alice] --(Has_Skill)--> [Cloud_Architecture]
        - [Alice] --(Leads)--> [Project_Y]
        
        **Exercise B: Query the Graph**
        
        Try these queries:
        - "Who has Python skill?"
        - "What skills does Project_X require?"
        - "Who leads projects that require Cloud Architecture?"
        """)
    
    st.divider()
    
    st.subheader("Interactive Lab: Add Triples")
    
    st.markdown("**Step 1: Add the following triples using the form below or copy-paste:**")
    
    lab_triples = [
        ("Employee:John", "possesses", "Skill:Python"),
        ("Employee:John", "manages", "Project:Project_X"),
        ("Project:Project_X", "requires", "Skill:Cloud_Architecture"),
        ("Employee:Alice", "possesses", "Skill:Python"),
        ("Employee:Alice", "possesses", "Skill:Cloud_Architecture"),
        ("Employee:Alice", "manages", "Project:Project_Y")
    ]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Triples to Add:**")
        for i, (s, p, o) in enumerate(lab_triples, 1):
            st.code(f"{i}. ({s}, {p}, {o})", language=None)
    
    with col2:
        st.markdown("**Quick Add:**")
        if st.button("➕ Add All Lab Triples"):
            for s, p, o in lab_triples:
                st.session_state.graph_db.add_triple(s, p, o)
            st.success(f"✅ Added {len(lab_triples)} triples!")
            st.rerun()
    
    st.divider()
    
    st.subheader("Step 2: Verify Your Graph")
    
    if st.button("🔍 Check Lab Graph"):
        stats = st.session_state.graph_db.get_stats()
        st.metric("Total Triples", stats["total_triples"])
        
        # Check if lab triples exist
        found_triples = []
        for s, p, o in lab_triples:
            if (s, p, o) in st.session_state.graph_db.triples:
                found_triples.append((s, p, o))
        
        if len(found_triples) == len(lab_triples):
            st.success(f"✅ All {len(lab_triples)} lab triples found in graph!")
        else:
            st.warning(f"⚠️ Only {len(found_triples)}/{len(lab_triples)} lab triples found. Add the missing ones above.")
        
        # Show current triples
        if st.session_state.graph_db.triples:
            st.markdown("**Current Triples in Graph:**")
            triples_df = pd.DataFrame(
                st.session_state.graph_db.triples,
                columns=["Subject", "Predicate", "Object"]
            )
            st.dataframe(triples_df, use_container_width=True, height=300)
    
    st.divider()
    
    st.subheader("Step 3: Test Queries")
    
    st.markdown("**Now try these queries in the 'Query Graph' tab:**")
    test_queries = [
        "Who has Python skill?",
        "What skills does Project_X require?",
        "Who manages projects that require Cloud Architecture?"
    ]
    
    for i, tq in enumerate(test_queries, 1):
        st.markdown(f"{i}. **{tq}**")
        if st.button(f"🔍 Run Query {i}", key=f"lab_query_{i}"):
            st.session_state.lab_query = tq
            st.info(f"💡 Go to 'Query Graph' tab and execute: {tq}")
    
    st.divider()
    
    st.subheader("Lab 2: Agent Reasoning")
    
    st.markdown("""
    **Objective:** Use KG for agentic reasoning to answer complex questions.
    
    **Scenario:** "Who should lead Project Y if it requires Python and Cloud Architecture?"
    """)
    
    if not st.session_state.openai_client:
        st.warning("⚠️ Connect to OpenAI to run agent reasoning")
    else:
        reasoning_query = st.text_input(
            "Reasoning Query",
            value="Who should lead Project Y if it requires Python and Cloud Architecture?",
            key="reasoning_query"
        )
        
        if st.button("🤖 Run Agent Reasoning", type="primary"):
            with st.spinner("Agent reasoning..."):
                client = st.session_state.openai_client
                
                # Step 1: Find entities with required skills
                python_skill_holders = []
                cloud_skill_holders = []
                
                for s, p, o in st.session_state.graph_db.triples:
                    if p == "possesses" and o == "Skill:Python":
                        python_skill_holders.append(s)
                    if p == "possesses" and o == "Skill:Cloud_Architecture":
                        cloud_skill_holders.append(s)
                
                # Step 2: Find intersection (people with both skills)
                candidates = set(python_skill_holders) & set(cloud_skill_holders)
                
                # Step 3: Check leadership experience
                experienced_leaders = []
                for candidate in candidates:
                    # Check if they manage any project
                    manages = st.session_state.graph_db.query(subject=candidate, predicate="manages")
                    if manages:
                        experienced_leaders.append(candidate)
                
                # Step 4: Generate reasoning
                reasoning_steps = []
                reasoning_steps.append(f"**Step 1:** Find people with Python skill → {', '.join(python_skill_holders) if python_skill_holders else 'None'}")
                reasoning_steps.append(f"**Step 2:** Find people with Cloud Architecture skill → {', '.join(cloud_skill_holders) if cloud_skill_holders else 'None'}")
                reasoning_steps.append(f"**Step 3:** Intersection (both skills) → {', '.join(candidates) if candidates else 'None'}")
                reasoning_steps.append(f"**Step 4:** Check leadership experience → {', '.join(experienced_leaders) if experienced_leaders else 'None'}")
                
                # Step 5: Final answer using LLM
                context = "\n".join(reasoning_steps)
                prompt = f"""Based on the following reasoning steps from a Knowledge Graph, provide a final answer.

Reasoning Steps:
{context}

Question: {reasoning_query}

Provide a clear answer with reasoning."""

                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are an agentic reasoning assistant that uses Knowledge Graph facts."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3
                    )
                    
                    final_answer = response.choices[0].message.content.strip()
                    
                    st.session_state.reasoning_result = {
                        "steps": reasoning_steps,
                        "final_answer": final_answer
                    }
                except Exception as e:
                    st.error(f"❌ Reasoning failed: {str(e)}")
        
        if "reasoning_result" in st.session_state:
            result = st.session_state.reasoning_result
            
            st.divider()
            st.subheader("Reasoning Steps")
            for step in result["steps"]:
                st.markdown(step)
            
            st.divider()
            st.subheader("Final Answer")
            st.success(result["final_answer"])
    
    st.divider()
    
    st.subheader("Lab 3: Virtual Mapping Simulation")
    
    st.markdown("""
    **Objective:** Simulate mapping relational data to Knowledge Graph.
    
    **Concept:** Virtual Knowledge Graph (VKG) allows querying SQL databases as if they were a KG.
    """)
    
    st.markdown("**Simulated Relational Data:**")
    
    # Simulate relational tables
    employees_table = pd.DataFrame({
        "ID": [101, 102, 103],
        "Name": ["Alice", "Bob", "Carol"],
        "Manager_ID": [102, 103, None],
        "Department": ["Engineering", "Engineering", "Engineering"]
    })
    
    projects_table = pd.DataFrame({
        "ID": [1, 2, 3],
        "Name": ["Payment Service", "Data Pipeline", "Project X"],
        "Manager_ID": [101, 102, 101],
        "Budget": [500000, 300000, 200000]
    })
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Employees Table:**")
        st.dataframe(employees_table, use_container_width=True)
    with col2:
        st.markdown("**Projects Table:**")
        st.dataframe(projects_table, use_container_width=True)
    
    if st.button("🔄 Map to Knowledge Graph"):
        st.info("💡 In production, this would use R2RML mappings to convert SQL to RDF triples")
        
        # Simulate mapping
        mapped_triples = []
        for _, row in employees_table.iterrows():
            mapped_triples.append((f"Employee:{row['ID']}", "hasName", f'"{row["Name"]}"'))
            if pd.notna(row['Manager_ID']):
                mapped_triples.append((f"Employee:{row['ID']}", "reportsTo", f"Employee:{int(row['Manager_ID'])}"))
        
        for _, row in projects_table.iterrows():
            mapped_triples.append((f"Project:{row['ID']}", "hasName", f'"{row["Name"]}"'))
            if pd.notna(row['Manager_ID']):
                mapped_triples.append((f"Employee:{int(row['Manager_ID'])}", "manages", f"Project:{row['ID']}"))
        
        st.success(f"✅ Mapped {len(mapped_triples)} triples from relational data")
        
        st.markdown("**Generated Triples:**")
        mapped_df = pd.DataFrame(mapped_triples, columns=["Subject", "Predicate", "Object"])
        st.dataframe(mapped_df, use_container_width=True)
        
        st.info("💡 These triples can now be queried using SPARQL instead of complex SQL JOINs")

if __name__ == "__main__":
    pass
