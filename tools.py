from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import ssl
import openai
from typing import List, Dict, Any, Annotated
import streamlit as st

URL = st.secrets["NEO4J_URI"]
USER = st.secrets["NEO4J_USERNAME"]
PASSWORD = st.secrets["NEO4J_PASSWORD"]


openai.api_key = os.environ["OPENAI_API_KEY"]

url = URL
username = USER
password = PASSWORD
os.environ["NEO4J_URI"] = URL
os.environ["NEO4J_USERNAME"] = USER
os.environ["NEO4J_PASSWORD"] = PASSWORD

    # Initialiser le graph
graph = Neo4jGraph(url=url, username=username, password=password)
    # Initialiser les embeddings
embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key = os.getenv("OPENAI_API_KEY") 
    )



query = """
MATCH (node:Fonction {machine_id: 'U7.5C'})-[r]->(m:SousFonction)
WITH node, score, collect({name: m.name, description: m.description, id: m.id}) as sousFunctions
RETURN node.name as text, score, {name: node.name, description: node.description, id: node.id, sousFunctions: sousFunctions} as metadata
"""


function_vector = Neo4jVector.from_existing_graph(embedding=embeddings, retrieval_query=query, index_name="function", node_label='Fonction', embedding_node_property='embedding',text_node_properties=["name", "description"] )

def search_function(search_term : Annotated[str, "The function to search for. "]):
    """
    Perform similarity search for a Fonction using the global Neo4jVector instance.

    Args:
        search_term (str): The term to search for.

    Returns:
        list: A list of dictionaries containing the Fonction details.
    """
    # Perform the similarity search
    results = function_vector.similarity_search_with_score(search_term)
    output = []

    # Iterate over results to extract details
    for document, score in results:
        metadata = document.metadata
        fonction_id = metadata.get('id', 'No Fonction ID available')  # Extract Fonction ID
        description = metadata.get('description', 'No description available')
        name = metadata.get('name', 'No name available')
        sous_functions = metadata.get('sousFunctions', [])

        # Format each result
        result = {
            "Fonction": fonction_id,
            "Name": name,
            "Score": round(score, 2),
            "Description": description,
            "SousFunctions": [
                {
                    "Name": sous_function.get('name', 'No SousFunction name'),
                    "Description": sous_function.get('description', 'No SousFunction description'),
                    "ID": sous_function.get('id', 'No SousFunction ID')
                }
                for sous_function in sous_functions
            ]
        }
        output.append(result)

    return output


query = """
MATCH (node:SousFonction)-[r]->(c:Composant)
WHERE node.machine_id = 'U7.5C'
WITH node, collect({reference: c.reference, description: c.description}) as composants, score
RETURN node.name as text, 
        score, 
       {name: node.name, description: node.description, id: node.id, composants: composants} as metadata
"""

SousFunction_vector = Neo4jVector.from_existing_graph(embedding=embeddings, retrieval_query=query, index_name="sousfunction", node_label='SousFonction', embedding_node_property='embedding', text_node_properties=["name", "description"] )

def search_sous_function(search_term : Annotated[str, "The sous function to search for. "]):
    """
    Perform similarity search for a SousFonction using the global Neo4jVector instance.

    Args:
        search_term (str): The term to search for.

    Returns:
        list: A list of dictionaries containing the SousFonction details.
    """
    # Perform the similarity search
    results = SousFunction_vector.similarity_search_with_score(search_term)
    output = []

    # Iterate over results to extract details
    for document, score in results:
        metadata = document.metadata
        sous_fonction = metadata.get('id', 'No SousFonction available')  # Extract SousFonction
        description = metadata.get('description', 'No description available')
        name = metadata.get('name', 'No name available')
        composants = metadata.get('composants', [])

        # Format each result
        result = {
            "SousFonction": sous_fonction,
            "Name": name,
            "Score": round(score, 2),
            "Description": description,
            "Composants": [
                {
                    "Description": component.get('description', 'No component description'),
                    "Reference": component.get('reference', 'No reference')
                }
                for component in composants
            ]
        }
        output.append(result)

    return output




query = """
MATCH (node:Composant)
WHERE node.machine_id = 'U7.5C'
 WITH node, score
RETURN node.name as text, 
       score, 
       {name: node.name, description: node.description, id: node.id, type : node.type, familly : node.familly} as metadata
"""


Component_vector = Neo4jVector.from_existing_graph(embedding=embeddings, retrieval_query=query, index_name="composant", node_label='Composant', embedding_node_property='embedding', text_node_properties=["famille", "name", "type", "description", "reference"] )

def search_component(search_term : Annotated[str, "The component, the familly, type, description to search for."]):
    """
    Perform similarity search for a Composant using the global Neo4jVector instance.

    Args:
        search_term (str): The term to search for.

    Returns:
        list: A list of dictionaries containing the Composant details.
    """
    # Perform the similarity search
    results = Component_vector.similarity_search_with_score(search_term)
    output = []

    # Iterate over results to extract details
    for document, score in results:
        metadata = document.metadata
        component_id = metadata.get('id', 'No Component ID available')  # Extract Composant ID
        description = metadata.get('description', 'No description available')
        name = metadata.get('name', 'No name available')
        component_type = metadata.get('type', 'No type available')
        family = metadata.get('familly', 'No family available')

        # Format each result
        result = {
            "Composant": component_id,
            "Name": name,
            "Score": round(score, 2),
            "Description": description,
            "Type": component_type,
            "Family": family
        }
        output.append(result)

    return output






def init_context(machine_id: str) -> str:
    """
    Initializes the context by fetching Machine -> Fonction -> SousFonction relationships
    for a given machine_id and formats the result for LLM context.

    Args:
        machine_id (str): The machine_id to filter the query.

    Returns:
        str: A formatted string representing the context for an LLM.
    """
    # Define the Cypher query
    query = """
    MATCH (m:Machine)-[:HAS_FONCTION]->(f:Fonction)-[:HAS_SOUSFONCTION]->(s:SousFonction)
    WHERE m.id = $machine_id
    RETURN m.name AS MachineName,
           m.description AS MachineDescription,
           f.name AS FonctionName,
           f.id AS FonctionID,
           f.description AS FonctionDescription,
           collect({
               id: s.id,
               name: s.name,
               description: s.description
           }) AS SousFonctions
    """
    # Execute the query
    try:
        results = graph.query(query, params={"machine_id": machine_id})
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred while fetching the context."

    # Format the results for LLM context
    context = []
    machine_name = ""
    machine_description = ""

    for record in results:
        # Fetch machine details
        if not machine_name:  # Machine details only need to be added once
            machine_name = record.get("MachineName", "Unknown Machine")
            machine_description = record.get("MachineDescription", "No description available")
            context.append(f"Machine: {machine_name}")
            context.append(f"Description: {machine_description}")
            context.append("")  # Add a blank line after machine details

        # Fetch Fonction and SousFonction details
        fonction_name = record.get("FonctionName", "Unknown Fonction")
        fonction_description = record.get("FonctionDescription", "No description available")
        sous_fonctions = record.get("SousFonctions", [])

        # Format Fonction
        context.append(f"Fonction: {fonction_name}")
        context.append(f"Description: {fonction_description}")

        # Format SousFonctions
        if sous_fonctions:
            context.append("Related SousFonctions:")
            for sf in sous_fonctions:
                sf_name = sf.get("name", "Unknown SousFonction")
                sf_description = sf.get("description", "No description available")
                context.append(f"- {sf_name}: {sf_description}")
        else:
            context.append("No related SousFonctions found.")

        context.append("")  # Add a blank line between Fonction entries

    # Join the context lines into a single formatted string
    return "\n".join(context)




query_data = """
MATCH (node:Composant)-[r]->(c:Composant)
WHERE node.machine_id = 'U7.5C'
WITH node, collect({reference: c.reference, description: c.description, id: c.id, type: c.type, familly: c.familly}) AS composants, r.weight AS score
RETURN node.name AS text,
       score,
       {
           name: node.name,
           description: node.description,
           id: node.id,
           composants: composants
       } AS metadata
"""




Interco_vector = Neo4jVector.from_existing_graph(embedding=embeddings, retrieval_query=query_data, index_name="interco_composant", node_label='Composant', embedding_node_property='embedding_tenant', text_node_properties=["famille", "name", "type", "description", "reference"] )
def search_component_interconnections(search_term: Annotated[str, "The component to search for interconnections. Return the component interconnections id, name, description, type, family and reference"]) -> List[Dict[str, Any]]:
    """
    Perform similarity search for a Composant and its interconnections using the global Neo4jVector instance.

    Args:
        search_term (str): The term to search for.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the Composant details
                              and its interconnections.
    """
    # Perform the similarity search
    results = Interco_vector.similarity_search_with_score(search_term)
    output = []

    # Iterate over results to extract details
    for document, score in results:
        metadata = document.metadata
        component_id = metadata.get('id', 'No Component ID available')  # Extract Composant ID
        name = metadata.get('name', 'No name available')
        description = metadata.get('description', 'No description available')
        composants = metadata.get('composants', [])

        # Format the interconnection details
        interconnections = [
            {
                "ID": component.get("id", "No Component ID"),
                "Reference": component.get("reference", "No Reference"),
                "Description": component.get("description", "No description available"),
                "Type": component.get("type", "No type available"),
                "Family": component.get("familly", "No family available"),
            }
            for component in composants
        ]

        # Format the result
        result = {
            "Component": {
                "ID": component_id,
                "Name": name,
                "Description": description,
            },
            "Score": round(score, 2),
            "Interconnections": interconnections,
        }
        output.append(result)

    return output