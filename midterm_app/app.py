from __future__ import annotations
import os
import chainlit as cl
import pandas as pd
from typing import List, Dict, Any, TypedDict, Callable, Annotated, Literal, Optional, Union, Tuple, TypeVar
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END
from langchain.tools import Tool
from tavily import TavilyClient
from dotenv import load_dotenv
import json
import asyncio
import time
from functools import wraps
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import Output
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dataclasses import dataclass, field
from state import FounderAnalysisState

# Load environment variables
load_dotenv()

# Validate API keys
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please add it to your .env file.")
if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY not found in environment variables. Please add it to your .env file.")

# Configuration
COLLECTION_NAME = "founders"
VECTOR_DIM = 1536  # OpenAI embedding dimension
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
MAX_RELEVANT_CHUNKS = 3
SIMILARITY_THRESHOLD = 0.75
DEFAULT_TIMEOUT = 60  # Default timeout in seconds
API_RATE_LIMIT_DELAY = 1  # Delay between API calls in seconds

StateType = TypeVar("StateType", bound=Dict[str, Any])

# Decorator for adding timeouts to async functions
def async_timeout(timeout_seconds=DEFAULT_TIMEOUT):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                # Create a meaningful timeout message
                func_name = func.__name__
                await cl.Message(content=f"⏱️ Operation timed out: {func_name} took longer than {timeout_seconds} seconds").send()
                # Return appropriate error state if the function was expecting to return a state
                if "state" in kwargs:
                    return {**kwargs["state"], "error": f"Operation timed out after {timeout_seconds} seconds"}
                raise
        return wrapper
    return decorator

# Rate limiter for API calls
async def rate_limit():
    """Simple rate limiter to prevent API throttling"""
    await asyncio.sleep(API_RATE_LIMIT_DELAY)

class VectorStore:
    def __init__(self):
        self.client = QdrantClient(":memory:")  # In-memory Qdrant instance
        self._create_collection()

    def _create_collection(self):
        """Create the founders collection if it doesn't exist."""
        self.client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
        )

    def upsert_profiles(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]):
        """Upsert founder profiles with their embeddings and metadata."""
        points = [
            PointStruct(
                id=idx,
                vector=embedding,
                payload=metadata[idx]
            )
            for idx, embedding in enumerate(embeddings)
        ]
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

    def search_profiles(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar profiles using the query vector."""
        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit
        )
        return [hit.payload for hit in results]

    def get_profile_by_metadata(self, metadata_key: str, metadata_value: Any) -> List[Dict[str, Any]]:
        """Retrieve profiles based on metadata filtering."""
        from qdrant_client.http import models as rest
        
        filter_condition = rest.Filter(
            must=[
                rest.FieldCondition(
                    key=metadata_key,
                    match=rest.MatchValue(value=metadata_value)
                )
            ]
        )
        
        results = self.client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=filter_condition
        )[0]
        
        return [point.payload for point in results]

class FounderAnalysisSystem:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.vector_store = VectorStore()
        self.llm = ChatOpenAI(model=LLM_MODEL, timeout=DEFAULT_TIMEOUT)
        self.tavily_client = TavilyClient()
        self.workflow = self._create_workflow()
        self.progress_message = None

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for founder analysis."""
        # Use a simple dict type for the state graph
        workflow = StateGraph(dict)
        
        # Add nodes to the graph
        workflow.add_node("process_query", self.process_query)
        workflow.add_node("vector_search", self.vector_search)
        workflow.add_node("filter_by_metadata", self.filter_by_metadata)
        workflow.add_node("web_search", self.web_search)
        workflow.add_node("analyze_profiles", self.analyze_profiles)
        workflow.add_node("format_response", self.format_response)
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "process_query",
            self.query_router,
            {
                "search": "vector_search",
                "filter": "filter_by_metadata",
                "error": END
            }
        )
        
        # Add standard edges
        workflow.add_edge("vector_search", "web_search")
        workflow.add_edge("filter_by_metadata", "web_search")
        workflow.add_edge("web_search", "analyze_profiles")
        workflow.add_edge("analyze_profiles", "format_response")
        workflow.add_edge("format_response", END)
        
        # Set entry point
        workflow.set_entry_point("process_query")
        
        return workflow

    async def update_progress(self, message, step, total_steps):
        """Update the progress message to show the system is still working"""
        progress_text = f"⏳ {message} (Step {step}/{total_steps})"
        if self.progress_message is None:
            self.progress_message = cl.Message(content=progress_text)
            await self.progress_message.send()
        else:
            # Fix: Use update() without content parameter, then set content property
            await self.progress_message.update()
            self.progress_message.content = progress_text

    @async_timeout(30)  # 30 second timeout for query processing
    async def process_query(self, state: FounderAnalysisState) -> FounderAnalysisState:
        """Process the user query and determine the query type."""
        # Initialize state if needed
        if not isinstance(state, dict):
            state = {}
        
        state.update({
            "query": state.get("query", ""),
            "query_type": "",
            "filter_key": "",
            "filter_value": "",
            "retrieved_profiles": [],
            "web_search_results": [],
            "analysis_results": [],
            "final_response": {},
            "error": ""
        })
        
        query = state["query"]
        
        # Log the processing step
        await self.update_progress("Processing your query...", 1, 5)
        
        # Check if it's a filter command
        if query.lower().startswith("filter:") or query.lower().startswith("filter "):
            # Remove the filter prefix and trim whitespace
            filter_text = query.replace("filter:", "").replace("filter ", "").strip()
            
            # Check if there's a colon separator for key:value format
            if ":" in filter_text:
                parts = filter_text.split(":", 1)
                filter_key, filter_value = parts
                
                # Provide a helpful message if the filter value is empty
                if not filter_value.strip():
                    return {
                        **state,
                        "error": f"Please provide a value to filter by. Example: filter:{filter_key}:value"
                    }
                
                return {
                    **state,
                    "query_type": "filter",
                    "filter_key": filter_key.strip(),
                    "filter_value": filter_value.strip()
                }
            else:
                # If no specific key is provided, search across all fields
                filter_value = filter_text
                
                # Provide a helpful message if the filter value is empty
                if not filter_value.strip():
                    return {
                        **state,
                        "error": "Please provide a value to filter by. Example: filter:Location:San Francisco"
                    }
                
                return {
                    **state,
                    "query_type": "filter",
                    "filter_key": "all_fields",  # Special value to indicate searching across all fields
                    "filter_value": filter_value.strip()
                }
        else:
            return {**state, "query_type": "search"}

    def query_router(self, state: FounderAnalysisState) -> str:
        """Route to the appropriate node based on query type."""
        if "error" in state and state["error"]:
            return "error"
        return state["query_type"]

    @async_timeout(45)  # 45 second timeout for vector search
    async def vector_search(self, state: FounderAnalysisState) -> FounderAnalysisState:
        """Search for similar profiles using vector similarity."""
        query = state["query"]
        
        # Log the vector search step
        await self.update_progress("Searching for relevant founder profiles...", 2, 5)
        
        try:
            # Convert query to embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search for similar profiles
            profiles = self.vector_store.search_profiles(query_embedding, limit=3)
            
            if not profiles:
                return {
                    **state,
                    "retrieved_profiles": [],
                    "error": "No matching profiles found."
                }
            
            return {**state, "retrieved_profiles": profiles}
        except Exception as e:
            return {**state, "error": f"Error during vector search: {str(e)}"}

    @async_timeout(45)  # 45 second timeout for metadata filtering
    async def filter_by_metadata(self, state: FounderAnalysisState) -> FounderAnalysisState:
        """Filter profiles by metadata."""
        filter_key = state["filter_key"]
        filter_value = state["filter_value"]
        
        # Log the filtering step
        if filter_key == "all_fields":
            await self.update_progress(f"Searching for '{filter_value}' across all profile fields...", 2, 5)
        else:
            await self.update_progress(f"Filtering profiles by {filter_key}: '{filter_value}'...", 2, 5)
        
        try:
            # Get all profiles first
            from qdrant_client.http import models as rest
            
            # Get all profiles from the collection
            results = self.vector_store.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100  # Adjust this limit based on your expected dataset size
            )[0]
            
            all_profiles = [point.payload for point in results]
            search_value = filter_value.lower()
            
            # Perform flexible filtering in Python
            filtered_profiles = []
            
            # Special case for searching across all fields
            if filter_key == "all_fields":
                for profile in all_profiles:
                    # Search across all fields in the profile
                    for key, value in profile.items():
                        if value and search_value in str(value).lower():
                            filtered_profiles.append(profile)
                            break  # Found a match, move to next profile
            else:
                # Regular field-specific search
                for profile in all_profiles:
                    # Check if the key exists in the profile
                    if filter_key in profile:
                        profile_value = str(profile[filter_key]).lower()
                        
                        # Check for partial match (case-insensitive)
                        if search_value in profile_value:
                            filtered_profiles.append(profile)
            
            if not filtered_profiles:
                if filter_key == "all_fields":
                    error_msg = f"No profiles found matching '{filter_value}' in any field"
                else:
                    error_msg = f"No profiles found matching '{filter_value}' in {filter_key} field"
                
                return {
                    **state,
                    "retrieved_profiles": [],
                    "error": error_msg
                }
            
            return {**state, "retrieved_profiles": filtered_profiles[:3]}  # Limit to 3 profiles
        except Exception as e:
            return {**state, "error": f"Error during metadata filtering: {str(e)}"}

    @async_timeout(90)  # 90 second timeout for web search
    async def web_search(self, state: FounderAnalysisState) -> FounderAnalysisState:
        """Gather additional information from web search."""
        profiles = state["retrieved_profiles"]
        
        if not profiles:
            return {**state, "web_search_results": []}
        
        await self.update_progress("Gathering additional information from web search...", 3, 5)
        
        web_search_results = []
        
        for i, profile in enumerate(profiles):
            name = profile.get("Full Name", "")
            position = profile.get("Current Position", "")
            company = profile.get("Company", "")
            
            # Update progress for each profile
            await self.update_progress(f"Searching web for info about {name} ({i+1}/{len(profiles)})...", 3, 5)
            
            search_query = f"{name} {position} {company}"
            try:
                results = self.tavily_client.search(
                    query=search_query, 
                    search_depth="advanced"
                ).get("results", [])
                
                web_search_results.append({
                    "profile_name": name,
                    "search_results": results
                })
                
                # Rate limit between API calls
                if i < len(profiles) - 1:
                    await rate_limit()
                
            except Exception as e:
                await cl.Message(content=f"⚠️ Error searching for {name}: {str(e)}").send()
        
        return {**state, "web_search_results": web_search_results}

    @async_timeout(120)  # 2 minute timeout for analysis
    async def analyze_profiles(self, state: FounderAnalysisState) -> FounderAnalysisState:
        """Analyze profiles with additional context."""
        profiles = state["retrieved_profiles"]
        web_results = state["web_search_results"]
        
        if not profiles:
            return {**state, "analysis_results": []}
        
        await self.update_progress("Analyzing profiles and generating recommendations...", 4, 5)
        
        analysis_results = []
        
        for i, profile in enumerate(profiles):
            name = profile.get("Full Name", "")
            
            # Find matching web results
            additional_info = []
            for result in web_results:
                if result["profile_name"] == name:
                    additional_info = result["search_results"]
                    break
            
            # Update progress for each profile
            await self.update_progress(f"Analyzing profile for {name} ({i+1}/{len(profiles)})...", 4, 5)
            
            # Extract social media and online presence
            linkedin = profile.get("LinkedIn", "")
            twitter = profile.get("Twitter", "")
            website = profile.get("Website", "")
            
            analysis_prompt = f"""
            Based on the following founder profile and additional information, analyze what types of companies 
            this person would be best suited to found. Consider their experience, skills, background, and online presence.
            
            Profile: {json.dumps(profile, indent=2)}
            Additional Information: {json.dumps(additional_info, indent=2)}
            
            Provide a detailed analysis including:
            1. Recommended industry sectors based on their expertise and background
            2. Type of company (B2B, B2C, etc.) that would align with their experience
            3. Key strengths that would contribute to success as a founder
            4. Potential challenges to consider based on their profile
            5. How their network and online presence could benefit their venture
            6. Specific opportunities or niches they might be well-positioned to address
            
            Be specific and provide actionable insights based on the information available.
            """
            
            try:
                response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
                
                analysis_results.append({
                    "founder_name": name,
                    "analysis": response.content,
                    "profile": profile,
                    "additional_info": additional_info
                })
                
                # Rate limit between API calls
                if i < len(profiles) - 1:
                    await rate_limit()
                
            except Exception as e:
                await cl.Message(content=f"⚠️ Error analyzing {name}: {str(e)}").send()
        
        return {**state, "analysis_results": analysis_results}

    @async_timeout(30)  # 30 second timeout for formatting
    async def format_response(self, state: FounderAnalysisState) -> FounderAnalysisState:
        """Format the final response for display."""
        analysis_results = state["analysis_results"]
        
        await self.update_progress("Formatting final results...", 5, 5)
        
        # Clear the progress message
        self.progress_message = None
        
        if not analysis_results:
            if "error" in state and state["error"]:
                await cl.Message(content=f"❌ {state['error']}").send()
            else:
                await cl.Message(content="❌ No results to display.").send()
            return {**state, "final_response": {"status": "error", "message": state.get("error", "No results")}}
        
        for result in analysis_results:
            founder_name = result["founder_name"]
            profile = result["profile"]
            analysis = result["analysis"]
            
            # Build profile summary with basic information
            profile_summary = f"""
            🎯 Profile Summary:
            
            - Name: {profile.get('Full Name', '')}
            - Current Position: {profile.get('Current Position', '')}
            - Company: {profile.get('Company', '')}
            - Location: {profile.get('Location', '')}
            """
            
            # Add LinkedIn profile with proper URL formatting
            if profile.get('LinkedIn') and profile.get('LinkedIn').strip():
                linkedin_url = profile.get('LinkedIn')
                # Make sure the URL has the proper format
                if not linkedin_url.startswith('http'):
                    linkedin_url = f"https://{linkedin_url}"
                profile_summary += f"- LinkedIn: {linkedin_url}\n"
            
            # Add any other social profiles or websites
            if profile.get('Twitter') and profile.get('Twitter').strip():
                twitter_url = profile.get('Twitter')
                if not twitter_url.startswith('http'):
                    twitter_url = f"https://{twitter_url}"
                profile_summary += f"- Twitter: {twitter_url}\n"
                
            if profile.get('Website') and profile.get('Website').strip():
                website_url = profile.get('Website')
                if not website_url.startswith('http'):
                    website_url = f"https://{website_url}"
                profile_summary += f"- Website: {website_url}\n"
            
            # Format the analysis
            analysis_text = f"""
            📊 Analysis:
            
            {analysis}
            """
            
            # Create elements for structured display using Text instead of Markdown
            elements = [
                cl.Text(content=profile_summary),
                cl.Text(content=analysis_text)
            ]
            
            await cl.Message(
                content=f"Analysis for {founder_name}:",
                elements=elements
            ).send()
        
        await cl.Message(content="✅ Analysis complete!").send()
        
        return {**state, "final_response": {"status": "success", "results": analysis_results}}

    @async_timeout(120)  # 2 minute timeout for loading profiles
    async def load_profiles(self, file):
        """Load and embed founder profiles from uploaded CSV."""
        # Read CSV file
        df = pd.read_csv(file)
        
        # Convert DataFrame rows to list of dictionaries
        profiles = df.to_dict('records')
        
        # Create more comprehensive text representations for embedding
        texts = []
        for p in profiles:
            # Build a rich text representation including all available fields
            text_parts = []
            
            # Add core identity information
            if p.get('Full Name'):
                text_parts.append(f"Name: {p.get('Full Name')}")
            
            if p.get('Current Position'):
                text_parts.append(f"Position: {p.get('Current Position')}")
                
            if p.get('Company'):
                text_parts.append(f"Company: {p.get('Company')}")
                
            if p.get('Location'):
                text_parts.append(f"Location: {p.get('Location')}")
            
            # Add contact and social media information
            if p.get('LinkedIn'):
                text_parts.append(f"LinkedIn: {p.get('LinkedIn')}")
                
            if p.get('Twitter'):
                text_parts.append(f"Twitter: {p.get('Twitter')}")
                
            if p.get('Website'):
                text_parts.append(f"Website: {p.get('Website')}")
                
            if p.get('Email'):
                text_parts.append(f"Email: {p.get('Email')}")
            
            # Add detailed professional information
            if p.get('About'):
                text_parts.append(f"About: {p.get('About')}")
                
            if p.get('Skills'):
                text_parts.append(f"Skills: {p.get('Skills')}")
                
            if p.get('Experience'):
                text_parts.append(f"Experience: {p.get('Experience')}")
                
            if p.get('Education'):
                text_parts.append(f"Education: {p.get('Education')}")
            
            # Add any industry or sector information
            if p.get('Industry'):
                text_parts.append(f"Industry: {p.get('Industry')}")
                
            if p.get('Sector'):
                text_parts.append(f"Sector: {p.get('Sector')}")
            
            # Add any entrepreneurial information
            if p.get('Previous Startups'):
                text_parts.append(f"Previous Startups: {p.get('Previous Startups')}")
                
            if p.get('Funding History'):
                text_parts.append(f"Funding History: {p.get('Funding History')}")
            
            # Add any additional fields that might be in the CSV
            for key, value in p.items():
                if (key not in ['Full Name', 'Current Position', 'Company', 'Location', 
                               'LinkedIn', 'Twitter', 'Website', 'Email',
                               'About', 'Skills', 'Experience', 'Education', 
                               'Industry', 'Sector', 'Previous Startups', 'Funding History'] 
                    and value and str(value).lower() != 'nan'):
                    text_parts.append(f"{key}: {value}")
            
            # Join all parts with newlines for better separation
            text = "\n".join(text_parts)
            texts.append(text)
            
            # Log the first few profiles to help with debugging
            if len(texts) <= 3:
                print(f"Profile {len(texts)} text representation:\n{text}\n")
        
        # Generate embeddings
        embeddings = self.embeddings.embed_documents(texts)
        
        # Store in vector database
        self.vector_store.upsert_profiles(embeddings, profiles)
        
        return len(profiles)

    @async_timeout(300)  # 5 minute overall timeout for the entire process
    async def process_message(self, query: str):
        """Process a user message through the workflow."""
        # Reset progress message
        self.progress_message = None
        
        # Initialize the state as a simple dictionary
        state = {
            "query": query,
            "query_type": "",
            "filter_key": "",
            "filter_value": "",
            "retrieved_profiles": [],
            "web_search_results": [],
            "analysis_results": [],
            "final_response": {},
            "error": ""
        }
        
        try:
            # Manually execute the workflow nodes in sequence
            # First process the query
            state = await self.process_query(state)
            
            # Route based on query type
            next_node = self.query_router(state)
            
            if next_node == "error":
                await cl.Message(content=f"❌ {state['error']}").send()
                return
            
            # Execute the appropriate search method
            if next_node == "search":
                state = await self.vector_search(state)
            elif next_node == "filter":
                state = await self.filter_by_metadata(state)
            
            # Check for errors after search
            if state.get("error"):
                await cl.Message(content=f"❌ {state['error']}").send()
                return
            
            # Continue with the rest of the workflow
            state = await self.web_search(state)
            state = await self.analyze_profiles(state)
            state = await self.format_response(state)
            
        except asyncio.TimeoutError:
            await cl.Message(content="❌ The operation timed out. Please try a simpler query or try again later.").send()
        except Exception as e:
            await cl.Message(content=f"❌ Error processing request: {str(e)}").send()

# Initialize the system
system = FounderAnalysisSystem()

@cl.on_chat_start
async def start():
    """Initialize the chat session and prompt for CSV upload."""
    await cl.Message(
        content="👋 Welcome to the Founder Analysis System! Please upload your CSV file with founder profiles."
    ).send()
    
    files = await cl.AskFileMessage(
        content="Please upload your CSV file",
        accept=["text/csv"],
        max_size_mb=10
    ).send()

    if not files:
        await cl.Message(
            content="No file was uploaded. Please try again."
        ).send()
        return

    file = files[0]
    
    # Show loading message
    msg = cl.Message(content=f"⏳ Processing {file.name}...")
    await msg.send()

    try:
        # Load the profiles with timeout
        num_profiles = await asyncio.wait_for(system.load_profiles(file.path), timeout=120)
        
        await cl.Message(
            content=f"✅ Successfully loaded {num_profiles} founder profiles!\n\n" + 
                    "You can now:\n\n" + 
                    "1. **Search for founders by expertise**:\n" +
                    "   Example: `AI experts in healthcare`\n\n" +
                    "2. **Filter by specific fields**:\n" +
                    "   Example: `filter:Location:San Francisco`\n" +
                    "   Example: `filter:Skills:Machine Learning`\n\n" +
                    "3. **Search across all fields**:\n" +
                    "   Example: `filter:Stanford`\n" +
                    "   Example: `filter blockchain`\n\n" +
                    "4. **Get founder recommendations**:\n" +
                    "   Example: `recommend founders for fintech startup`"
        ).send()
    except asyncio.TimeoutError:
        await cl.Message(content="❌ Loading profiles timed out. The CSV file might be too large or complex.").send()
    except Exception as e:
        await cl.Message(content=f"❌ Error loading profiles: {str(e)}").send()

@cl.on_message
async def main(message: cl.Message):
    """Handle user messages and provide responses."""
    await system.process_message(message.content) 