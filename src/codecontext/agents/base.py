"""
Base agent for test generation

Uses LangGraph for structured workflows.
"""

from typing import Dict, Any
from pathlib import Path
import warnings

from langgraph.graph.state import CompiledStateGraph
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage

from codecontext.core.codebase import Codebase
from codecontext.search.vectorstore import MongoDBVectorStore
from codecontext.agents.tools import (
    initialize_tools,
    get_all_tools
)


class TestGenerationAgent:
    """
    Agent for generating tests using LangGraph
    
    Workflow:
    1. Analyze project patterns
    2. Search for relevant code
    3. Generate test code
    4. Validate and refine
    """
    
    def __init__(
        self,
        codebase: Codebase,
        vector_store: MongoDBVectorStore,
        project_root: Path,
        llm_config: Dict[str, Any]
    ):
        """
        Initialize test generation agent
        
        Args:
            codebase: Parsed codebase
            vector_store: Vector store for search
            project_root: Project root directory
            llm_config: LLM configuration (model, api_key, temperature)
        """
        self.codebase = codebase
        self.vector_store = vector_store
        self.project_root = project_root
        
        # Initialize LLM
        self.llm = ChatAnthropic(
            model=llm_config.get("model", "claude-sonnet-4-20250514"),
            api_key=llm_config.get("api_key"),
            temperature=llm_config.get("temperature", 0.0),
        )
        
        # Initialize tools
        initialize_tools(codebase, vector_store, project_root)
        self.tools = get_all_tools()
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build graph
        self.graph: CompiledStateGraph = self._build_graph()
    
    def _build_graph(self) -> CompiledStateGraph:
        """
        Build LangGraph workflow
        
        Returns:
            Compiled state graph
        """
        from langgraph.prebuilt import create_react_agent
        
        # Suppress deprecation warning (we'll upgrade LangGraph later)
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        
        # Create ReAct agent with tools
        agent = create_react_agent(
            model=self.llm,  # type: ignore
            tools=self.tools,
        )
        
        return agent
    
    def generate_test(
        self,
        target_function: str,
        target_file: str,
        additional_context: str = ""
    ) -> str:
        """
        Generate a test for a specific function
        
        Args:
            target_function: Name of function to test
            target_file: File containing the function
            additional_context: Optional additional context
        
        Returns:
            Generated test code
        """
        # Build prompt
        prompt = self._build_test_generation_prompt(
            target_function,
            target_file,
            additional_context
        )
        
        # Run agent
        result = self.graph.invoke({
            "messages": [HumanMessage(content=prompt)]
        })
        
        # Extract final answer
        return self._extract_test_code(result)

    @staticmethod
    def _build_test_generation_prompt(
        target_function: str,
        target_file: str,
        additional_context: str
    ) -> str:
        """Build prompt for test generation"""
        
        prompt = f"""Generate a comprehensive test for the following function:

Target Function: {target_function}
File: {target_file}

Steps:
1. Use analyze_project_patterns() to understand the testing framework and conventions
2. Use search_codebase() to find the target function and understand its implementation
3. Use get_function_dependencies() to understand what the function calls
4. Use find_package_usage_examples() to see how relevant libraries are used
5. Generate a test that:
   - Follows the project's testing conventions
   - Tests all major code paths
   - Uses appropriate fixtures and mocking
   - Includes edge cases
   - Has clear, descriptive test names

{additional_context}

Generate the complete test code following the project's patterns.
"""
        
        return prompt

    @staticmethod
    def _extract_test_code(result: Dict) -> str:
        """Extract test code from the agent result"""
        
        messages = result.get("messages", [])
        
        # Get the last AI message
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                content = msg.content
                
                # Extract code block if present
                if "```python" in content:
                    # Extract between ```python and ```
                    start = content.find("```python") + 9
                    end = content.find("```", start)
                    if end != -1:
                        return content[start:end].strip()
                
                return content
        
        return "No test code generated"
