# test_langgraph_types.py

from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic

# Create a simple agent
llm = ChatAnthropic(api_key="test", model="claude-3-5-sonnet-20241022")
agent = create_react_agent(llm, tools=[])

# Check what type it actually is
print(f"Type: {type(agent)}")
print(f"Module: {type(agent).__module__}")
print(f"Class name: {type(agent).__name__}")

# Check what it's an instance of
print(f"MRO: {type(agent).__mro__}")

# List available methods
print(f"\nMethods:")
for attr in dir(agent):
    if not attr.startswith('_'):
        print(f"  - {attr}")