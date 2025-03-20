import asyncio
import sys
import json
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()  # load environment variables from .env

# Initialize Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
    
    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])
    
    async def process_query(self, query: str) -> str:
        """Process a query using Gemini and available tools"""
        response = await self.session.list_tools()
        
        # Convert MCP tools to Gemini format
        available_tools = []
        tool_descriptions = []
        
        for tool in response.tools:
            # Convert the schema to remove 'title' fields which Gemini doesn't expect
            schema = json.loads(tool.inputSchema) if isinstance(tool.inputSchema, str) else tool.inputSchema
            
            # Recursively remove 'title' fields from schema
            def clean_schema(obj):
                if isinstance(obj, dict):
                    if 'title' in obj:
                        del obj['title']
                    for key, value in list(obj.items()):
                        obj[key] = clean_schema(value)
                elif isinstance(obj, list):
                    return [clean_schema(item) for item in obj]
                return obj
            
            cleaned_schema = clean_schema(schema)
            
            available_tools.append({
                "function_declarations": [{
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": cleaned_schema
                }]
            })
            
            # Add to our tool descriptions for the enhanced prompt
            tool_descriptions.append(f"- {tool.name}: {tool.description}")
            
            # Add parameter details to descriptions
            if isinstance(schema, dict) and 'properties' in schema:
                for param_name, param_details in schema['properties'].items():
                    param_desc = param_details.get('description', '')
                    tool_descriptions.append(f"  - Parameter '{param_name}': {param_desc}")

        # Enhance the query with instructions about location conversion
        enhanced_query = f"""
I need you to help me with a weather query: "{query}"

When using weather tools, if a location is mentioned, you should automatically convert the location name to its approximate latitude and longitude.
For example:
- Miami, FL is approximately at latitude 25.7617, longitude -80.1918
- New York City is approximately at latitude 40.7128, longitude -74.0060
- Los Angeles is approximately at latitude 34.0522, longitude -118.2437

I have the following tools available:
{chr(10).join(tool_descriptions)}

Please use these tools to answer my query. If you need latitude and longitude for a location mentioned in my query, convert it automatically - don't ask me for coordinates.
"""

        # Create Gemini model
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={"temperature": 0.2},
            tools=available_tools
        )
        
        # Start chat session
        chat = model.start_chat(history=[])
        
        # Process response and handle tool calls
        final_text = []
        
        try:
            # Get initial response with our enhanced query
            response = chat.send_message(enhanced_query)
            
            if not response.parts:
                return "No response from Gemini"
            
            for part in response.parts:
                # Check if this is text content
                if hasattr(part, 'text') and part.text:
                    final_text.append(part.text)
                
                # Check if this is a function call
                if hasattr(part, 'function_call'):
                    tool_name = part.function_call.name
                    tool_args = part.function_call.args
                    
                    # Execute tool call
                    result = await self.session.call_tool(tool_name, tool_args)
                    final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                    
                    # Get the result content
                    tool_result = result.content
                    final_text.append(f"[Tool result: {tool_result}]")
                    
                    # Simply send the result as a new message
                    human_readable = f"The tool {tool_name} returned the following result: {tool_result}"
                    follow_up_response = chat.send_message(human_readable)
                    final_text.append("this is Ujjwal's weather agent :)")
                    if follow_up_response.parts and hasattr(follow_up_response.parts[0], 'text'):
                        final_text.append(follow_up_response.parts[0].text)
        
        except Exception as e:
            final_text.append(f"Error in processing with Gemini: {str(e)}")
            import traceback
            final_text.append(traceback.format_exc())
        
        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")
                import traceback
                print(traceback.format_exc())

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:    
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())