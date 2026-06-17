from google import genai
from google.genai import types
from google.genai.types import AutomaticFunctionCallingConfig
import chromadb
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize clients
client = genai.Client()

# 🚨 THE CRITICAL STEP: Disable the automated engine background loop
client.automatic_function_calling = False

chromaddb_client = chromadb.PersistentClient(path='./chroma_db')
collection = chromaddb_client.get_collection(name="city_air_quality")

def search_air_quality_db(query: str) -> str:
    """Searches the local vector database for real-time air quality metrics, 
    temperature, humidity, and PM2.5 values of various world cities.
    
    Args:
        query: The natural language search query string.
    """
    logging.info(f"[TOOL] search_air_quality_db called with query: '{query}'")
    
    response = client.models.embed_content(
        model='gemini-embedding-2',
        contents=query,
        config=types.EmbedContentConfig(
            task_type='RETRIEVAL_QUERY',
            output_dimensionality=768
        )
    )
    query_vector = response.embeddings[0].values
    final_result = collection.query([query_vector], n_results=5)
    context_doc = final_result["documents"][0]
    
    return "\n".join(context_doc)

def run_agent(user_prompt: str):
    logging.info(f"[AGENT] Initializing request: '{user_prompt}'")

    config = types.GenerateContentConfig(
        tools=[search_air_quality_db],
        temperature=0.0,
        system_instruction="You are an air quality assistant. Use your search tool when asked about city metrics.",
        automatic_function_calling=AutomaticFunctionCallingConfig(disable=True)
    )

    # First turn: Gemini decides whether to request tool execution
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=user_prompt,
        config=config
    )

    # Explicitly check if Gemini requested a function call metadata packet
    if response.function_calls:
        tool_responses = []
        
        for function_call in response.function_calls:
            name = function_call.name
            args = function_call.args
            
            
            logging.info(f"[AGENT] LLM requested action: {name} | Args: {args}")
            
            if name == "search_air_quality_db":
                # Execute the tool function locally using the model's generated argument
                tool_output = search_air_quality_db(query=args["query"])
                
                # Bundle the output text and anchor it to the specific execution Call ID
                response_part = types.Part.from_function_response(
                    name=name,
                    response={"result": tool_output})
                
                tool_responses.append(response_part)
                
        logging.info("[AGENT] Sending manual observation payloads back to the LLM core.")
        
        # Second turn: Pass the interaction sequence back over the wire
        final_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                user_prompt,
                response.candidates[0].content,
                types.Content(role="tool", parts=tool_responses)
            ],
            config=config
        )
        print(f"\nFinal Manual Agent Response:\n{final_response.text}")
    else:
        print(f"\nFinal Manual Agent Response (No Tool Used):\n{response.text}")

if __name__ == "__main__":
    # Testing manual loop with a parallel query that requests multiple cities at once
    run_agent("Compare the air metrics of New Delhi and Johannesburg.")