from google import genai
from google.genai import types
from google.genai.types import AutomaticFunctionCallingConfig
import chromadb
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize clients
client = genai.Client()

#  THE CRITICAL STEP: Disable the automated engine background loop
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

def run_stateful_agent(chat_session,user_prompt: str):

    logging.info(f"[AGENT] Initializing request: '{user_prompt}'")

    response=chat_session.send_message(user_prompt)

    if response.function_calls:
        tool_response=[]
        for res in response.function_calls:
            name=res.name
            args=res.args

            logging.info(f"[AGENT] LLM requested action: {name} | Args: {args}")

            if name=="search_air_quality_db":
                tool_output=search_air_quality_db(query=args['query'])
                
                response_part=types.Part.from_function_response(
                    name=name,
                    response={'result':tool_output}
                )
                tool_response.append(response_part)

                logging.info("[AGENT] Sending manual observation payloads back to the Chat Session.")
        
        # Turn 2: Send the list of tool parts back into the chat session
        final_response = chat_session.send_message(tool_response)
        print(f"\nFinal Agent Response:\n{final_response.text}")
    else:
        # If no tool call was needed, print the direct conversation text
        print(f"\nFinal Agent Response (No Tool Used):\n{response.text}")


if __name__ == "__main__":
    # Create the unified configuration mapping
    agent_config=types.GenerateContentConfig(
        tools=[search_air_quality_db],
        temperature=0.0,
        system_instruction="You are an air quality assistant. Use your search tool when asked about city metrics.",
        automatic_function_calling=AutomaticFunctionCallingConfig(disable=True)
    )

    # instantiate the stateful chat
    stateful_chat=client.chats.create(
        model="gemini-2.5-flash",
        config=agent_config
    )

    logging.info("First test run....")
    run_stateful_agent(chat_session=stateful_chat,user_prompt="Compare the air metrics of New Delhi and Johannesburg.")

    logging.info("chat continuation..")
    run_stateful_agent(chat_session=stateful_chat,user_prompt="Which city has highest humidity ? ")