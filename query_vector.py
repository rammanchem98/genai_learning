from google import genai
from google.cloud import bigquery
from google.genai import types
import chromadb
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

client=genai.Client()
chromaddb_client=chromadb.PersistentClient(path='./chroma_db')
collection=chromaddb_client.get_collection(name="city_air_quality")


def search_air_quality_db(query:str):
    """
        Searches the local vector database for real-time air quality metrics, 
        temperature, humidity, and PM2.5 values of various world cities.
    
        Args:
            query: The natural language search query string.
    """

    logging.info(f"[TOOL] search_air_quality_db called with query: '{query}'")

    response=client.models.embed_content(
        model='gemini-embedding-2',
        contents=query,
        config=types.EmbedContentConfig(
            task_type='RETRIEVAL_QUERY',
            output_dimensionality=768
        )
    )

    query_vector = response.embeddings[0].values

    # Query ChromaDB
    final_result = collection.query([query_vector], n_results=5)
    context_doc = final_result["documents"][0]
    
    return "\n".join(context_doc)

def run_agent(user_prompt:str):
    logging.info(f"[Agent] initializing the request: {user_prompt}")

    config=types.GenerateContentConfig(
        tools=[search_air_quality_db],
        temperature=0.0,
        system_instruction="You are an air quality assistant, use your search tool when asked about city metrics"
    )

    response=client.models.generate_content(
        model='gemini-3.5-flash',
        config=config,
        contents=user_prompt,

    )

    if response.function_calls:
        for function_call in response.function_calls:
            name=function_call.name
            args=function_call.args
            logging.info(f"[Agent] LLM requested tool execution : {name} with args {args}")

            if name == search_air_quality_db:
                tool_query=args['query']
                tool_output=search_air_quality_db(query=tool_query)

                logging.info("[AGENT] Sending tool results back to LLM for final generation.")

                final_response=client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[
                        user_prompt,
                        response.candidates[0].content,
                        types.Content(
                            role='tool',
                            parts=[
                                types.Part.from_function_response(
                                    name=name,
                                    response={'result':tool_output}
                                )
                            ]
                        )
                    ],
                    config=config
                )
                logging.info(f"\nFinal Agent Response:\n{final_response.text}")
    
    else:
        logging.info(f"\nFinal Agent Response (No Tool Used):\n{response.text}")

    

if __name__=="__main__":
    logging.info("Test 1.......")
    run_agent("How safe is it to jog in delhi in morning time")

    logging.info("Test 2.......")
    run_agent("what is the weather today in SF.")

    logging.info("Test 3: No need to call tool for response")
    run_agent("Hello, You alright ?")
    
