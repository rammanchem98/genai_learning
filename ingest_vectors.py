from google.cloud import bigquery
from google import genai
from google.genai import types
import chromadb
import logging
import time
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting the vector ingestion process.")

client=genai.Client()
chroma_client=chromadb.PersistentClient(path="./chroma_db")
collection=chroma_client.get_or_create_collection(name="city_air_quality")
bq_client=bigquery.Client(project="london-air-quality-pipeline")

def ingest_vectors_to_bq(query:str,table_id:str):
    """Fetches data from BigQuery, generates embeddings, and stores them in a ChromaDB collection."""

    query="""SELECT * FROM `london-air-quality-pipeline.world_weather_dataset.world_air_quality`
    """

    logging.info("Executing BigQuery to fetch data.")

    df= bq_client.query(query).to_dataframe()
    df=df.head(100)

    logging.info(f"Fetched {len(df)} records from BigQuery. Generating embeddings and ingesting into ChromaDB.")
    
    def format_row(row):
        return f"city:{row['city']},temp:{row['temp']},humidity:{row['humidity']},wind_Speed:{row['wind_speed']},pm25_value:{row['pm25_value']}"
                           
    documents=df.apply(format_row, axis=1).tolist()
    logging.info(f"Formatted {len(documents)} documents for embedding generation.")
    logging.info(f"Sample formatted document: {documents[0]}")

    logging.info("Generating embeddings for each document.")

    
    list_of_embeddings=[]

    # THE BELOW CODE IS FOR IMPLEMENTING THE EMBEDDING FROM LOCAL NUMPY
    # fake_chunk_vectors=[np.random.rand(768).tolist() for _ in documents]
    # list_of_embeddings.extend(fake_chunk_vectors)
    # logging.info(f"Length of embeddings:{str(len(list_of_embeddings))}")


    batch_size=20
    for i in range(0,len(documents),batch_size):
        chunk_docs=documents[i:i+batch_size]
        logging.info(f"current batch size:{len(chunk_docs)}")
        formatted_contents = [types.Content(parts=[types.Part.from_text(text=doc)]) for doc in chunk_docs]
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=formatted_contents,
            config=types.EmbedContentConfig(
            output_dimensionality=768
            )
        ) 
        time.sleep(8)  # Sleep to respect rate limits
        list_of_embeddings.extend([e.values for e in response.embeddings])
    
    logging.info("Total accumulated master embeddings: " + str(len(list_of_embeddings)))

    if len(list_of_embeddings) != len(documents):
        raise ValueError(f"Length mismatch between embeddings:{len(list_of_embeddings)} and  documents:{len(documents)}")

    # upsert= update+insert rows
    collection.upsert(documents=documents,
                   embeddings=list_of_embeddings,
                   ids=df.apply(lambda row:f"{row['city']}_{row['update_timestamp']}".lower(),axis=1).tolist()
                
                   )
    
    
    logging.info(f"Successfully ingested {len(documents)} records into ChromaDB collection 'city_air_quality'.")



if __name__ == "__main__":
    ingest_vectors_to_bq(query="",table_id="")