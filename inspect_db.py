import chromadb
import logging


chroma_client=chromadb.PersistentClient(path="./chroma_db")
collection=chroma_client.get_collection(name="city_air_quality")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"single storage record:{collection.peek(limit=1)}")

# To delete the rows in DB
chroma_client.delete_collection(name="city_air_quality")
logging.info("deleted the collection")