import chromadb
import time

# Initialize client with version 0.6.2 compatible settings
client = chromadb.PersistentClient(path="chroma_db")

def add_documents_safely():
    # Delete collection if it exists to start fresh
    try:
        client.delete_collection("my_collection")
        time.sleep(1.5)  # Give some time for deletion to complete
    except:
        pass
    
    # Create new collection
    collection = client.create_collection(name="my_collection")
    
    # Add documents one by one with delay to prevent race conditions
    try:
        # First document
        collection.add(
            documents=["India is famous for it spices"],
            ids=["id1"]
        )
        time.sleep(0.5)  # Small delay between operations
        
        # Second document
        collection.add(
            documents=["london is famous for its breakfast"],
            ids=["id2"]
        )
        
        print("Documents added successfully!")
        
        # Verify the contents
        results = collection.get()
        print("\nStored documents:")
        print(results)
        queries = [
            "tell me about Indian food",
            "what is famous in India",
            "tell me about breakfast"
        ]
        
        for query in queries:
            result = collection.query(
                query_texts=[query],
                n_results=1
            )
            print(f"\nQuery: '{query}'")
            print(f"Results: {result['documents'][0]}")
            print(f"Distances: {result['distances'][0]}")
        return True
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return False
    

# Execute the function
add_documents_safely()