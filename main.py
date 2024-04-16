from transformers import BertTokenizer, BertModel
import torch

app = FastAPI()

model = SentenceTransformer('sentence-transformers/LaBSE')

def get_bert_embeddings(text):
    embeddings = model.encode([text])
    return embeddings


@app.get("/")
async def read_root():
    return {"message": "Welcome to BERT Embeddings API"}


@app.post("/embeddings/")
async def get_embeddings(text: str):
    embeddings = get_bert_embeddings(text)
    return {"embeddings": embeddings.tolist()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
