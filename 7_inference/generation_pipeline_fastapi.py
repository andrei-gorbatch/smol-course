from fastapi import FastAPI, HTTPException
from transformers import pipeline
import uvicorn
from fastapi.responses import HTMLResponse

app = FastAPI()

# Initialize pipeline globally
generator = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    device_map="auto"
)

from fastapi import Form

@app.post("/generate")
async def generate_text(prompt: str = Form(...)):
    try:
        if not prompt:
            raise HTTPException(status_code=400, detail="No prompt provided")
            
        response = generator(
            prompt,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7
        )
        
        return {"generated_text": response[0]['generated_text']}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Text Generation API</title>
        </head>
        <body>
            <h1>Welcome to the text generation API</h1>
            <form action="/generate" method="post">
                <label for="prompt">Enter prompt:</label>
                <input type="text" id="prompt" name="prompt">
                <input type="submit" value="Generate">
            </form>
        </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)