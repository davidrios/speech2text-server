# Whisper Server

To run using [whisper.cpp](https://github.com/ggerganov/whisper.cpp) and Gradio interface:

```
# create and activate virtualenv
python3 -m venv .venv && source .venv/bin/activate

# install dependencies
pip install -r requirements.txt -r requirements-whispercpp.txt -r requirements-gradio.txt

# run the server:
cp .env.template .env
uvicorn main:app --env-file .env --reload
```

Navigate to http://127.0.0.1:8000/gradio/ to test the Gradio interface.