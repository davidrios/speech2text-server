# Whisper Server

A server to abstract the available whisper implementations and serve
through a common websocket interface. Use it with 
[speech2text-app](https://github.com/davidrios/speech2text-app).


## How to run

### Install and configure

First install ffmpeg in the appropriate way for your platform. Check
https://github.com/kkroening/ffmpeg-python for more info.

Create a new venv, activate it and install the base dependencies:

```
pip install -r requirements.txt
```

Create a copy of the `.env.template` file as `.env` and edit the settings as desired.

Now you choose an engine and set it in the env file:

- `whisper`: The original [Whisper from OpenAI](https://github.com/openai/whisper)
- `whisper.cpp`: Port of OpenAI's Whisper model in C/C++ (https://github.com/ggerganov/whisper.cpp)
- `openai`: Uses the OpenAI online API

And finish the configuration for each:


#### whisper

Install the extra packages.

- Only if you have a NVIDIA GPU: `pip install -r requirements-whisper-nvidia.txt`
- Or else: `pip install -r requirements-whisper.txt`


#### whisper.cpp

**Unix only**

You'll probably need to install compilation related packages before.

Install the extra packages:

```
pip install -r requirements-whispercpp.txt
```


#### openai

Uses the online OpenAI API platform. You'll need to
[set up an account](https://platform.openai.com/signup), get an
API key and set it in the `.env` file.


### Gradio test interface

There's a test Gradio interface available. Set `USE_GRADIO=1` in the `.env` file
and install extra packages:

```
pip install -r requirements-gradio.txt
```


### Start the server

Run this to start the server:

```
uvicorn main:app --env-file .env --port 9748 --host 0.0.0.0
```

The server will be available at http://127.0.0.1:9748/. If you enabled Gradio,
the test interface will be available at http://127.0.0.1:9748/gradio/.