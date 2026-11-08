import os

# The offline tests never call a real model, but some modules build an OpenAI client when they are
# imported, and the client refuses to start without a key. A dummy one is enough.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
