# Online demo of lua-cgemma

[![demo-hf-space](https://github.com/ufownl/lua-cgemma/assets/9405195/8c71920d-fb0d-43da-b396-89a5140adbee)](https://huggingface.co/spaces/RangerUFO/lua-cgemma-demo)

This demo requires [OpenResty](https://openresty.org/) to run, see [here](https://openresty.org/en/installation.html) to learn how to install it.

## Usage

**1st step:** Put the model weights (`2.0-2b-it-sfp.sbs`) and tokenizer weights (`tokenizer.spm`) into this directory.

*The other variants are also OK, but the relevant scripts need to be modified accordingly.*

**2nd step:** Generate the context state of chat session: `resty print_prompt.lua | resty ../tools/dump_prompt.lua`

*This step is optional, you can run this demo with no context state or any context state dumped from other chat sessions.*

**3rd step:** Start the demo server: `openresty -p . -c cgemma_demo.conf`

*When you want to stop the demo server, just run: `openresty -p . -c cgemma_demo.conf -s stop`*

*[OpenResty](https://openresty.org/)'s core is [NGINX](https://nginx.org/), so the commands are extremely similar to [NGINX](https://nginx.org/).*

**4th step:** Open [http://localhost:8042](http://localhost:8042) with your browser, and then you can chat with [Gemma](https://ai.google.dev/gemma).
