[![Open in Kaggle](https://img.shields.io/badge/Open_in_Kaggle-blue?style=plastic&logo=kaggle&labelColor=grey)](https://www.kaggle.com/code/ufownl/how-to-start-a-gemma-chatbot) [![Open in HF Spaces](https://img.shields.io/badge/%F0%9F%A4%97-Open_in_HF_Spaces-white?style=plastic)](https://huggingface.co/spaces/RangerUFO/lua-cgemma-demo)
# lua-cgemma

Lua bindings for [gemma.cpp](https://github.com/google/gemma.cpp).

## Requirements

Before starting, you should have installed:

- [CMake](https://cmake.org/)
- C++ compiler, supporting at least C++17
- [LuaJIT](https://luajit.org/), recommended to install [OpenResty](https://openresty.org/) directly

## Installation

**1st step:** Clone the source code from [GitHub](https://github.com/ufownl/lua-cgemma): `git clone https://github.com/ufownl/lua-cgemma.git`

**2nd step:** Build and install:

To build and install using the default settings, just enter the repository's directory and run the following commands:

```bash
mkdir build
cd build
cmake .. && make
sudo make install
```

**3rd step:** See [here](https://github.com/google/gemma.cpp?tab=readme-ov-file#step-1-obtain-model-weights-and-tokenizer-from-kaggle-or-hugging-face-hub) to learn how to obtain model weights and tokenizer.

## Usage

### Synopsis

```lua
-- Create a Gemma instance
local gemma, err = require("cgemma").new({
  tokenizer = "/path/to/tokenizer.spm",
  model = "gemma2-2b-it",
  weights = "/path/to/2.0-2b-it-sfp.sbs"
})
if not gemma then
  print("Opoos! ", err)
  return
end

-- Create a chat session
local session, seed = gemma:session()
if not session then
  print("Opoos! ", seed)
  return
end

print("Random seed of session: ", seed)
while true do
  print("New conversation started")

  -- Multi-turn chat
  while session:ready() do
    io.write("> ")
    local text = io.read()
    if not text then
      print("End of file")
      return
    end
    local reply, err = session(text)
    if not reply then
      print("Opoos! ", err)
      return
    end
    print("reply: ", reply)
  end

  print("Exceed the maximum number of tokens")
  session:reset()
end
```

### APIs for Lua

#### cgemma.info

**syntax:** `cgemma.info()`

Show information of cgemma module.

#### cgemma.new

**syntax:** `<cgemma.instance>inst, <string>err = cgemma.new(<table>options)`

Create a Gemma instance.

A successful call returns a Gemma instance. Otherwise, it returns `nil` and a string describing the error.

Available options:

```lua
{
  tokenizer = "/path/to/tokenizer.spm",  -- Path of tokenizer model file. (required)
  model = "gemma2-2b-it",  -- Model type:
                           -- 2b-it (Gemma 2B parameters, instruction-tuned),
                           -- 2b-pt (Gemma 2B parameters, pretrained),
                           -- 7b-it (Gemma 7B parameters, instruction-tuned),
                           -- 7b-pt (Gemma 7B parameters, pretrained),
                           -- 9b-it (Gemma2 9B parameters, instruction-tuned),
                           -- 9b-pt (Gemma2 9B parameters, pretrained),
                           -- 27b-it (Gemma2 27B parameters, instruction-tuned),
                           -- 27b-pt (Gemma2 27B parameters, pretrained),
                           -- gr2b-it (Griffin 2B parameters, instruction-tuned),
                           -- gr2b-pt (Griffin 2B parameters, pretrained),
                           -- gemma2-2b-it (Gemma2 2B parameters, instruction-tuned),
                           -- gemma2-2b-pt (Gemma2 2B parameters, pretrained).
                           -- (required)
  weights = "/path/to/2.0-2b-it-sfp.sbs",  -- Path of model weights file. (required)
  weight_type = "sfp",  -- Weight type:
                        -- sfp (8-bit FP, default)
                        -- f32 (float)
                        -- bf16 (bfloat16)
  scheduler = sched_inst,  -- Instance of scheduler, if not provided a default
                           -- scheduler will be attached.
  disabled_words = {...},  -- Words you don't want to generate.
}
```

#### cgemma.scheduler

**syntax:** `<cgemma.scheduler>sched, <string>err = cgemma.scheduler([<number>max_threads, <number>max_clusters])`

Create a scheduler instance.

A successful call returns a scheduler instance. Otherwise, it returns `nil` and a string describing the error.

Available parameters:

| Parameter | Description |
| --------- | ----------- |
| max_threads | Maximum number of threads to use. (default: `0` means unlimited) |
| max_clusters | Maximum number of sockets/CCXs to use. (default: `0` means unlimited) |

#### cgemma.scheduler.cpu_topology

**syntax:** `<table>clusters = sched:cpu_topology()`

Query CPU topology.

#### cgemma.instance.disabled_tokens

**syntax:** `<table>tokens = inst:disabled_tokens()`

Query the disabled tokens of a Gemma instance.

#### cgemma.instance.session

**syntax:** `<cgemma.session>sess, <number or string>seed = inst:session([<table>options])`

Create a chat session.

A successful call returns the session and its random seed. Otherwise, it returns `nil` and a string describing the error.

Available options and default values:

```lua
{
  max_tokens = 3072,  -- Maximum number of tokens in prompt + generation.
  max_generated_tokens = 2048,  -- Maximum number of tokens to generate.
  prefill_tbatch = 64,  -- Prefill: max tokens per batch.
  decode_qbatch = 16,  -- Decode: max queries per batch.
  temperature = 1.0,  -- Temperature for top-K.
  seed = 42,  -- Random seed. (default is random setting)
}
```

#### cgemma.session.ready

**syntax:** `<boolean>ok = sess:ready()`

Check if the session is ready to chat.

#### cgemma.session.reset

**syntax:** `sess:reset()`

Reset the session to start a new conversation.

#### cgemma.session.dumps

**syntax:** `<string>data, <string>err = sess:dumps()`

Dump the current state of the session to a Lua string.

A successful call returns a Lua string that stores state data (binary) of the session. Otherwise, it returns `nil` and a string describing the error.

#### cgemma.session.loads

**syntax:** `<boolean>ok, <string>err = sess:loads(<string>data)`

Load the state data from the given Lua string to restore a previous session.

A successful call returns `true`. Otherwise, it returns `false` and a string describing the error.

#### cgemma.session.dump

**syntax:** `<boolean>ok, <string>err = sess:dump(<string>path)`

Dump the current state of the session to a specific file.

A successful call returns `true`. Otherwise, it returns `false` and a string describing the error.

#### cgemma.session.load

**syntax:** `<boolean>ok, <string>err = sess:load(<string>path)`

Load the state data from the given file to restore a previous session.

A successful call returns `true`. Otherwise, it returns `false` and a string describing the error.

#### cgemma.session.stats

**syntax:** `<table>statistics = sess:stats()`

Get statistics for the current session.

Example of statistics:

```lua
{
  prefill_tokens_per_second = 34.950446398036,
  generate_tokens_per_second = 9.0089134969039,
  time_to_first_token = 0.8253711364232,
  tokens_generated = 85
}
```

#### metatable(cgemma.session).__call

**syntax:** `<string or boolean>reply, <string>err = sess(<string>text[, <function>stream])`

Generate reply.

A successful call returns the content of the reply (without a stream function) or `true` (with a stream function). Otherwise, it returns `nil` and a string describing the error.

The stream function is defined as follows:

```lua
function stream(token, pos, prompt_size)
  if pos < prompt_size then
    -- Gemma is processing the prompt
    io.write(pos == 0 and "reading and thinking ." or ".")
  elseif token then
    -- Stream the token text output by Gemma here
    if pos == prompt_size then
      io.write("\nreply: ")
    end
    io.write(token)
  else
    -- Gemma's output reaches the end
    print()
  end
  io.flush()
  -- return `true` indicates success; return `false` indicates failure and terminates the generation
  return true
end
```

## License

BSD-3-Clause license. See [LICENSE](https://github.com/ufownl/lua-cgemma?tab=BSD-3-Clause-1-ov-file) for details.
