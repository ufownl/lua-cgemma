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

First of all:

```lua
-- Create a Gemma instance
local gemma, err = require("cgemma").new({
  tokenizer = "/path/to/tokenizer.spm",
  model = "gemma2-2b-it",
  weights = "/path/to/2.0-2b-it-sfp.sbs"
})
if not gemma then
  error("Opoos! "..err)
end
```

Single call API example:

```lua
-- Create a chat session
local session, err = gemma:session()
if not session then
  error("Opoos! "..err)
end

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
      error("Opoos! "..err)
    end
    print("reply: ", reply)
  end

  print("Exceed the maximum number of tokens")
  session:reset()
end
```

Batch call API example:

```lua
-- Create 2 chat sessions
local sessions = {}
for i = 1, 2 do
  local session, err = gemma:session()
  if not session then
    error("Opoos! "..err)
  end
  table.insert(sessions, session)
end

-- Run multiple queries using batch interface
local turns = {
  {sessions[1], "Tell me 1+1=?",          sessions[2], "Hello, world!"},
  {sessions[1], "Write it using Python.", sessions[2], "Write what I said in uppercase."}
}
for i, queries in ipairs(turns) do
  print(string.format("Turn %d:\n", i))

  -- Make a batch call
  local result, err = require("cgemma").batch(unpack(queries))
  if not result then
    error("Opoos! "..err)
  end

  -- Display the result of this batch call
  local idx = 1
  for j = 1, #queries do
    if type(queries[j]) == "string" then
      print(string.format("Q%d: %s\n", idx, queries[j]))
      local resp, err = result(queries[j - 1])
      if resp then
        print(resp)
      else
        error("Opoos! "..err)
      end
      idx = idx + 1
    end
  end

  print()
end
```

### APIs for Lua

#### cgemma.info

**syntax:** `cgemma.info()`

Show information of cgemma module.

#### cgemma.scheduler

**syntax:** `<cgemma.scheduler>sched, <string>err = cgemma.scheduler([<table>options])`

Create a scheduler instance.

A successful call returns a scheduler instance. Otherwise, it returns `nil` and a string describing the error.

Available options and default values:

```lua
{
  num_threads = 0,  -- Maximum number of threads to use. (0 = unlimited)
  pin = -1,  -- Pin threads? (-1 = auto, 0 = no, 1 = yes)
  skip_packages = 0,  -- Index of the first socket to use. (0 = unlimited)
  max_packages = 0,  -- Maximum number of sockets to use. (0 = unlimited)
  skip_clusters = 0,  -- Index of the first CCX to use. (0 = unlimited)
  max_clusters = 0,  -- Maximum number of CCXs to use. (0 = unlimited)
  skip_lps = 0,  -- Index of the first LP to use. (0 = unlimited)
  max_lps = 0,  -- Maximum number of LPs to use. (0 = unlimited)
}
```

#### cgemma.scheduler.cpu_topology

**syntax:** `<string>desc = sched:cpu_topology()`

Query CPU topology.

#### cgemma.new

**syntax:** `<cgemma.instance>inst, <string>err = cgemma.new(<table>options)`

Create a Gemma instance.

A successful call returns a Gemma instance. Otherwise, it returns `nil` and a string describing the error.

Available options:

```lua
{
  tokenizer = "/path/to/tokenizer.spm",  -- Path of tokenizer model file.
  model = "gemma2-2b-it",  -- Model type:
                           -- 2b-it (Gemma 2B parameters, instruction-tuned),
                           -- 2b-pt (Gemma 2B parameters, pretrained),
                           -- 7b-it (Gemma 7B parameters, instruction-tuned),
                           -- 7b-pt (Gemma 7B parameters, pretrained),
                           -- gr2b-it (Griffin 2B parameters, instruction-tuned),
                           -- gr2b-pt (Griffin 2B parameters, pretrained),
                           -- gemma2-2b-it (Gemma2 2B parameters, instruction-tuned),
                           -- gemma2-2b-pt (Gemma2 2B parameters, pretrained).
                           -- 9b-it (Gemma2 9B parameters, instruction-tuned),
                           -- 9b-pt (Gemma2 9B parameters, pretrained),
                           -- 27b-it (Gemma2 27B parameters, instruction-tuned),
                           -- 27b-pt (Gemma2 27B parameters, pretrained),
                           -- paligemma-224 (PaliGemma 224*224),
                           -- paligemma-448 (PaliGemma 448*448),
                           -- paligemma2-3b-224 (PaliGemma2 3B 224*224),
                           -- paligemma2-3b-448 (PaliGemma2 3B 448*448),
                           -- paligemma2-10b-224 (PaliGemma2 10B 224*224),
                           -- paligemma2-10b-448 (PaliGemma2 10B 448*448),
  weights = "/path/to/2.0-2b-it-sfp.sbs",  -- Path of model weights file. (required)
  weight_type = "sfp",  -- Weight type:
                        -- sfp (8-bit FP, default)
                        -- f32 (float)
                        -- bf16 (bfloat16)
  seed = 42,  -- Random seed. (default is random setting)
  scheduler = sched_inst,  -- Instance of scheduler, if not provided a default
                           -- scheduler will be attached.
  disabled_words = {...},  -- Words you don't want to generate.
}
```

> [!NOTE]
> If the weights file is not in the new single-file format, then `tokenizer` and `model` options are required.

#### cgemma.instance.disabled_tokens

**syntax:** `<table>tokens = inst:disabled_tokens()`

Query the disabled tokens of a Gemma instance.

#### cgemma.instance.session

**syntax:** `<cgemma.session>sess, <string>err = inst:session([[<cgemma.image>image, ]<table>options])`

Create a chat session.

A successful call returns the session. Otherwise, it returns `nil` and a string describing the error.

Available options and default values:

```lua
{
  max_generated_tokens = 2048,  -- Maximum number of tokens to generate.
  prefill_tbatch = 64,  -- Prefill: max tokens per batch.
  decode_qbatch = 16,  -- Decode: max queries per batch.
  temperature = 1.0,  -- Temperature for top-K.
  top_k = 1,  -- Number of top-K tokens to sample from.
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
  prefill_duration = 1.6746909224894,
  prefill_tokens = 26,
  prefill_tokens_per_second = 15.525252839701,
  time_to_first_token = 1.9843131969683,
  generate_duration = 38.562645539409,
  tokens_generated = 212,
  generate_tokens_per_second = 5.4975481332926
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

#### cgemma.image

**syntax:** `<cgemma.image>img, <string>err = cgemma.image(<string>data_or_path)`

Load image data from the given Lua string or a specific file (PPM format: P6, binary).

**syntax:** `<cgemma.image>img, <string>err = cgemma.image(<integer>width, <integer>height, <table>values)`

Create an image object with the given width, height, and pixel values.

A successful call returns a `cgemma.image` object containing the image data. Otherwise, it returns `nil` and a string describing the error.

#### cgemma.batch

**syntax:** `<cgemma.batch_result>result, <string>err = cgemma.batch(<cgemma.session>sess, <string>text[, <function>stream], ...)`

Generate replies for multiple queries via the batch interface.

A successful call returns a `cgemma.batch_result` object. Otherwise, it returns `nil` and a string describing the error.

The stream function is the same as in [metatable(cgemma.session).call](#metatablecgemmasession__call).

> [!NOTE]
> 1. Each element in a batch must start with a session, followed by a string and an optional stream function, with a stream function means that the corresponding session will be in stream mode instead of normal mode;
> 2. All sessions in a batch must be created by the same Gemma instance;
> 3. Sessions in a batch must not be duplicated;
> 4. Inference arguments of batch call: `max_generated_tokens`, `prefill_tbatch`, and `decode_qbatch` will be the minimum value of all sessions, `temperature` will be the average value of all sessions, and `top_k` will be the maximum value of all sessions.

#### cgemma.batch_result.stats

**syntax:** `<table>statistics = result:stats()`

Get statistics for the batch call that returned the current result.

The statistics fields are the same as in [cgemma.session.stats](#cgemmasessionstats).

#### metatable(cgemma.batch_result).call

**syntax:** `<string or boolean>reply, <string>err = result(<cgemma.session>sess)`

Query the reply corresponding to the session in the result.

A successful call returns the content of the reply (normal mode) or `true` (stream mode). Otherwise, it returns `nil` and a string describing the error.

## License

BSD-3-Clause license. See [LICENSE](https://github.com/ufownl/lua-cgemma?tab=BSD-3-Clause-1-ov-file) for details.
