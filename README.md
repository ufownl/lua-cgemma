# lua-cgemma

Lua bindings for [gemma.cpp](https://github.com/google/gemma.cpp).

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
  model = "2b-it",
  compressed_weights = "/path/to/2b-it-sfp.sbs"
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

#### cgemma.new

**syntax:** `<cgemma.instance>inst, <string>err = cgemma.new(<table>options)`

Create a Gemma instance.

A successful call returns a Gemma instance. Otherwise, it returns `nil` and a string describing the error.

Available options:

```lua
{
  tokenizer = "/path/to/tokenizer.spm",  -- Path of tokenizer model file. (required)
  model = "2b-it",  -- Model type - can be 2b-it (2B parameters, instruction-tuned),
                    -- 2b-pt (2B parameters, pretrained), 7b-it (7B parameters,
                    -- instruction-tuned), or 7b-pt (7B parameters, pretrained).
                    -- (required)
  compressed_weights = "/path/to/2b-it-sfp.sbs",  -- Path of compressed weights file. (required)
  weights = "/path/to/weights.sbs",  -- Path of uncompressed weights file. Only required if
                                     -- compressed weights file is not present and needs
                                     -- to be regenerated.
  scheduler = sched_inst,  -- Instance of scheduler, if not provided a default scheduler
                           -- will be attached.
}
```

#### cgemma.scheduler

**syntax:** `<cgemma.scheduler>sched, <string>err = cgemma.scheduler([<number>num_threads])`

Create a scheduler instance.

A successful call returns a scheduler instance. Otherwise, it returns `nil` and a string describing the error.

The only parameter `num_threads` indicates the number of threads in the internal thread pool. If not provided or `num_threads <= 0`, it will create a default scheduler with the number of threads depending on the concurrent threads supported by the implementation.

#### cgemma.instance.session

**syntax:** `<cgemma.session>sess, <number or string>seed = inst:session([<table>options])`

Create a chat session.

A successful call returns the session and its random seed. Otherwise, it returns `nil` and a string describing the error.

Available options and default values:

```lua
{
  max_tokens = 3072,  -- Maximum number of tokens in prompt + generation.
  max_generated_tokens = 2048,  -- Maximum number of tokens to generate.
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

#### cgemma.session.dump

**syntax:** `<string>data, <string>err = sess:dump()`

Dump the current state of the session.

A successful call returns the state data (binary) of the session. Otherwise, it returns `nil` and a string describing the error.

#### cgemma.session.restore

**syntax:** `<boolean>ok, <string>err = sess:restore(<string>data)`

Restore a session from the given state data.

A successful call returns `true`. Otherwise, it returns `false` and a string describing the error.

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
