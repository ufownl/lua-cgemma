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

while true do
  -- Start a new chat session
  local seed, err = gemma:start_session()
  if not seed then
    print("Opoos! ", err)
    return
  end
  print("New session started")
  print("Random seed of current session: ", seed)

  -- Multi-turn chat
  while gemma:ready() do
    io.write("> ")
    local text = io.read()
    if not text then
      print("End of file")
      return
    end
    local reply, err = gemma(text)
    if not reply then
      print("Opoos! ", err)
      return
    end
    print("reply: ", reply)
  end
  print("Exceed the maximum number of tokens")
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
  tokenizer = "/path/to/tokenizer.spm"  -- Path of tokenizer model file. (required)
  model = "2b-it",  -- Model type - can be 2b-it (2B parameters, instruction-tuned),
                    -- 2b-pt (2B parameters, pretrained), 7b-it (7B parameters,
                    -- instruction-tuned), or 7b-pt (7B parameters, pretrained).
                    -- (required)
  compressed_weights = "/path/to/2b-it-sfp.sbs",  -- Path of compressed weights file. (required)
  num_threads = 1,  -- Number of threads in the internal thread pool. (default is number
                    -- of concurrent threads supported by the implementation)
}
```

#### cgemma.instance.start_session

**syntax:** `<number>seed, <string>err = inst:start_session([<table>options])`

Start a new chat session.

A successful call returns the random seed of the session. Otherwise, it returns `nil` and a string describing the error.

Available options and default values:

```lua
{
  max_tokens = 3072,  -- Maximum number of tokens in prompt + generation.
  max_generated_tokens = 2048,  -- Maximum number of tokens to generate.
  temperature = 1.0,  -- Temperature for top-K.
  seed = 42,  -- Random seed. (default is random setting)
}
```

#### cgemma.instance.ready

**syntax:** `<boolean>ok = inst:ready()`

Check if the Gemma instance is ready to chat.

#### metatable(cgemma.instance).__call

**syntax:** `<string>reply, <string>err = inst(<string>text)`

Generate reply.

A successful call returns the content of the reply. Otherwise, it returns `nil` and a string describing the error.
