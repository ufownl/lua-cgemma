-- Create a Gemma instance
local gemma, err = require("cgemma").new({
  tokenizer = "tokenizer.spm",
  model = "2b-it",
  compressed_weights = "2b-it-sfp.sbs"
})
if not gemma then
  print("Opoos! ", err)
  return
end

while true do
  -- Start a chat session
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
    local ok, err = gemma(text, function(token, pos, prompt_size)
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
    end)
    if not ok then
      print("Opoos! ", err)
      return
    end
  end
  print("Exceed the maximum number of tokens")
end
