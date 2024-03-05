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
    local ok, err = session(text, function(token, pos, prompt_size)
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
  session:reset()
end
