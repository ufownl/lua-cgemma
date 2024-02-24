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
    local reply, err = gemma(text)
    if not reply then
      print("Opoos! ", err)
      return
    end
    print("reply: ", reply)
  end
  print("Exceed the maximum number of tokens")
end
