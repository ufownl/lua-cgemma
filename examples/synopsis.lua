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
