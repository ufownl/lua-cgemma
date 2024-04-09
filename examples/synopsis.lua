-- Create a Gemma instance
local gemma, err = require("cgemma").new({
  tokenizer = "tokenizer.spm",
  model = "2b-it",
  weights = "2b-it-sfp.sbs"
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
  -- Restore the previous session from "dump.bin"
  local ok, err = session:load("dump.bin")
  if ok then
    print("Previous conversation restored")
  else
    print("New conversation started")
  end

  -- Multi-turn chat
  while session:ready() do
    io.write("> ")
    local text = io.read()
    if not text then
      print("End of file, dumping current session ...")
      -- Dump the current session to "dump.bin"
      local ok, err = session:dump("dump.bin")
      if not ok then
        print("Opoos! ", err)
        return
      end
      print("Done")
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
