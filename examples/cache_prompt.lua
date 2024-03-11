print("Loading model ...")
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

print("Reading prompt ...")
local prompt = io.read("*a")
local reply, err = session(prompt)
if not reply then
  print("Opoos! ", err)
  return
end
print()
print("reply: ", reply)

-- Dump the current session to "dump.bin"
local data, err = session:dump()
if not data then
  print("Opoos! ", err)
  return
end
local file, err = io.open("dump.bin", "wb")
if not file then
  print("Opoos! ", err)
  return
end
file:write(data)
file:close()
print()
print("Done! Session states of the prompt have been dumped to \"dump.bin\"")
