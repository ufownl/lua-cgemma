print("Loading model ...")
-- Create a Gemma instance
local gemma, err = require("cgemma").new({
  tokenizer = "tokenizer.spm",
  model = "2b-pt",
  compressed_weights = "2b-it-sfp.sbs"
})
if not gemma then
  print("Opoos! ", err)
  return
end

-- Create a chat session
local session, seed = gemma:session({
  max_generated_tokens = 1
})
if not session then
  print("Opoos! ", seed)
  return
end
print("Random seed of session: ", seed)

print("Reading prompt ...")
local prompt = io.read("*a")
local ok, err = session(prompt, function(token, pos, prompt_size)
  if pos >= prompt_size then
    return false
  end
  io.write(string.format("%d / %d\r", pos + 1, prompt_size))
  io.flush()
  return true
end)
if not ok then
  print("Opoos! ", err)
  return
end
print()

-- Dump the current session to "dump.bin"
local ok, err = session:dump("dump.bin")
if not ok then
  print("Opoos! ", err)
  return
end
print("Done! Session states of the prompt have been dumped to \"dump.bin\"")
