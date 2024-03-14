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
local ok, err = session(prompt, function(token, pos, prompt_size)
  if pos < prompt_size then
    -- Gemma is processing the prompt
    io.write(string.format("%d / %d\r", pos + 1, prompt_size))
  elseif token then
    -- Stream the token text output by Gemma here
    if pos == prompt_size then
      io.write("\n\nreply: ")
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
print()

-- Dump the current session to "dump.bin"
local ok, err = session:dump("dump.bin")
if not ok then
  print("Opoos! ", err)
  return
end
print()
print("Done! Session states of the prompt have been dumped to \"dump.bin\"")
