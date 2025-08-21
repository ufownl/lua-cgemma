local function prefill_fn(_, pos, prompt_size)
  if pos >= prompt_size then
    return false
  end
  return true
end

local function stream_fn(token, pos, prompt_size)
  if pos >= prompt_size then
    if token then
      if pos == prompt_size then
        print()
      end
      io.write(token)
      io.flush()
    else
      print("\n")
    end
  end
  return true
end

local prompt = [[<start_of_turn>user
You are a talented poet who is very good at writing poems in the style of %s. Your task is to create a poem based on the content of this picture. Please only reply with the poem you created, and do not reply with any description, explanation, or other content.]]

local args = require("argparse").parse(arg)
if args.help then
  require("argparse").help(
    "Create a poem based on the given image.",
    "resty ai_poet.lua --image IMAGE [other options]"
  )
  print("  --image: Path of image file (PPM format: P6, binary).")
  print("  --poet: Poet's name. (default: Thomas Stearns Eliot)")
  print("  --temperature: Temperature for top-K. (default: 1.0)")
  print("  --top_k: Number of top-K tokens to sample from. (default: 5)")
  return
end
if not args.image then
  error("Opoos! Image MUST be set.")
end

print(string.format("[%s] Loading model...", os.date("%Y-%m-%d %H:%M:%S")))
local gemma = assert(require("cgemma").new({
  tokenizer = args.tokenizer or "tokenizer.spm",
  weights = args.weights or "4b-it-sfp.sbs"
}))
print(string.format("[%s] Embedding image...", os.date("%Y-%m-%d %H:%M:%S")))
local image = assert(gemma:embed_image(args.image))
local session = assert(gemma:session({
  temperature = tonumber(args.temperature),
  top_k = tonumber(args.top_k) or 5,
  no_wrapping = true
}))
print(string.format("[%s] Prefilling prompt...", os.date("%Y-%m-%d %H:%M:%S")))
assert(session(image, string.format(prompt, args.poet or "Thomas Stearns Eliot"), prefill_fn))
print(string.format("[%s] Writing poem...", os.date("%Y-%m-%d %H:%M:%S")))
assert(session("<end_of_turn>\n<start_of_turn>model\n", stream_fn))
print(string.format("[%s] Done.", os.date("%Y-%m-%d %H:%M:%S")))
