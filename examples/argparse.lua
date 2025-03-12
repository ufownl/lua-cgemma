local _M = {}

function _M.parse(cli_args)
  local args = {}
  for i, v in ipairs(cli_args) do
    if string.sub(v, 1, 2) == "--" then
      if cli_args[i + 1] and string.sub(cli_args[i + 1], 1, 2) ~= "--" then
        args[string.sub(v, 3)] = cli_args[i + 1]
      else
        args[string.sub(v, 3)] = true
      end
    end
  end
  return args
end

function _M.help(description, usage)
  require("cgemma").info()
  print()
  print(description)
  print()
  print(string.format("Usage: %s", usage))
  print()
  print("Available options:")
  print("  --tokenizer: Path of tokenizer model file. (default: tokenizer.spm)")
  print("  --model: Model type (default: gemma2-2b-it)")
  print("    2b-it = Gemma 2B parameters, instruction-tuned")
  print("    7b-it = Gemma 7B parameters, instruction-tuned")
  print("    gr2b-it = Griffin 2B parameters, instruction-tuned")
  print("    gemma2-2b-it = Gemma2 2B parameters, instruction-tuned")
  print("    9b-it = Gemma2 9B parameters, instruction-tuned")
  print("    27b-it = Gemma2 27B parameters, instruction-tuned")
  print("    paligemma-224 = PaliGemma 224*224")
  print("    paligemma-448 = PaliGemma 448*448")
  print("    paligemma2-3b-224 = PaliGemma2 3B 224*224")
  print("    paligemma2-3b-448 = PaliGemma2 3B 448*448")
  print("    paligemma2-10b-224 = PaliGemma2 10B 224*224")
  print("    paligemma2-10b-448 = PaliGemma2 10B 448*448")
  print("    gemma3-4b = Gemma3 4B parameters")
  print("    gemma3-1b = Gemma3 1B parameters")
  print("    gemma3-12b = Gemma3 12B parameters")
  print("    gemma3-27b = Gemma3 27B parameters")
  print("  --weights: Path of model weights file. (default: 2.0-2b-it-sfp.sbs)")
  print("  --weight_type: Weight type (default: sfp)")
end

return _M
