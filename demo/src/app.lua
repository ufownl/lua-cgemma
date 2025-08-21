local sched, err = require("cgemma").scheduler(config().scheduler)
if not sched then
  ngx.log(ngx.ERR, "cgemma error: ", err)
end

local function gemma_inst()
  if not worker_gemma_inst then
    local gemma_cfg = config().gemma
    gemma_cfg.scheduler = sched
    local gemma, err = require("cgemma").new(gemma_cfg)
    if not gemma then
      ngx.log(ngx.ERR, "cgemma error: ", err)
      ngx.exit(ngx.HTTP_INTERNAL_SERVER_ERROR)
    end
    worker_gemma_inst = gemma
  end
  return worker_gemma_inst
end

local function send_resp(ws, data)
  return ws:send_text(require("cjson.safe").encode(data))
end

function gemma_loop(ws)
  local session = assert(gemma_inst():session(config().session))
  local ok, err = session:load("dump.bin")
  if not ok then
    ngx.log(ngx.ERR, "cgemma error: ", err)
  end
  assert(send_resp(ws, {role = "system", text = "New chat session started!", vlm_mode = config().vlm_mode}))
  while session:ready() do
    local data, tp, err = ws:recv_frame()
    while err == "again" do
      local frag, ct
      frag, ct, err = ws:recv_frame()
      assert(ct == "continuation", err)
      data = data..frag
    end
    if tp == "text" then
      local msg = assert(require("cjson.safe").decode(data))
      if msg.role == "user" then
        local embedded_image
        if config().vlm_mode and msg.image then
          local img_buf = ngx.decode_base64(msg.image)
          if img_buf then
            local img = require("vips").Image.new_from_buffer(img_buf)
            if img then
              assert(send_resp(ws, {role = "gemma", pos = -2, prompt_size = 0}))
              img = img:resize(config().vlm_mode.resize_to / img:width(), {vscale = config().vlm_mode.resize_to / img:height(), kernel = "linear"})
              local ppm = require("vips").Target.new_to_memory()
              img:write_to_target(ppm, ".ppm")
              embedded_image = assert(gemma_inst():embed_image(ppm:vobject():get("blob")))
            end
          end
        end
        if embedded_image or msg.text then
          assert(send_resp(ws, {role = "gemma", pos = -1, prompt_size = 0}))
          local function stream_fn(token, pos, prompt_size)
            local bytes, err = send_resp(ws, {role = "gemma", token = token, pos = pos, prompt_size = prompt_size})
            if not bytes then
              ngx.log(ngx.ERR, "websocket error: ", err)
              return false
            end
            return true
          end
          if embedded_image then
            assert(session(embedded_image, msg.text or "", stream_fn))
          else
            assert(session(msg.text, stream_fn))
          end
        end
      else
        assert(send_resp(ws, {role = "system", text = "Unsupported role!"}))
      end
    elseif tp == "ping" then
      assert(wb:send_pong())
    elseif tp == "close" then
      return
    elseif tp ~= "pong" then
      ws:send_close()
      assert(not err, err)
      return
    end
  end
  assert(send_resp(ws, {role = "system", text = "Exceed the maximum number of tokens!"}))
end
