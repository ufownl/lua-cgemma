local sched, err = require("cgemma").scheduler(config().scheduler)
if not sched then
  ngx.log(ngx.ERR, "cgemma error: ", err)
end

function gemma_inst()
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

function gemma_loop()
  local session, err = gemma_inst():session(config().session)
  if not session then
    ngx.log(ngx.ERR, "cgemma error: ", err)
    return ngx.HTTP_INTERNAL_SERVER_ERROR
  end
  local ok, err = session:load("dump.bin")
  if not ok then
    ngx.log(ngx.ERR, "cgemma error: ", err)
  end
  local ws, err = require("resty.websocket.server"):new(config().websocket)
  if not ws then
    ngx.log(ngx.ERR, "websocket error: ", err)
    return ngx.HTTP_CLOSE
  end
  local bytes, err = ws:send_text(require("cjson.safe").encode({
    role = "system",
    text = "New chat session started!",
    vlm_mode = config().vlm_mode
  }))
  if not bytes then
    ngx.log(ngx.ERR, "websocket error: ", err)
    return ngx.OK
  end
  while session:ready() do
    local data, tp, err = ws:recv_frame()
    while err == "again" do
      local frag, ct
      frag, ct, err = ws:recv_frame()
      if ct ~= "continuation" then
        ngx.log(ngx.ERR, "websocket error: ", err)
        ws:send_close()
        return ngx.OK
      end
      data = data..frag
    end
    if tp == "text" then
      local msg = require("cjson.safe").decode(data)
      if not msg or not msg.role then
        ngx.log(ngx.ERR, "protocol error: unknown format")
        ws:send_close()
        return ngx.OK
      end
      if msg.role == "user" then
        local embedded_image
        if config().vlm_mode and msg.image then
          local img_buf = ngx.decode_base64(msg.image)
          if img_buf then
            local img = require("vips").Image.new_from_buffer(img_buf)
            if img then
              img = img:resize(config().vlm_mode.resize_to / img:width(), {vscale = config().vlm_mode.resize_to / img:height(), kernel = "linear"})
              local ppm = require("vips").Target.new_to_memory()
              img:write_to_target(ppm, ".ppm")
              local img_tks, err = gemma_inst():embed_image(ppm:vobject():get("blob"))
              if not img_tks then
                ngx.log(ngx.ERR, "cgemma error: ", err)
                ws:send_close()
                return ngx.OK
              end
              embedded_image = img_tks
            end
          end
        end
        if embedded_image or msg.text then
          local function stream_fn(token, pos, prompt_size)
            local bytes, err = ws:send_text(require("cjson.safe").encode({
              role = "gemma",
              token = token,
              pos = pos,
              prompt_size = prompt_size
            }))
            if not bytes then
              ngx.log(ngx.ERR, "websocket error: ", err)
              return false
            end
            return true
          end
          local ok, err
          if embedded_image then
            ok, err = session(embedded_image, msg.text or "", stream_fn)
          else
            ok, err = session(msg.text, stream_fn)
          end
          if not ok then
            ngx.log(ngx.ERR, "cgemma error: ", err)
            ws:send_close()
            return ngx.OK
          end
        end
      else
        local bytes, err = ws:send_text(require("cjson.safe").encode({
          role = "system",
          text = "Unsupported role!"
        }))
        if not bytes then
          ngx.log(ngx.ERR, "websocket error: ", err)
          return ngx.OK
        end
      end
    elseif tp == "ping" then
      local bytes, err = wb:send_pong()
      if not bytes then
        ngx.log(ngx.ERR, "websocket error: ", err)
        return ngx.OK
      end
    elseif tp == "close" then
      return ngx.OK
    elseif tp ~= "pong" then
      if err then
        ngx.log(ngx.ERR, "websocket error: ", err)
      end
      ws:send_close()
      return ngx.OK
    end
  end
  ws:send_text(require("cjson.safe").encode({
    role = "system",
    text = "Exceed the maximum number of tokens!"
  }))
  return ngx.OK
end
