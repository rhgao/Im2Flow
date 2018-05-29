require 'torch'
require 'nn'
util = paths.dofile('util.lua')

local content = {}

function content.defineResnet18(content_layer,opt)
  local contentFunc = nn.Sequential()
  require 'util/Resnet_preprocess'
  cnn = util.load('./model/resnet-18_motion.t7',opt)
  contentFunc:add(nn.SpatialUpSamplingBilinear({oheight=224, owidth=224}))
  contentFunc:add(nn.Resnet_postprocess())
  for i = 1, #cnn do
    local layer = cnn:get(i):clone()
    local layer_type = torch.typename(layer)
    contentFunc:add(layer)
    if i==6 or layer_type == 'nn.View' then
      print("Setting up content layer: ", layer_type)
      break
    end
  end
  cnn = nil
  collectgarbage()
  contentFunc:cuda()
  print(contentFunc)
  return contentFunc
end

function content.defineContent(content_loss, layer_name, opt)
  -- print('content_loss_define', content_loss)
  if content_loss == 'pixel' or content_loss == 'none' then
    return nil
  elseif content_loss == 'resnet18' then
    return content.defineResnet18(layer_name,opt)
  else
    print("unsupported content loss")
    return nil
  end
end

function content.lossUpdate(criterionContent, real_source, fake_target, contentFunc, loss_type, weight)
  if loss_type == 'pixel' or loss_type == 'none' then
    local errCont = 0.0
    local df_d_content = torch.zeros(fake_target:size())
    return errCont, df_d_content
  elseif loss_type == 'resnet18' then
    local f_real = contentFunc:forward(real_source):clone()
    local f_fake = contentFunc:forward(fake_target):clone()
    local errCont = criterionContent:forward(f_fake, f_real) * weight
    local df_do_tmp = criterionContent:backward(f_fake, f_real) * weight
    local df_do_content = contentFunc:updateGradInput(fake_target, df_do_tmp)
    return errCont, df_do_content
  else error("unsupported content loss")
  end
end

return content
