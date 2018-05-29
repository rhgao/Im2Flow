--[[
    This data loader is adapted from https://github.com/phillipi/pix2pix/blob/master/data/donkey_folder.lua

    This data loader is a modified version of the one from dcgan.torch
    (see https://github.com/soumith/dcgan.torch/blob/master/data/donkey_folder.lua).
    Copyright (c) 2016, Deepak Pathak [See LICENSE file for details]
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.
]]--

require 'image'
paths.dofile('dataset.lua')
print(os.getenv('DATA_ROOT'))
opt.data = os.getenv('DATA_ROOT')
if not paths.dirp(opt.data) then
    error('Did not find directory: ' .. opt.data)
end

-- a cache file of the training metadata (if doesnt exist, will be created)
local cache = "cache"
local cache_prefix = opt.data:gsub('/', '_')
os.execute('mkdir -p cache')
local trainCache = paths.concat(cache, cache_prefix .. '_trainCache.t7')

--------------------------------------------------------------------------------------------
local input_nc = opt.input_nc -- input channels
local loadSize   = {input_nc, opt.loadSize}
local sampleSize = {input_nc, opt.fineSize}

local preprocessImage = function(im)
  im = image.scale(im, loadSize[2], loadSize[2])
  local perm = torch.LongTensor{1, 2, 3}
  im = im:index(1, perm)--:mul(256.0): brg, rgb
  im = im:mul(2):add(-1)

  assert(im:max()<=1,"A: badly scaled inputs")
  assert(im:min()>=-1,"A: badly scaled inputs")
  
  local oW = sampleSize[2]
  local oH = sampleSize[2]
  local iH = im:size(2)
  local iW = im:size(3)
  
  if iH~=oH then     
    h1 = math.ceil(torch.uniform(1e-2, iH-oH))
  end
  
  if iW~=oW then
    w1 = math.ceil(torch.uniform(1e-2, iW-oW))
  end
  if iH ~= oH or iW ~= oW then 
    im = image.crop(im, w1, h1, w1 + oW, h1 + oH)
  end
  
  if opt.flip == 1 and torch.uniform() > 0.5 then 
    im = image.hflip(imA)
  end
  
  return im
end

--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path)
   collectgarbage()
   local im =  image.load(path, 3, 'float')
   im2input = preprocessImage(im)
   return im2input
end

--------------------------------------
-- trainLoader
print('trainCache', trainCache)
--if paths.filep(trainCache) then
--   print('Loading train metadata from cache')
--   trainLoader = torch.load(trainCache)
--   trainLoader.sampleHookTrain = trainHook
--   trainLoader.loadSize = {input_nc, opt.loadSize, opt.loadSize}
--   trainLoader.sampleSize = {input_nc+output_nc, sampleSize[2], sampleSize[2]}
--   trainLoader.serial_batches = opt.serial_batches
--   trainLoader.split = 100
--else
print('Creating train metadata')
--   print(opt.data)
print('serial batch:, ', opt.serial_batches)
trainLoader = dataLoader{
    paths = {opt.data},
    loadSize = {input_nc, loadSize[2], loadSize[2]},
    sampleSize = {input_nc, sampleSize[2], sampleSize[2]},
    split = 100,
    serial_batches = opt.serial_batches, 
    verbose = true
 }
--   print('finish')
--torch.save(trainCache, trainLoader)
--print('saved metadata cache at', trainCache)
trainLoader.sampleHookTrain = trainHook
--end
collectgarbage()

-- do some sanity checks on trainLoader
do
   local class = trainLoader.imageClass
   local nClasses = #trainLoader.classes
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")
end