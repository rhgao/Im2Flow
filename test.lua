-- code derived from https://github.com/phillipi/pix2pix
-- code derived from https://github.com/soumith/dcgan.torch

require 'image'
require 'nn'
require 'nngraph'
util = paths.dofile('util/util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
    DATA_ROOT = 'demo_images',      -- path to test images 
    batchSize = 1,                  -- # images in batch
    loadSize = 256,                 -- scale images to this size
    fineSize = 256,                 --  then crop to this size
    flip=0,                         -- horizontal mirroring data augmentation
    gpu = 1,                        -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
    how_many = 'all',               -- how many test images to run (set to all to run on every image found in the data/phase folder)
    aspect_ratio = 1.0,             -- aspect ratio of result images
    input_nc = 3,                   -- #  of input image channels
    output_nc = 3,                  -- #  of output image channels
    serial_batches = 1,             -- if 1, takes images in order to make batches, otherwise takes them randomly
    serial_batch_iter = 1,          -- iter into serial image list
    cudnn = 1,                      -- set to 0 to not use cudnn (untested)
    model_path = './Im2Flow.t7',    -- set path of Im2Flow model
    results_dir='./results/',       -- saves results here
}


-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
opt.nThreads = 1 -- test only works with 1 thread...
print(opt)

opt.manualSeed = torch.random(1, 10000) -- set seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

local data_loader = paths.dofile('data/data_test.lua')
print('#threads...' .. opt.nThreads)
local data = data_loader.new(opt.nThreads, opt)
print("Dataset Size: ", data:size())

-- translation directions
----------------------------------------------------------------------------

local input = torch.FloatTensor(opt.batchSize,3,opt.fineSize,opt.fineSize)
local target = torch.FloatTensor(opt.batchSize,3,opt.fineSize,opt.fineSize)

local netG = util.load(opt.model_path, opt)
netG:evaluate()

print(netG)


function TableConcat(t1,t2)
    for i=1,#t2 do
        t1[#t1+1] = t2[i]
    end
    return t1
end

if opt.how_many=='all' then
    opt.how_many=data:size()
end
opt.how_many=math.min(opt.how_many, data:size())

local filepaths = {} -- paths to images tested on
for n=1,math.floor(opt.how_many/opt.batchSize) do
    print('processing batch ' .. n)
    
    local input, filepaths_curr = data:getBatch()
    filepaths_curr = util.basename_batch(filepaths_curr)
    print('filepaths_curr: ', filepaths_curr)
    
    
    if opt.gpu > 0 then
        input = input:cuda()
    end
    
    output = util.deprocess_batch(netG:forward(input))
    input = util.deprocess_batch(input):float()
    output = output:float()
    target = util.deprocess_batch(target):float()
    
    paths.mkdir(opt.results_dir)
    paths.mkdir(paths.concat(opt.results_dir,'input'))
    paths.mkdir(paths.concat(opt.results_dir,'output'))
    -- print(input:size())
    -- print(output:size())
    -- print(target:size())
    for i=1, opt.batchSize do
        image.save(paths.concat(opt.results_dir,'input',filepaths_curr[i]), image.scale(input[i],input[i]:size(2),input[i]:size(3)/opt.aspect_ratio))
        image.save(paths.concat(opt.results_dir,'output',filepaths_curr[i]), image.scale(output[i],output[i]:size(2),output[i]:size(3)/opt.aspect_ratio))
    end
    print('Saved images to: ', opt.results_dir)
    filepaths = TableConcat(filepaths, filepaths_curr)
end

-- make webpage
io.output(paths.concat(opt.results_dir, 'index.html'))

io.write('<table style="text-align:center;">')

io.write('<tr><td>Image #</td><td>Input</td><td>Output</td></tr>')
for i=1, #filepaths do
    io.write('<tr>')
    io.write('<td>' .. filepaths[i] .. '</td>')
    io.write('<td><img src="./input/' .. filepaths[i] .. '"/></td>')
    io.write('<td><img src="./output/' .. filepaths[i] .. '"/></td>')
    io.write('</tr>')
end

io.write('</table>')
