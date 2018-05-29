-- define nn module for Resnet postprocessing
local Resnet_postprocess, parent = torch.class('nn.Resnet_postprocess', 'nn.Module')

function Resnet_postprocess:__init()
	parent.__init(self)
end

function Resnet_postprocess:updateOutput(input)
    self.output = input:add(1):mul(0.5)
	local mean_pixel = torch.FloatTensor({0.485, 0.456, 0.406})
	local std_pixel = torch.FloatTensor({0.229, 0.224, 0.225})
	mean_pixel = mean_pixel:reshape(1,3,1,1)
	std_pixel = std_pixel:reshape(1,3,1,1)
	mean_pixel = mean_pixel:repeatTensor(input:size(1), 1, input:size(3), input:size(4)):cuda()
	std_pixel = std_pixel:repeatTensor(input:size(1), 1, input:size(3), input:size(4)):cuda()
	self.output:add(-1, mean_pixel):cdiv(std_pixel)
	return self.output
end

function Resnet_postprocess:updateGradInput(input, gradOutput)
	local std_pixel = torch.FloatTensor({0.229, 0.224, 0.225})
	std_pixel = std_pixel:reshape(1,3,1,1)
	std_pixel = std_pixel:repeatTensor(input:size(1), 1, input:size(3), input:size(4)):cuda()
	self.gradInput = gradOutput:cdiv(std_pixel):mul(0.5)
	return self.gradInput
end