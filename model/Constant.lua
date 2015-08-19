local Constant, parent = torch.class('nn.Constant', 'nn.Module')

function Constant:__init( const_weights )
   parent.__init(self)
   self.weight = const_weights:clone()
   self.gradWeight = const_weights:clone():zero()
   self.output = self.weight
end

function Constant:updateGradInput(input, gradOutput)
   self.gradInput = input:clone():zero()
   return self.gradInput
end

function Constant:accGradParameters(input, gradOutput, scale)
   self.gradWeight:add(scale, gradOutput)
end

function Constant:accUpdateGradParameters(input, gradOutput, lr)
   local gradWeight = self.gradWeight
   self.gradWeight = self.weight
   self:accGradParameters(gradOutput, -lr)
   self.gradWeight = gradWeight
end

-- we do not need to accumulate parameters when sharing
Constant.sharedAccUpdateGradParameters = Constant.accUpdateGradParameters

function Constant:__tostring__()
  return torch.type(self) ..
      tostring(self.weight:size())
end

return Constant
