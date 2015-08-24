--[[
Evaluates a trained model

Much of the code is borrowed from the following implementations
https://github.com/karpathy/char-rnn
https://github.com/wojzaremba/lstm
]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'util.Squeeze'
require 'util.misc'

BatchLoader = require 'util.BatchLoaderUnk'
model_utils = require 'util.model_utils'

local stringx = require('pl.stringx')

cmd = torch.CmdLine()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/ptb','data directory. Should contain train.txt/valid.txt/test.txt with input data')
cmd:option('-savefile', 'final-results/lm_results.t7', 'save results to')
cmd:option('-model', 'final-results/en-large-word-model.t7', 'model checkpoint file')
-- GPU/CPU
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt2 = cmd:parse(arg)
if opt2.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt2.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt2.gpuid + 1)
end

if opt2.cudnn == 1 then
    assert(opt2.gpuid >= 0, 'GPU must be used if using cudnn')
    print('using cudnn')
    require 'cudnn'
end

HighwayMLP = require 'model.HighwayMLP'
TDNN = require 'model.TDNN'
LSTMTDNN = require 'model.LSTMTDNN'
BoW = require 'model.BoW'
Segmenter = require 'model.Segmenter'
Constant = require 'model.Constant'

checkpoint = torch.load(opt2.model)
opt = checkpoint.opt
protos = checkpoint.protos
--opt.batch_size = 1
print('opt: ')
print(opt)
print('val_losses: ')
print(checkpoint.val_losses)
-- recreate the data loader class, with batchsize = 1
loader = BatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, opt.padding, 
                                    opt.max_word_l, opt.max_factor_l, opt.max_window,
                                    opt.use_morpho, opt.use_segmenter)

idx2word, word2idx, idx2char, char2idx = table.unpack(checkpoint.vocab)

print('Word vocab size: ' .. #loader.idx2word .. ', Char vocab size: ' .. #loader.idx2char
	    .. ', Max word length (incl. padding): ', loader.max_word_l)

-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
end

-- training criterion (negative log likelihood)
protos.criterion = nn.ClassNLLCriterion()

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

params, grad_params = model_utils.combine_all_parameters(protos.rnn)

print('number of parameters in the model: ' .. params:nElement())

-- for easy switch between using words/chars (or both)
function get_input(x, x_char, t, prev_states)
    local u = {}
    if opt.use_chars == 1 or opt.use_morpho == 1 then table.insert(u, x_char[{{},t}]) end
    if opt.use_words == 1 then table.insert(u, x[{{},t}]) end

    for i = 1, #prev_states do table.insert(u, prev_states[i]) end
    --print(u)
    return u
end

-- evaluate the loss over an entire split
function eval_split(split_idx, max_batches)
    print('evaluating loss over split index ' .. split_idx)
    local n = loader.split_sizes[split_idx]
    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_idx) -- move batch iteration pointer for this split to front
    local loss = 0
    local rnn_state = {[0] = init_state}    
    if split_idx<=2 then -- batch eval        
	for i = 1,n do -- iterate over batches in the split
	    -- fetch a batch
	    local x, y, x_char = loader:next_batch(split_idx)
	    if opt.gpuid >= 0 then -- ship the input arrays to GPU
		-- have to convert to float because integers can't be cuda()'d
		x = x:float():cuda()
		y = y:float():cuda()
		x_char = x_char:float():cuda()
	    end
	    -- forward pass
	    for t=1,opt.seq_length do
		clones.rnn[t]:evaluate() -- for dropout proper functioning
		local lst = clones.rnn[t]:forward(get_input(x, x_char, t, rnn_state[t-1]))
		rnn_state[t] = {}
		for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
		prediction = lst[#lst] 
		loss = loss + clones.criterion[t]:forward(prediction, y[{{}, t}])
	    end
	    -- carry over lstm state
	    rnn_state[0] = rnn_state[#rnn_state]
	    -- print(i .. '/' .. n .. '...')
	end
	loss = loss / opt.seq_length / n
    else -- full eval on test set
        local x, y, x_char = loader:next_batch(split_idx)
	if opt.gpuid >= 0 then -- ship the input arrays to GPU
	    -- have to convert to float because integers can't be cuda()'d
	    x = x:float():cuda()
	    y = y:float():cuda()
	    x_char = x_char:float():cuda()
	end
	protos.rnn:evaluate() -- just need one clone
	for t = 1, x:size(2) do

	    local lst = protos.rnn:forward(get_input(x, x_char, t, rnn_state[0]))
	    rnn_state[0] = {}
	    for i=1,#init_state do table.insert(rnn_state[0], lst[i]) end
	    prediction = lst[#lst] 
	    local tok_perp = protos.criterion:forward(prediction, y[{{},t}])
	    loss = loss + tok_perp
	end
	loss = loss / x:size(2)
    end    
    local perp = torch.exp(loss)    
    return perp
end

clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end


function eval_split(split_idx, max_batches)
    print('evaluating loss over split index ' .. split_idx)
    local n = loader.split_sizes[split_idx]
    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_idx) -- move batch iteration pointer for this split to front
    local loss = 0
    local rnn_state = {[0] = init_state}    
    if split_idx<=2 then -- batch eval        
	for i = 1,n do -- iterate over batches in the split
	    -- fetch a batch
	    local x, y, x_char = loader:next_batch(split_idx)
	    if opt.gpuid >= 0 then -- ship the input arrays to GPU
		-- have to convert to float because integers can't be cuda()'d
		x = x:float():cuda()
		y = y:float():cuda()
		x_char = x_char:float():cuda()
	    end
	    -- forward pass
	    for t=1,opt.seq_length do
		clones.rnn[t]:evaluate() -- for dropout proper functioning
		local lst = clones.rnn[t]:forward(get_input(x, x_char, t, rnn_state[t-1]))
		rnn_state[t] = {}
		for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
		prediction = lst[#lst] 
		loss = loss + clones.criterion[t]:forward(prediction, y[{{}, t}])
	    end
	    -- carry over lstm state
	    rnn_state[0] = rnn_state[#rnn_state]
	    -- print(i .. '/' .. n .. '...')
	end
	loss = loss / opt.seq_length / n
    else -- full eval on test set
        local x, y, x_char = loader:next_batch(split_idx)
	if opt.gpuid >= 0 then -- ship the input arrays to GPU
	    -- have to convert to float because integers can't be cuda()'d
	    x = x:float():cuda()
	    y = y:float():cuda()
	    x_char = x_char:float():cuda()
	end
	protos.rnn:evaluate() -- just need one clone
	for t = 1, x:size(2) do

	    local lst = protos.rnn:forward(get_input(x, x_char, t, rnn_state[0]))
	    rnn_state[0] = {}
	    for i=1,#init_state do table.insert(rnn_state[0], lst[i]) end
	    prediction = lst[#lst] 
	    local tok_perp = protos.criterion:forward(prediction, y[{{},t}])
	    loss = loss + tok_perp
	end
	loss = loss / x:size(2)
    end    
    local perp = torch.exp(loss)    
    return perp
end

print('trusted valid',  eval_split(2))
print('trusted test',  eval_split(3))

total_perp, token_loss, token_count = eval_split_full(3)
print(total_perp)
test_results = {}
test_results.perp = total_perp
test_results.token_loss = token_loss
test_results.token_count = token_count
test_results.vocab = {idx2word, word2idx, idx2char, char2idx}
test_results.opt = opt
test_results.val_losses = checkpoint.val_losses
torch.save(opt2.savefile, test_results)
collectgarbage()
