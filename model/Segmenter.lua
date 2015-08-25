--[[
 Soft segmentation unit, which uses dynamic programming to average (or 
 max) over all possible decompositions
--]]

local Segmenter = {}

function Segmenter.segnet(length, batch_size, max_window, conv_dim, mode, net_type)
   -- length = length of sentences/words (zero padded to be of same length)
   -- input_size = character embedding_size
   -- max_window = window size
   -- conv_dim = RNN state size
   -- mode in {'max', 'mixture'}
   -- net_type in {'lstm', 'rnn'}
   print('params', length, batch_size, max_window, conv_dim, mode, net_type)
   
   local all_convs = nn.Identity()()
   
   -- then, define the recurrent units
   local units = {}
   local output
   if net_type == 'lstm' then -- LSTM version
      -- output is {h: batch_size x 1 x conv_dim, c: batch_size x 1 x  conv_dim}
      -- padding units
      for i = 1, max_window do
        local padd_h = nn.Constant(torch.Tensor(batch_size, 1, conv_dim):zero())
        local padd_c = nn.Constant(torch.Tensor(batch_size, 1, conv_dim):zero())
        table.insert(units, {padd_h(all_convs), padd_c(all_convs)} )
      end
      -- recurrent "dynamic" network
      for i = max_window+1, length do
        --print('unit ',i)
        local pre_h = {}
        local pre_c = {}
        for j = 1, max_window do
          --print('input ',i-j)
          table.insert(pre_h, units[i-j][1])
          table.insert(pre_c, units[i-j][2])
        end
        -- define inputs
        local prev_h = nn.JoinTable(2)(pre_h)    -- batch_size * max_window * conv_dim
        local prev_c = nn.JoinTable(2)(pre_c)
        local ngrams = nn.Select(2, i - max_window + 1)(all_convs)
        -- LSTM
        local flat_x = nn.View(batch_size * max_window, conv_dim)(ngrams)
        local Wx = nn.View(batch_size, max_window, 4 * conv_dim)(
                        nn.Linear(conv_dim, 4 * conv_dim)(flat_x))
        local flat_h = nn.View(batch_size * max_window, conv_dim)(prev_h)
        local Uh = nn.View(batch_size, max_window, 4 * conv_dim)(
                        nn.Linear(conv_dim, 4 * conv_dim)(flat_h))
        local all_input_sums = nn.CAddTable()({Wx, Uh})
        -- gates
        local sigmoid_chunk = nn.Narrow(3, 1, 3 * conv_dim)(all_input_sums)
        sigmoid_chunk = nn.Sigmoid()(sigmoid_chunk)
        local in_gate = nn.Narrow(3, 1, conv_dim)(sigmoid_chunk)
        local forget_gate = nn.Narrow(3, conv_dim + 1, conv_dim)(sigmoid_chunk)
        local out_gate = nn.Narrow(3, 2 * conv_dim + 1, conv_dim)(sigmoid_chunk)
        -- transform
        local in_transform = nn.Narrow(3, 3 * conv_dim + 1, conv_dim)(all_input_sums)
        in_transform = nn.Tanh()(in_transform)
        -- new memory cell
        local next_c = nn.CAddTable()({
         nn.CMulTable()({forget_gate, prev_c}),
         nn.CMulTable()({in_gate, in_transform})
        }) -- batch_size * max_window * conv_dim
        -- gated cells form the output
        local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
        if mode == 'max' then
          -- dynamic programming with max pooling
          local max_c = nn.TemporalMaxPooling(max_window)(next_c)
          local max_h = nn.TemporalMaxPooling(max_window)(next_h)
          units[i] = {max_h, max_c}
        else
          -- dynamic programming with averaging
          -- first, we compute the segmentation weights
          local Vh = nn.View(batch_size * max_window, 1, conv_dim)(
                          nn.Linear(conv_dim, conv_dim)(flat_h))
          local mat_x = nn.View(batch_size * max_window, conv_dim, 1)(flat_x)
          local pre_weights = nn.View(batch_size, max_window)(
                                 nn.MM()({Vh, mat_x}))
          local mix_weights = nn.View(batch_size, 1, max_window)(
                                 nn.SoftMax()(pre_weights))
          -- then, we apply the weights to h and c
          local av_c = nn.View(batch_size, 1, conv_dim)(
                            nn.MM()({mix_weights, next_c}))
          local av_h = nn.View(batch_size, 1, conv_dim)(
                            nn.MM()({mix_weights, next_h}))
          units[i] = {av_h, av_c}
        end
      end
      output = nn.View(batch_size, conv_dim)(units[length][1])
   elseif net_type == 'rnn' then -- Linear RNN version
      -- padding units
      for i = 1, max_window do
        local padd_h = nn.Constant(torch.Tensor(batch_size, 1, conv_dim):zero())
        table.insert(units, padd_h(all_convs))
      end
      for i = max_window+1, length do
        -- print('unit ',i)
        local pre_h = {}
        for j = 1, max_window do
          --  print('input ',i-j)
          table.insert(pre_h, units[i-j])
        end
        local prev_h = nn.JoinTable(2)(pre_h)    -- batch_size * max_window * conv_dim
        local ngrams = nn.Select(2, i - max_window + 1)(all_convs)
        local next_h = nn.CAddTable()({prev_h, ngrams})
        if mode == 'max' then
          -- dynamic programming with max pooling
          local max_h = nn.TemporalMaxPooling(max_window)(next_h)
          units[i] = max_h
        else
          -- dynamic programming with averaging
          -- first, we compute the segmentation weights
          local flat_h = nn.View(batch_size * max_window, conv_dim)(prev_h)
          local Vh = nn.View(batch_size * max_window, 1, conv_dim)(
                          nn.Linear(conv_dim, conv_dim)(flat_h))
          local flat_x = nn.View(batch_size * max_window, conv_dim)(ngrams)
          local mat_x = nn.View(batch_size * max_window, conv_dim, 1)(flat_x)
          local pre_weights = nn.View(batch_size, max_window)(
                                 nn.MM()({Vh, mat_x}))
          local mix_weights = nn.View(batch_size, 1, max_window)(
                                 nn.SoftMax()(pre_weights))
          -- then, we apply the weights to h
          local av_h = nn.View(batch_size, 1, conv_dim)(
                            nn.MM()({mix_weights, next_h}))
          units[i] = av_h
        end
      end
      output = nn.View(batch_size, conv_dim)(units[length])
   end
   
   
   return nn.gModule({all_convs}, {output})
end

return Segmenter
