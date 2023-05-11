--VARIABLE DESCRIPTIONS:
--long_term_memory: memory={index=int, inputs={index}, outputs={index}, error=float, connected_memories={index}}
--short_term_memory: (position in long term memory paired with the value at that location)x20 Ex: 10, 4, nil, 0.7 (the error remembered from the 10th memory in the list)
--network: node={index=int, input_nodes={index}, output_nodes={index}, output=float, weights={float}, biases={float} inputs_recieved=int}
--queue_start: table of network indices connected to input nodes
--queue_end: table of network indices connected to output nodes

local data = {}--datapoint={inputs={}, outputs={}} (training data)
local layers = {5,10,11,12,10,3}--amount of input nodes, amount of nodes in each hidden layer, amount of output nodes

--GENERATE A NEURAL NETWORK:
function generate(layers)
    local brain = {network={}, short_term_memory={}, long_term_memory={}, queue_start={}, queue_end={}, last_modified_weight={node_index=nil, weight_index=nil, previous_val=nil}, last_modified_bias={node_index=nil, previous_val=nil}}
    local last_layer = {}
    local current_layer = {}
    for layer_index, node_amount in ipairs(layers) do 
        for i=1,node_amount do
            node_index = #brain.network+1
            node = {index=node_index, input_nodes={}, output_nodes={}, output=0, weights={}, weight_sensitivities={}, bias=math.random(0,100)/100, bias_sensitivity=0, inputs_recieved=0}
            if layer_index > 1 then
                for j,node_index2 in ipairs(last_layer) do
                    node2 = brain.network[node_index2]
                    node2.output_nodes[#node2.output_nodes+1] = node_index
                    node.input_nodes[#node.input_nodes+1] = node_index2
                    node.weights[#node.weights+1] = math.random(0,100)/100
                    node.weight_sensitivities[#node.weights+1] = 0
                    brain.network[node_index2] = node2
                end
            end
            if layer_index == 1 then
                brain.queue_start[#brain.queue_start+1] = node_index
            elseif layer_index == #layers then 
                brain.queue_end[#brain.queue_end+1] = node_index
            end
            brain.network[node_index] = node
            current_layer[#current_layer+1] = node_index
        end
        last_layer = current_layer
        current_layer = {}
    end
    return brain
end

--RUN A NEURAL NETWORK:
function think(brain, inputs)
    local network = brain.network
    local queue = brain.queue_start
    local outputs = {}
    local i = 1
    while #queue > 0 do
        node_index = queue[1]
        --print("CURRENT NODE: "..node_index)
        local node = network[node_index]
        if node.inputs_recieved == #node.input_nodes then
            if #node.input_nodes > 0 then
                --apply bias:
                node.output = node.output + node.bias
                --add up all the inputs, applying their weights:
                for j, input_node_index in ipairs(node.input_nodes) do
                    local input_node = network[input_node_index]
                    local input = input_node.output
                    local weight = node.weights[j]
                    node.output = node.output+(input*weight)
                end
                node.output = math.tanh(node.output)
                network[node_index] = node
            else
                node.output = math.tanh(inputs[i]) --inject inputs into first layer
            end
            --print("OUTPUT: "..node.output)

            --add the next node to the queue:
            if #node.output_nodes > 0 then
                for j, output_node_index in ipairs(node.output_nodes) do
                    local output_node = network[output_node_index]
                    output_node.inputs_recieved = output_node.inputs_recieved + 1
                    network[output_node_index] = output_node
                    --check if output node is already queued:
                    local already_done = false
                    for k=1,#queue do
                        local other_node_index = queue[k]
                        if other_node_index ~= nil then
                            --print(output_node_index.." : "..other_node_index)
                            if output_node_index == other_node_index then already_done = true break end
                        end
                    end
                    if not already_done then table.insert(queue, output_node_index) end
                end
                --print("OUTPUT NODES: "..dump(node.output_nodes))
            else
                --print("final output")
                outputs[#outputs+1] = node.output
            end
        end
        --print("QUEUE: "..dump(queue))
        table.remove(queue, 1)--queue[i] = nil
        i = i + 1
    end
    return network, outputs
end

--MAKE A NEURAL NETWORK LEARN FROM MISTAKES (reinforcement learning)
function learn(brain, last_error, current_error)
    local network = brain.network
    local smallnum = 0.001
    if network.last_modified_weight.node_index then
        local modified_node_index, modified_weight_index, previous_val = brain.last_modified_weight
        local weight_last_modified = network[modified_node_index].weights[modified_weight_index]
        local partial_derivative = (current_error-last_error)/smallnum--find slope with respect to weight
        brain.network[modified_node_index].weights[modified_weight_index] = previous_val - (partial_derivative*1)--reset weight and push toward slope
        brain.network[modified_node_index].weight_sensitivities[modified_weight_index] = math.abs(partial_derivative)--assign sensitivity value to weight
        brain.last_modified_weight = {node_index=nil, weight_index=nil, previous_val=nil}
    elseif network.last_modified_bias.node_index then
        local modified_node_index, previous_val = brain.last_modified_bias
        local bias_last_modified = network[modified_bias_index].bias
        local partial_derivative = (current_error-last_error)/smallnum--find slope with respect to bias
        brain.network[modified_node_index].bias = previous_val - (partial_derivative*1)--reset weight and push toward slope
        brain.network[modified_node_index].bias_sensitivity = math.abs(partial_derivative)--assign sensitivity value to bias
        brain.last_modified_bias = {node_index=nil, previous_val=nil}
    end

    --select random weight/bias to modify (more sensitive weights/biases are more likely to be selected):
    local node_index
    local weight_index
    local bias_index
    local rand = 0
    for i, node in ipairs(network) do
        for j, weight in ipairs(node.weights) do
            rand = rand + node.weight_sensitivities[j]
        end
        rand = rand + node.bias_sensitivity
    end
    rand = math.random(math.floor(rand*100000))/100000

    local weight_selected
    local bias_selected
    local index = 0
    for i, node in ipairs(network) do
        for j, weight in ipairs(node.weights) do
            index = index + node.weight_sensitivities[j]
            if index >= rand then
                node_index = i
                weight_index = j
                weight_selected = true
                break
            end
        end
        index = index + node.bias_sensitivity
        if index >= rand then
            node_index = i
            bias_selected = true
            break
        end
        if weight_selected or bias_selected then break end
    end

    --modify selected weight/bias:
    if weight_selected then
        local weight = network[node_index].weights[weight_index]
        brain.network[node_index].weights[weight_index] = weight + smallnum
        brain.last_modified_weight = {node_index, weight_index, weight}
    else
        local bias = network[node_index].bias
        brain.network[node_index].bias = bias + smallnum
        brain.last_modified_bias = {node_index, bias}
    end
    return brain
    --idea: run every frame, but rewind a frame every time you change a weight, so its training for the best possible output for a single frame
    --that would require an error for each frame though
end

--TRAIN A NEURAL NETWORK (backprop)
function train(brain, data, iterations) --needs to be fixed
    local network = brain.network
    local queue_start = brain.queue_start
    local queue_end = brain.queue_end
    function loss(brain, map)
        --compare output values to desired output values from map:
        local network2, outputs = think(brain, map.inputs)
        local loss = 0
        for i,output in ipairs(outputs) do
            loss = loss + (map.outputs[i]-output)^2
        end
        return loss
    end
    for i=1,iterations do
        for j, map in ipairs(data) do
            local queue = queue_end
            local step_size = 1
            local smallnum = 0.0001
            for k, node_index in ipairs(queue) do
                --optimize weights:
                local node = network[node_index]
                local new_weights = {}
                for weight_index, weight in ipairs(node.weights) do
                    local brain_copy = brain
                    local weight_copy = weight

                    local loss1 = loss(brain, map)
                    brain_copy.network[node_index].weights[weight_index] = weight_copy + smallnum
                    local loss2 = loss(brain_copy, map)

                    local partial_derivative = (loss2-loss1)/smallnum
                    new_weights[weight_index] = weight - (partial_derivative*step_size)
                end
                --optimize bias:
                local brain_copy = brain
                local bias_copy = node.bias

                local loss1 = loss(brain, map)
                brain_copy.network[node_index].bias = bias_copy + smallnum
                local loss2 = loss(brain_copy, map)

                local partial_derivative = (loss2-loss1)/smallnum
                local new_bias = node.bias - (partial_derivative*step_size)

                --apply changes:
                node.weights = new_weights
                node.bias = new_bias
                brain.network[node_index] = node

                --add the next node to the queue:
                if node.input_nodes ~= {} then
                    for l, input_node_index in ipairs(node.input_nodes) do
                        local input_node = network[input_node_index]
                        table.insert(queue, input_node.index)
                    end
                end
                queue[k] = nil
            end
        end
    end
    return brain
end
