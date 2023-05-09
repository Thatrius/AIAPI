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
    local brain = {network={}, short_term_memory={}, long_term_memory={}, queue_start={}, queue_end={}, input_nodes={}, last_modified_weight={node_index=nil, weight_index=nil}}
    local last_layer = {}
    local current_layer = {}
    for layer_index, node_amount in ipairs(layers) do 
        for i=1,node_amount do
            node_index = #brain.network+1
            node = {index=node_index, input_nodes={}, output_nodes={}, output=0, weights={}, weight_sensitivities={}, bias=math.random(0,100)/100, bias_sensitivity=1, inputs_recieved=0}
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
                brain.input_nodes[#brain.input_nodes+1] = node_index
            elseif layer_index == 2 then 
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
function think(brain)
    local network = brain.network
    local queue = brain.queue_start
    local outputs = {}
    for i, node_index in ipairs(queue) do
        local node = network[node_index]
        if node.inputs_recieved == #node.input_nodes then
            --apply bias:
            node.output = node.output + bias
            --add up all the inputs, applying their weights:
            for j, input_node_index in ipairs(node.input_nodes) do --when/how determine inputs?
                local input_node = network[input_node_index]
                local input = input_node.output
                local weight = node.weights[input_node]
                local bias = node.biases[input_node]
                node.output = node.output+(input*weight)
            end
            node.output = math.tanh(node.output)
            network[node_index] = node

            --add the next node to the queue:
            if node.output_nodes ~= {} then
                for j, output_node_index in ipairs(node.output_nodes) do
                    local output_node = network[output_node_index]
                    output_node.inputs_recieved = output_node.inputs_recieved + 1
                    network[output_node_index] = output_node
                    table.insert(queue, output_node.index)
                end
            else
                outputs[#outputs+1] = node.output
            end
        end
        queue[i] = nil
    end
    return network, outputs
end

--MAKE A NEURAL NETWORK LEARN FROM MISTAKES (reinforcement learning)
function learn(brain, last_error, current_error) --needs to modify biases as well
    local network = brain.network
    local smallnum = 0.001
    if last_error then
        local modified_node_index, modified_weight_index = brain.last_modified_weight
        local weight_last_modified = network[modified_node_index].weights[modified_weight_index]
        --find slope with respect to weight:
        local partial_derivative = (current_error-last_error)/smallnum
        --reset weight and push toward slope:
        network[modified_node_index].weights[modified_weight_index] = (weight_last_modified-smallnum) - (partial_derivative*1)
        --assign sensitivity value to weight:
        network[modified_node_index].weight_sensitivities[modified_weight_index] = partial_derivative
    end

    --select random weight to modify (more sensitive weights are more likely to be selected):
    local node_index
    local weight_index
    local rand = 0
    for i, node in ipairs(network) do
        for j, weight in ipairs(node.weights) do
            rand = rand + node.weight_sensitivities[j]
        end
    end
    rand = math.random(math.floor(rand*100000))/100000

    local index = 0
    local weight_selected = false
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
        if weight_selected then break end
    end

    --modify selected weight:
    local weight = network[node_index].weights[weight_index]
    brain.network[node_index].weights[weight_index] = weight + 0.001
    brain.last_modified_weight = {node_index, weight_index}
    return brain
    --idea: run every frame, but rewind a frame every time you change a weight, so its training for the best possible output for a single frame
    --that would require an error for each frame though
end

--TRAIN A NEURAL NETWORK (backprop)
function train(brain, data, iterations)
    local network = brain.network
    local queue_start = brain.queue_start
    local queue_end = brain.queue_end
    function loss(network, map)
        --compare output values to desired output values from map:
        local network2, outputs = think(brain)
        local loss = 0
        for i,output in ipairs(outputs) do
            loss = loss + (map.outputs[i]-output)^2
        end
        return loss
    end
    for map=1,#data do
        local queue = queue_end
        local step_size = 1
        local smallnum = 0.0001
        for i, node_index in ipairs(queue) do
            local node = network[node_index]
            local new_weights = {}
            for j, weight in ipairs(node.weights) do
                local network_copy = network
                local weight_copy = weight

                local loss1 = loss(brain, map)
                brain_copy.network.weights[j] = weight_copy + smallnum
                local loss2 = loss(brain_copy, map)

                local partial_derivative = (loss2-loss1)/smallnum
                new_weights[j] = weight - (partial_derivative*step_size)
            end
            node.weights = new_weights
            network[node_index] = node

            --add the next node to the queue:
            if node.input_nodes ~= {} then
                for j, input_node_index in ipairs(node.input_nodes) do
                    local input_node = network[input_node_index]
                    table.insert(queue, input_node.index)
                end
            end
            queue[i] = nil
        end
        return network
    end
end
