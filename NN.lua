--VARIABLES DESCRIPTIONS:
--brain = {network={}, short_term_memory={}, long_term_memory={}, queue_start={}, queue_end={}, input_nodes={}, sensitive_weights={}, last_modified_weight={node_index, weight_index}}
--long_term_memory: memory={index=int, inputs={index}, outputs={index}, error=float, connected_memories={index}}
--short_term_memory: (position in long term memory paired with the value at that location)x20 Ex: 10, 4, nil, 0.7 (the error remembered from the 10th memory in the list)
--network: node={index=int, input_nodes={index}, output_nodes={index}, output=float, weights={float}, biases={float} inputs_recieved=int}
--queue_start: table of network indices connected to input nodes
--queue_end: table of network indices connected to output nodes

local data = {}--datapoint={inputs={}, outputs={}} (training data)
local layers = {5,10,11,12,10,3}--amount of input nodes, amount of nodes in each hidden layer, amount of output nodes

--GENERATE A NEURAL NETWORK:
function generate(layers)
    local brain = {network={}, short_term_memory={}, long_term_memory={}, queue_start={}, queue_end={}, input_nodes={}}
    local last_layer = {}
    local current_layer = {}
    for layer_index, node_amount in ipairs(layers) do 
        for i=1,node_amount do
            node_index = #brain.network+1
            node = {index=node_index, input_nodes={}, output_nodes={}, output=0, weights={}, bias=math.random(0,100)/100, inputs_recieved=0}
            if layer_index > 1 then
                for j,node_index2 in ipairs(last_layer) do
                    node2 = brain.network[node_index2]
                    node2.output_nodes[#node2.output_nodes+1] = node_index
                    node.input_nodes[#node.input_nodes+1] = node_index2
                    node.weights[#node.weights+1] = math.random(0,100)/100
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
function learn(brain, last_error, current_error)
    local network = brain.network
    if last_error then
        local modified_node_index, modified_weight_index = brain.last_modified_weight
        local weight_last_modified = network[modified_node_index][modified_weight_index]
        local partial_derivative = (current_error-last_error)/0.001
        network[modified_node_index][modified_weight_index] = (weight_last_modified-0.001) - (partial_derivative*1)
        --add weight to sensitive_weights
    end

    --select random weight:
    local modified_node_index = math.random(#network)
    local modified_weight_index = math.random(#network[modified_node_index].weights)
    --chance to re-select random weight (better-scoring weights are more likely to be selected):
    if math.random(2)==1 and last_error then
        local rand = 0
        for i, weight in ipairs(brain.sensitive_weights) do
            rand = rand + weight[2]
        end
        rand = math.random(rand)

        local index = 0
        for i, weight in ipairs(brain.sensitive_weights) do
            index = index + weight[2]
            if index >= rand then rand = weight; break end
        end
        modified_node_index = rand[1].node_index
        modified_weight_index = rand[1].weight_index
    end
    --modify selected weight:
    local modified_weight = network[modified_node_index][modified_weight_index]
    brain.network[modified_node_index][modified_weight_index] = modified_weight + 0.001
    brain.last_modified_weight = {modified_node_index, modified_weight_index}
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
