--VARIABLE DESCRIPTIONS:
--long_term_memory: memory={index=int, inputs={index}, outputs={index}, error=float, connected_memories={index}}
--short_term_memory: (position in long term memory paired with the value at that location)x20 Ex: 10, 4, nil, 0.7 (the error remembered from the 10th memory in the list)
--network: node={index=int, input_nodes={index}, output_nodes={index}, output=float, weights={float}, biases={float} inputs_recieved=int}
--queue_start: table of network indices connected to input nodes
--queue_end: table of network indices connected to output nodes

--data = {datapoint={inputs={}, outputs={}} (training data)
--layers = {amount of input nodes, amount of nodes in each hidden layer, amount of output nodes}, Ex: {5,10,11,12,10,3}

function copy_table(table)--dumb function that i shouldn't need because variables should copy tables in the first place like they do with ANY OTHER VALUE
    local table2 = {}
    for i, val in pairs(table) do
        if type(table[i]) == "table" then
            table2[i] = copy_table(table[i])
        else
            table2[i] = table[i]
        end
    end
    return table2
end

function clamp(val, min, max)
    if val > max then return max elseif val < min then return min else return val end
end

--GENERATE A NEURAL NETWORK:
function generate(layers, imagination_size)
    local brain = {network={}, imagination_size=imagination_size or 0, short_term_memory={}, long_term_memory={}, queue_start={}, queue={}, queue_end={}, last_modified_weight={node_index=nil, weight_index=nil, previous_val=nil}, last_modified_bias={node_index=nil, previous_val=nil}}
    local last_layer = {}
    local current_layer = {}
    for layer_index, node_amount in pairs(layers) do 
        for i=1,node_amount do
            node_index = #brain.network+1
            node = {index=node_index, input_nodes={}, output_nodes={}, output=0, weights={}, weight_sensitivities={}, weight_velocities={}, bias=math.random(-100,100)/100, bias_sensitivity=1, bias_velocity=0, inputs_recieved=0}
            if layer_index > 1 then
                for j,node_index2 in pairs(last_layer) do
                    node2 = brain.network[node_index2]
                    node2.output_nodes[#node2.output_nodes+1] = node_index
                    node.input_nodes[#node.input_nodes+1] = node_index2
                    node.weights[#node.weights+1] = math.random(-100,100)/100
                    node.weight_sensitivities[#node.weight_sensitivities+1] = 1
                    node.weight_velocities[#node.weight_velocities+1] = 0
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
    local network = copy_table(brain.network)
    local queue = copy_table(brain.queue_start)
    local imagination_size = brain.imagination_size
    local outputs = {}
    local i = 1
    while #queue > 0 do
        node_index = queue[1]
        local node = network[node_index]
        if node.inputs_recieved == #node.input_nodes then
            if #node.input_nodes > 0 then
                --apply bias:
                node.output = node.bias
                --add up all the inputs, applying their weights:
                for j, input_node_index in pairs(node.input_nodes) do
                    local input_node = network[input_node_index]
                    local input = input_node.output
                    local weight = node.weights[j]
                    node.output = node.output+(input*weight)
                end
                node.output = math.tanh(node.output)
                network[node_index] = node
            elseif i <= (#brain.queue_start-imagination_size) then
                node.output = math.tanh(inputs[i]) --inject inputs into first layer
            end

            --add the next node to the queue:
            if #node.output_nodes > 0 then
                for j, output_node_index in pairs(node.output_nodes) do
                    local output_node = network[output_node_index]
                    output_node.inputs_recieved = output_node.inputs_recieved + 1
                    network[output_node_index] = output_node
                    --check if output node is already queued:
                    local already_done = false
                    for k=1,#queue do
                        local other_node_index = queue[k]
                        if other_node_index ~= nil then
                            if output_node_index == other_node_index then already_done = true break end
                        end
                    end
                    if not already_done then table.insert(queue, output_node_index) end
                end
            else
                outputs[#outputs+1] = node.output
                --pass imagination outputs to next run:
                if #outputs > (#brain.queue_end-imagination_size) then
                    network[(#brain.queue_start-imagination_size) + #outputs] = node.output
                end
            end
        end
        table.remove(queue, 1)
        i = i + 1
    end
    return network, outputs
end

--MAKE A NEURAL NETWORK LEARN FROM MISTAKES (reinforcement learning)
function learn(brain, last_error, current_error) 
    local network = brain.network
    local smallnum = 0.0001
    local step_size = 0.1
    local max = 10
    if #brain.last_modified_weight > 0 then
        local modified_node_index, modified_weight_index, previous_val = brain.last_modified_weight[1], brain.last_modified_weight[2], brain.last_modified_weight[3]
        local weight_last_modified = network[modified_node_index].weights[modified_weight_index]
        local gradient = ((current_error-last_error)/smallnum)*step_size--find slope with respect to weight
        local velocity = brain.network[modified_node_index].weight_velocities[modified_weight_index]
        velocity = clamp(previous_val - (velocity+gradient), -max, max) - previous_val

        brain.network[modified_node_index].weights[modified_weight_index] = previous_val+velocity--reset weight and push toward slope
        brain.network[modified_node_index].weight_sensitivities[modified_weight_index] = math.min(math.abs(gradient/1000), 10000)+1--assign sensitivity value to weight
        brain.network[modified_node_index].weight_velocities[modified_weight_index] = velocity*0.5
        brain.last_modified_weight = {}
        return
    elseif #brain.last_modified_bias > 0 then
        local modified_node_index, previous_val = brain.last_modified_bias[1], brain.last_modified_bias[2]
        local bias_last_modified = network[modified_node_index].bias
        local gradient = ((current_error-last_error)/smallnum)*step_size--find slope with respect to bias
        local velocity = brain.network[modified_node_index].bias_velocity
        velocity = clamp(previous_val - (velocity+gradient), -max, max) - previous_val

        brain.network[modified_node_index].bias = previous_val+velocity--reset weight and push toward slope
        brain.network[modified_node_index].bias_sensitivity = math.min(math.abs(gradient/1000), 10000)+1--assign sensitivity value to bias
        brain.network[modified_node_index].bias_velocity = velocity*0.5
        brain.last_modified_bias = {}
        return
    end

    --select random weight/bias to modify (more sensitive weights/biases are more likely to be selected):
    local node_index
    local weight_index
    local bias_index
    local rand = 0
    for i, node in pairs(network) do
        for j, weight in pairs(node.weights) do
            rand = rand + node.weight_sensitivities[j]
        end
        rand = rand + node.bias_sensitivity
    end
    rand = (math.random(1000000000)/1000000000)*rand

    local weight_selected
    local bias_selected
    local index = 0
    for i, node in pairs(network) do
        for j, weight in pairs(node.weights) do
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
    elseif bias_selected then
        local bias = network[node_index].bias
        brain.network[node_index].bias = bias + smallnum
        brain.last_modified_bias = {node_index, bias}
    end
    --idea: run every frame, but rewind a frame every time you change a weight, so its training for the best possible output for a single frame
    --that would require an error for each frame though
    --idea 2: randomize outputs slightly, if it improves then use backprop to reflect map of inputs to said randomized outputs
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
        for i,output in pairs(outputs) do
            loss = loss + (map.outputs[i]-output)^2
        end
        return loss
    end
    for i=1,iterations do
        for j, map in pairs(data) do
            local queue = queue_end
            local step_size = 1
            local smallnum = 0.0001
            for k, node_index in pairs(queue) do
                --optimize weights:
                local node = network[node_index]
                local new_weights = {}
                for weight_index, weight in pairs(node.weights) do
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
                    for l, input_node_index in pairs(node.input_nodes) do
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
