--SCRIPT TO MAKE AI TEACH ITSELF TO BE AND-GATE
dofile("NN.lua")

function dump(o, i)
    i = (i or 0) + 1
    if type(o) == 'table' then
       local s = '{'
       for k,v in pairs(o) do
          if type(k) ~= 'string' then 
             k = '' 
             if type(v) == 'table' then k = "\n" end
          else
             k = k..'='
          end
          s = s..k..dump(v, i)
          if k ~= o[#o] then s = s..', ' end
       end
       s = s..'}'
       return s
    else
       return tostring(o)
    end
 end

function loss(inputs, outputs)
    local correct_outputs = {-1}
    if (inputs[1]==1 and inputs[2]==-1) or (inputs[1]==-1 and inputs[2]==1) then correct_outputs[1] = 1 end
    if (correct_outputs[1]-outputs[1])^2 < 0.0001 then return 0 end
    return (correct_outputs[1]-outputs[1])^2
end

local layers = {2,3,1}--amount of input nodes, amount of nodes in each hidden layer, amount of output nodes
--find brain:
local success, brain = pcall(dofile, "brain.lua")
if (not success) or (not brain) then brain = generate(layers) end--no brain? maek new one

local error1 = 1
for i=1,10000 do
    local inputs = {math.random() < 0.5 and -1 or 1, math.random() < 0.5 and -1 or 1}
    local network, outputs = think(brain, inputs)
    local error2 = loss(inputs, outputs)

    if (i<6) or (i>9994) then --display progress in first 5 iterations and last 5 iterations
        print(i..":")
        print("INPUTS: "..dump(inputs))
        print("OUTPUT: "..outputs[1])
        print("ERROR: "..error2)
    end

    learn(brain, error1, error2)

    error1 = error2
end
--update brain on file:
local file = io.open("brain.lua", "w+")
file:write("return "..dump(brain))