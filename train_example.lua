--SCRIPT TO TEACH AI TO BE AND-GATE
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

local data = {
    {inputs={1,1}, outputs={-1}},
    {inputs={-1,-1}, outputs={-1}},
    {inputs={-1,1}, outputs={1}},
    {inputs={1,-1}, outputs={1}},
}

local layers = {2,3,1}--amount of input nodes, amount of nodes in each hidden layer, amount of output nodes
--find brain:
local success, brain = pcall(dofile, "brain.lua")
if (not success) or (not brain) then brain = generate(layers) end--no brain? maek new one

train(brain, data, 100)
for i, map in pairs(data) do
   local network, outputs = think(brain, map.inputs)
   print("INPUTS: "..dump(map.inputs))
   print("OUTPUT: "..outputs[1])
   print("ERROR: "..(map.outputs[1]-outputs[1])^2)
end
--update brain on file:
local file = io.open("brain.lua", "w+")
file:write("return "..dump(brain))