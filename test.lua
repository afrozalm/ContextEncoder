require 'image';
require 'torch';
require 'nn';
require 'cunn';
require 'cutorch';
require 'paths';

cmd = torch.CmdLine()
cmd:option('-old', 0)
cmd:option('-device', 1)
cmd:option('-start', 1)
cmd:option('-stop', 1)
params = cmd:parse(arg)

cutorch.setDevice(params['device'])
cutorch.synchronize()

generator =  nil
discriminator =  nil
classifier = nil

print('Loading Models ...')

if params['old'] == 1 then
    generatorName = './generatorOld.net'
    discriminatorName = './discriminatorOld.net'
else
    generatorName = './generator.net'
    discriminatorName = './discriminator.net'
end

if path.exists(generatorName) then
    generator = torch.load(generatorName)
    print('generator loaded')
else
    print('generator not available')
end
if path.exists(discriminatorName) then
    discriminator = torch.load(discriminatorName)
    print('discriminator loaded')
else
    print('discriminator not available')
end

print('Loading Data ... ')
testset = torch.load('../FaceScrub/FaceScrub_testset_128x128')

function redoutline( image )
    image[{{1},{32, 97},{32}}] = 255
    image[{{2, 3},{32, 97},{32}}] = 0
    image[{{1},{32, 97},{97}}] = 255
    image[{{2, 3},{32, 97},{97}}] = 0
    image[{{1},{32},{32, 97}}] = 255
    image[{{2, 3},{32},{32, 97}}] = 0
    image[{{1},{97},{32, 97}}] = 255
    image[{{2, 3},{97},{32, 97}}] = 0
    return image
end

M = torch.CudaTensor( 3, 128, 128 ):fill(1)      -- filling mask
M[{ {}, {33, 96}, {33, 96} }] = 0
print('Loaded')
print('Processing ...')

for index = params['start'], params['stop'] do
    maskedInput = torch.CudaTensor( 3, 128, 128 ):copy( testset.data[index] ):cmul( M )
    if generator ~= nil then
        local temp4DTensor = torch.CudaTensor( 2, 3, 128, 128 )
        temp4DTensor[1] = maskedInput
        temp4DTensor[2]:fill(0)
        output = generator:forward( temp4DTensor )
        outputImage = output[1]
        maskedInput[{ {}, {33, 96}, {33, 96} }] = outputImage
        temp4DTensor[1] = maskedInput
        probability = discriminator:forward(temp4DTensor)
        print( 'The output is real with probability ' , probability[1] )

        print('Saving results')
        outname = index .. '_output.jpg'
        image.save( outname, redoutline( maskedInput ) )
        temp4DTensor = nil
        collectgarbage()
    end
end
