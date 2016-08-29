require 'image';
require 'torch';
require 'nn';
require 'cutorch';
require 'cunn';

cmd = torch.CmdLine()
cmd:option('-max', 10)
cmd:option('-device', 1)
params = cmd:parse(arg)

cutorch.setDevice(params['device'])
cutorch.synchronize()

testset = torch.load('../FaceScrub/FaceScrub_testset_128x128')
trainset = torch.load('../FaceScrub/FaceScrub_trainset_128x128')
generator = torch.load('./generatorFace.net')
discriminator = torch.load('./discriminatorFace.net')

salientTest = torch.Tensor( params['max'] ):zero()
salientTrain = torch.Tensor( params['max'] ):zero()

lamda_rec = 0.30
lamda_adv = 1 - lamda_rec

criterion_rec = nn.MSECriterion():cuda()
criterion_adv = nn.BCECriterion():cuda()

M = torch.CudaTensor( 2, 3, 128, 128 ):fill(1)      -- filling mask
M[{{},{},{33, 96},{33, 96}}]:zero()
maskedInput = torch.CudaTensor( 2, 3, 128, 128 )
targetImage = torch.CudaTensor( 2, 3, 64, 64 )

stitch = function( small, large )
    large[{{}, {}, {33, 96}, {33, 96}}] = small[{{}, {}, {1, 64}, {1, 64}}]
    return large
end

function insert( err, tree, x )
    if tree == nil then
        tree = {
            index = x,
            value = err,
            left = nil,
            right = nil
        }
        return tree
    elseif tree['value'] < err then
        tree['right'] = insert( err, tree['right'], x )
        return tree
    else
        tree['left'] = insert( err, tree['left'], x )
        return tree
    end
end

function removeMin( tree )
    if tree == nil then
        return nil
    end

    if tree['left'] == nil then
        local temp = tree['right']
        tree = nil
        collectgarbage()
        return temp
    else
        tree['left'] = removeMin( tree['left'] )
        return tree
    end
end

function fillArr( a, tree )
    if tree == nil then
        return
    end
    if tree['left'] ~= nil then
        fillArr( a, tree['left'] )
    end
    a[ current_index ] = tree['index']
    current_index = current_index + 1
    if tree['right'] ~= nil then
        fillArr( a, tree['right'] )
    end
    return
end

root = nil
current_size = 0
for i = 1, trainset.size do
    collectgarbage()
    targetImage[1] = trainset.data[{{ i }, {}, {33, 96}, {33, 96}}]
    targetImage[2] = targetImage[1]
    maskedInput[1] = trainset.data[i]
    maskedInput[2] = maskedInput[1]
    targetLabel = torch.CudaTensor( 2 ):fill(1)
    maskedInput:cuda()
    targetImage:cuda()
    maskedInput:cmul(M)
    outputImage = generator:forward( maskedInput )
    print( i )
    err_rec = criterion_rec:forward( outputImage, targetImage )
    err_adv = criterion_adv:forward( discriminator:forward( stitch( outputImage, maskedInput ) ), targetLabel )
    err = err_rec * lamda_rec + err_adv * lamda_adv
    if current_size == 0 then
        current_size = 1
        root = insert( err, root, i )
    elseif current_size == params['max'] then
        root = removeMin( root )
        root = insert( err, root, i )
    else
        root = insert( err, root, i )
        current_size = current_size + 1
    end
end

current_index = 1
fillArr( salientTrain, root )

i = params['max']
while i > 0 do
    name = params['max'] + 1 - i .. '_point_3_rankedTrainSalient.jpg'
    image.save( name, trainset.data[salientTrain[i]] )
    i = i - 1
end

root = nil
current_size = 0
for i = 1, testset.size do
    collectgarbage()
    targetImage[1] = testset.data[{{ i }, {}, {33, 96}, {33, 96}}]
    targetImage[2] = targetImage[1]
    maskedInput[1] = testset.data[i]
    maskedInput[2] = maskedInput[1]
    targetLabel = torch.CudaTensor( 2 ):fill(1)
    maskedInput:cuda()
    targetImage:cuda()
    maskedInput:cmul(M)
    outputImage = generator:forward( maskedInput )
    print( i )
    err_rec = criterion_rec:forward( outputImage, targetImage )
    err_adv = criterion_adv:forward( discriminator:forward( stitch( outputImage, maskedInput ) ), targetLabel )
    err = err_rec * lamda_rec + err_adv * lamda_adv
    if current_size == 0 then
        current_size = 1
        root = insert( err, root, i )
    elseif current_size == params['max'] then
        root = removeMin( root )
        root = insert( err, root, i )
    else
        root = insert( err, root, i )
        current_size = current_size + 1
    end
end

current_index = 1
fillArr( salientTest, root )

i = params['max']
while i > 0 do
    name = params['max'] - i + 1 .. '_point_3_rankedTestSalient.jpg'
    image.save( name, testset.data[salientTest[i]] )
    i = i - 1
end
