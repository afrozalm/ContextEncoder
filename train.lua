require 'image';
require 'torch';
require 'optim';
require 'paths';
require 'nn';
require 'cunn';
require 'cutorch';

cmd = torch.CmdLine()
cmd:option('-fresh', 0)
cmd:option('-advLimit', 0.25)
cmd:option('-advInitial', 0.20)
cmd:option('-device', 1)
cmd:option('-trSize', 30000)
cmd:option('-valSize', 700)
cmd:option('-trainPath','../FaceScrub/FaceScrub_trainset_128x128' )
cmd:option('-testPath','../FaceScrub/FaceScrub_testset_128x128' )
cmd:option('-start', 9)
cmd:option('-stop', 16)
params = cmd:parse(arg)

cutorch.setDevice(params['device'])
cutorch.synchronize()

print('Loading Model ... ')
if path.exists('./generator.net') == false or params['fresh'] == 1 then
    print('Loading new model')
    require './model.lua'
    hyperParams = {
        epoch = 1, 
        lamda_adv = params['advInitial'],
        lamda_rec = 1 - params['advInitial'],
        learningRate = 1e-3
    }
else
    print('Loading saved model')
    generator = torch.load('generator.net')
    discriminator = torch.load('discriminator.net')
    os.execute('mv generator.net generatorOld.net')
    os.execute('mv discriminator.net discriminatorOld.net')
    hyperParams = torch.load('./hyperParams')
end
print( hyperParams )
print('Loaded')

print('Loading Data ... ')
fullset = torch.load(params['trainPath'])
testset = torch.load(params['testPath'])
print('Loaded')

trainset = {}
trainset.size = math.min( params['trSize'], fullset.size - 700 )
trainset.data = fullset.data[{{1, trainset.size}}]
trainset.label = fullset.label[{{1, trainset.size}}]

validationset = {}
validationset.size = math.min( params['valSize'], fullset.size - trainset.size )
validationset.data = fullset.data[{{trainset.size + 1, trainset.size + validationset.size}}]
validationset.label = fullset.label[{{trainset.size + 1, trainset.size + validationset.size}}]
--------------------------------------------------------------------------
--                  Loss Criterions                                     --
--------------------------------------------------------------------------

lamda_adv = hyperParams.lamda_adv
lamda_rec = hyperParams.lamda_rec

criterion_rec = nn.MSECriterion():cuda()
criterion_adv = nn.BCECriterion():cuda()

params_G, gradParams_G = generator:getParameters()
params_D, gradParams_D = discriminator:getParameters()

---------------------------------------------------------------------------


optimState_G = {
   learningRate = hyperParams.learningRate,
   beta1 = 0.5,
}

optimState_D = {
   learningRate = hyperParams.learningRate,
   beta1 = 0.5
}

----------------------------------------------------------------------
--                      Training Full Model                         --
----------------------------------------------------------------------

stitch = function( small, large )
    large[{{}, {}, {33, 96}, {33, 96}}] = small[{{}, {}, {1, 64}, {1, 64}}]
    return large
end

TrainingStep = function( batchsize )
    --print('___Entering TrainingStep')

    local size = batchsize or 50
    --print('___Allocating gpu memo ...')
    local targetLabel = torch.CudaTensor(size)
    --print('___Allocated gpu memo')
    for minibatch_number = 1, trainset.size, batchsize do
        start = minibatch_number
        if start + batchsize - 1 <= trainset.size then
            local maskedInput = trainset.data[{{start, start + batchsize - 1}}]:cuda()
            local targetImage = trainset.data[{{start, start + batchsize - 1}, {}, {33, 96}, {33, 96}}]:cuda()
            local fullInput = trainset.data[{{start, start + batchsize - 1}}]:cuda()
            maskedInput:cmul( M )

            local outputImage = generator:forward( maskedInput )
            local stitchedImage = stitch( outputImage, maskedInput )
            feval_D = function( x_new )
                --print('_________Entering feval_D')
                collectgarbage()
                if params_D ~= x_new then params_D:copy(x_new) end
                --discriminator:apply( function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end )
                --generator:apply( function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end )
                gradParams_D:zero()

                 -- real updation
                local real = discriminator:forward( fullInput )
                targetLabel:fill(1)
                local loss = criterion_adv:forward( real, targetLabel )
                df_do = criterion_adv:backward( real, targetLabel )
                discriminator:backward( fullInput, df_do )

                -- fake updation
                local fake = discriminator:forward( stitchedImage )
                targetLabel:zero()
                loss = loss + criterion_adv:forward( fake, targetLabel )
                local df_do = criterion_adv:backward( fake, targetLabel )
                discriminator:backward( stitchedImage, df_do )

                return loss, gradParams_D
            end
                
            feval_G = function( x_new )
                --print('_________Entering feval_G')
                collectgarbage()
                if params_G ~= x_new then params_G:copy(x_new) end
                --discriminator:apply( function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end )
                --generator:apply( function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end )
                gradParams_G:zero()

                local loss = lamda_rec * criterion_rec:forward( outputImage, targetImage )
                local df_do_rec = criterion_rec:backward( outputImage, targetImage )
                
                local fake = discriminator:forward( stitchedImage )
                targetLabel:fill(1)
                local loss = loss + lamda_adv * criterion_adv:forward( fake, targetLabel )
                local df_do_adv = criterion_adv:backward( fake, targetLabel )
 
                local df_dg_full = discriminator:updateGradInput( stitchedImage, df_do_adv )
                local df_dg = df_dg_full[{{}, {}, {33, 96}, {33, 96}}]
                local df_do = df_dg * lamda_adv + df_do_rec * lamda_rec

                generator:backward( maskedInput, df_do )

                return loss, gradParams_G
            end
           
            --print('______Entering sgd_D ' )
            for k = 1, 3 do
                optim.sgd(feval_D, params_D, optimState_D)
            end

            --print('______Entering sgd_G ' )
            optim.adam(feval_G, params_G, optimState_G)
         end
    end
    --print('___Freeing gpu memory ...')
    maskedInput = nil
    targetImage = nil
    targetLabel = nil
    output = nil
    outputImage = nil
    fake = nil
    real = nil
    collectgarbage()
    --os.execute('nvidia-smi')
    --print('___Exiting TrainingStep')
end

eval = function(  )
    --print('___Entering eval')
    collectgarbage()
    local size = validationset.size
    --print('___Allocating memory')
    local targetLabel = torch.CudaTensor( size ):fill(1)
    --print('___Copying Data')
    local maskedInput = validationset.data
    targetImage = maskedInput[{{}, {}, {33, 96}, {33, 96}}]
    maskedInput = maskedInput:cuda()
    targetImage = targetImage:cuda()

    for index = 1, size do
        maskedInput[index]:cmul( M[1] )
    end
    --print('___Forwarding inputs')
    local outputImage = generator:forward( maskedInput )
    --local loss = criterion_rec:forward( outputImage, targetImage )
    local probs = discriminator:forward( stitch( outputImage, maskedInput ) )
    local loss = lamda_rec * criterion_rec:forward( outputImage, targetImage )
    local loss = loss + lamda_adv * criterion_adv:forward( discriminator:forward( stitch( outputImage, maskedInput ) ), targetLabel  )
    --print('___Freeing memory')
    maskedInput = nil
    targetImage = nil
    targetLabel = nil
    collectgarbage()

    meanRealism = probs:sum()

    --os.execute('nvidia-smi')
    --print('___Exiting eval')
    return loss, meanRealism/validationset.size
end

function test( epoch )

    function redoutline( image )
        image[{{1},{32, 97},{32}}] = 255
        image[{{1},{32, 97},{97}}] = 255
        image[{{1},{32},{32, 97}}] = 255
        image[{{1},{97},{32, 97}}] = 255
        image[{{2, 3},{32, 97},{32}}] = 0
        image[{{2, 3},{32, 97},{97}}] = 0
        image[{{2, 3},{32},{32, 97}}] = 0
        image[{{2, 3},{97},{32, 97}}] = 0
        return image
    end

    for index = params['start'], params['stop'] do
        local M = torch.CudaTensor( 3, 128, 128 ):fill(1)
        M[{{},{33, 96},{33, 96}}]:zero()
        local maskedInput = torch.CudaTensor( 3, 128, 128 ):copy( testset.data[index] ):cmul( M )
        local temp4DTensor = torch.CudaTensor( 2, 3, 128, 128 )
        temp4DTensor[1] = maskedInput
        temp4DTensor[2]:fill(0)
        output = generator:forward( temp4DTensor )
        outputImage = output[1]
        maskedInput[{ {}, {33, 96}, {33, 96} }] = outputImage
        outname = epoch .. '_epoch__' .. index .. '_indexFULL.jpg'
        image.save( outname, redoutline( maskedInput ) )
        maskedInput = nil
        M = nil
        temp4DTensor = nil
        collectgarbage()
    end
end

--test(0)
--os.execute( 'th test.lua -stop 1 -old 1' )
-- Training
print('Training')
increasing = 0
prev_loss = 0
validation_loss = 0
converged = false
segment_count = 0
i = hyperParams.epoch
offest = 1
while not converged do
    collectgarbage()
    --a = 1 + ( offest - 1 ) * validationset.size
    --b = a + validationset.size - 1
    --if b < fullset.size then
        --validationset.data = fullset.data[{{a, b}}]
        --validationset.label = fullset.label[{{a, b}}]
        --offest = offest + 1
    --else 
        --offest = 1
    --end

    --os.execute('nvidia-smi')

    if i < 6 then 
        local bsize = 1 + 1
        M = torch.CudaTensor( bsize, 3, 128, 128 ):fill(1)
        M[{{},{},{33, 96},{33, 96}}]:zero()

        TrainingStep( bsize ) 
    else 
        local bsize = 4 + 1
        M = torch.CudaTensor( bsize, 3, 128, 128 ):fill(1)
        M[{{},{},{33, 96},{33, 96}}]:zero()
        
        TrainingStep( bsize ) 
    end
    validation_loss, realism = eval()

    if prev_loss < validation_loss and increasing == 5 then
        increasing = increasing + 1
        converged = true
    end

    if  prev_loss < validation_loss then
        if segment_count == 30 then
            increasing = 0
            segment_count = 0
        else
            increasing = increasing + 1
        end
    else
        increasing = 0
    end

    print('Epoch : ' .. i, 'Diff ' .. validation_loss - prev_loss, 'FullVal Loss : ' .. validation_loss, 'MeanRealism : ' .. realism, 'lamda_adv ' .. lamda_adv .. '/' .. params['advLimit'])
    prev_loss = validation_loss
    segment_count = segment_count + 1

    if i % 5 == 0 then
        print('Saving Results')
        test( i )
        optimState_G['learningRate'] = math.max( 2e-4, optimState_G['learningRate']*0.8 )
        optimState_D['learningRate'] = math.max( 2e-4, optimState_D['learningRate']*0.8 )
    end

    if i % 10 == 0 then
        if path.exists('./generator.net') ~= false then
            os.execute('rm generator.net discriminator.net')
        end
        torch.save('generator.net', generator)
        torch.save('discriminator.net', discriminator)
        hyperParams.lamda_adv = lamda_adv
        hyperParams.lamda_rec = lamda_rec
        hyperParams.learningRate = optimState_G['learningRate']
        hyperParams.epoch = i
        torch.save('hyperParams', hyperParams)
        print('Saved Itermediate Models')
    end

    if lamda_adv > params['advLimit'] and i % 10 == 0 then
        lamda_adv = lamda_adv - 0.01
        lamda_rec = lamda_rec + 0.01
    end

    if lamda_adv < params['advLimit'] and i % 10 == 0 then
        lamda_adv = lamda_adv + 0.01
        lamda_rec = lamda_rec - 0.01
    end

    --print('_')
    collectgarbage()
    i = i + 1
end

hyperParams.lamda_adv = lamda_adv
hyperParams.lamda_rec = lamda_rec
hyperParams.epoch = i
hyperParams.learningRate = optimState_G['learningRate']
torch.save('hyperParams', hyperParams)

print('Training Complete')
test( i )
print('Saving and exiting ...')
if path.exists('./generator.net') ~= false then
    os.execute('rm generator.net discriminator.net')
end
torch.save('generator.net', generator)
torch.save('discriminator.net', discriminator)
