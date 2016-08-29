require 'cutorch';
require 'cunn';
require 'torch';
require 'nn';

--------------------------------------------------------------------------
--                          Model                                       --
--------------------------------------------------------------------------



encoder = nn.Sequential()

encoder:add(nn.SpatialConvolution(3, 64, 4, 4, 2, 2, 1, 1))         -- 3x128x128 -> 64x64x64
    --encoder.modules[#encoder.modules].weight:normal(0, 0.02)
    --encoder.modules[#encoder.modules].bias:fill(0)
encoder:add(nn.SpatialBatchNormalization(64))
   -- encoder.modules[#encoder.modules].weight:normal(1.0, 0.02)
   -- encoder.modules[#encoder.modules].bias:fill(0)
encoder:add(nn.ReLU(true))                                                --relu1
encoder:add(nn.SpatialDropout(0.25))
encoder:add(nn.SpatialConvolution(64, 64, 4, 4, 2, 2, 1, 1))        --64x64x64 -> 64x32x32
   -- encoder.modules[#encoder.modules].weight:normal(0, 0.02)
   -- encoder.modules[#encoder.modules].bias:fill(0.1)
encoder:add(nn.SpatialBatchNormalization(64))
   -- encoder.modules[#encoder.modules].weight:normal(1.0, 0.02)
   -- encoder.modules[#encoder.modules].bias:fill(0)
encoder:add(nn.ReLU(true))                                                --relu2
encoder:add(nn.SpatialConvolution(64, 128, 4, 4, 2, 2, 1, 1))       --64x32x32 -> 128x16x16
   -- encoder.modules[#encoder.modules].weight:normal(0, 0.02)
   -- encoder.modules[#encoder.modules].bias:fill(0)
encoder:add(nn.SpatialBatchNormalization(128))
   -- encoder.modules[#encoder.modules].weight:normal(1.0, 0.02)
   -- encoder.modules[#encoder.modules].bias:fill(0)
encoder:add(nn.ReLU(true))                                                --relu3
encoder:add(nn.SpatialConvolution(128, 256, 4, 4, 2, 2, 1, 1))      --128x16x16 -> 256x8x8
   -- encoder.modules[#encoder.modules].weight:normal(0, 0.02)
   -- encoder.modules[#encoder.modules].bias:fill(0.1)
encoder:add(nn.SpatialBatchNormalization(256))
   -- encoder.modules[#encoder.modules].weight:normal(1.0, 0.02)
   -- encoder.modules[#encoder.modules].bias:fill(0)
encoder:add(nn.ReLU(true))                                             --relu4
encoder:add(nn.SpatialConvolution(256, 512, 4, 4, 2, 2, 1, 1))      --256x8x8 -> 512x4x4
   -- encoder.modules[#encoder.modules].weight:normal(0, 0.02)
   -- encoder.modules[#encoder.modules].bias:fill(0.1)
encoder:add(nn.SpatialBatchNormalization(512))
   -- encoder.modules[#encoder.modules].weight:normal(1.0, 0.02)
   -- encoder.modules[#encoder.modules].bias:fill(0)
encoder:add(nn.ReLU(true))                                                --relu5
encoder:add(nn.View( 512*4*4 ))
encoder:add( nn.Linear( 512*4*4, 4500 ) )                           --512*4*4 -> 4000
encoder:add( nn.Tanh() )

------------------------------------------------------------------
--                  classifier for pre-training                 --
------------------------------------------------------------------

classifier = nn.Sequential()
classifier:add( encoder )           -- 3x128x128 -> 4000
classifier:add( nn.Linear( 4500, 400 ) )
classifier:add( nn.Tanh() )
classifier:add( nn.Linear( 400, 1 ) )
classifier:add( nn.Sigmoid() )

------------------------------------------------------------------
-- Decoder

decoder = nn.Sequential()

decoder:add( nn.Linear( 4500, 512*4*4 ) )
decoder:add( nn.Reshape( 512, 4, 4 ) )
decoder:add( nn.SpatialFullConvolution( 512, 256, 4, 4, 2, 2, 1, 1 ) )       --512x4x4 -> 256x8x8
   -- decoder.modules[#decoder.modules].weight:normal(0, 0.02)
   -- decoder.modules[#decoder.modules].bias:fill(0.1)
decoder:add( nn.SpatialBatchNormalization( 256 ) )
   -- decoder.modules[#decoder.modules].weight:normal(1.0, 0.02)
   -- decoder.modules[#decoder.modules].bias:fill(0)
decoder:add(nn.ReLU(true))
decoder:add( nn.SpatialFullConvolution( 256, 128, 4, 4, 2, 2, 1, 1 ) )       --256x8x8 -> 128x16x16
   -- decoder.modules[#decoder.modules].weight:normal(0, 0.02)
   -- decoder.modules[#decoder.modules].bias:fill(0.1)
decoder:add( nn.SpatialBatchNormalization( 128 ) )
   -- decoder.modules[#decoder.modules].weight:normal(1.0, 0.02)
   -- decoder.modules[#decoder.modules].bias:fill(0)
decoder:add(nn.ReLU(true))
decoder:add( nn.SpatialFullConvolution( 128, 64, 4, 4, 2, 2, 1, 1 ) )        --128x16x16 -> 64x32x32
   -- decoder.modules[#decoder.modules].weight:normal(0, 0.02)
   -- decoder.modules[#decoder.modules].bias:fill(0.1)
decoder:add( nn.SpatialBatchNormalization( 64 ) )
   -- decoder.modules[#decoder.modules].weight:normal(1.0, 0.02)
   -- decoder.modules[#decoder.modules].bias:fill(0)
decoder:add(nn.ReLU(true))
decoder:add( nn.SpatialFullConvolution( 64, 3, 4, 4, 2, 2, 1, 1 ) )          --64x32x32 -> 3x64x64
decoder:add(nn.Tanh())

generator = nn.Sequential()
generator:add( encoder )
generator:add( decoder )

------------------------------------------------------------------

-- discriminator
discriminator = nn.Sequential()

discriminator:add( nn.SpatialConvolution( 3, 32, 4, 4, 2, 2, 1, 1 ) )           -- 3x128x128 -> 32x64x64
   -- discriminator.modules[#discriminator.modules].weight:normal(0, 0.02)
   -- discriminator.modules[#discriminator.modules].bias:fill(0.1)
discriminator:add( nn.SpatialBatchNormalization( 32 ) )
   -- discriminator.modules[#discriminator.modules].weight:normal(1.0, 0.02)
   -- discriminator.modules[#discriminator.modules].bias:fill(0)
discriminator:add(nn.LeakyReLU(0.2, true))
discriminator:add( nn.SpatialConvolution( 32, 64, 4, 4, 2, 2, 1, 1 ) )           -- 32x64x64 -> 64x32x32
   -- discriminator.modules[#discriminator.modules].weight:normal(0, 0.02)
   -- discriminator.modules[#discriminator.modules].bias:fill(0.1)
discriminator:add( nn.SpatialBatchNormalization( 64 ) )
   -- discriminator.modules[#discriminator.modules].weight:normal(1.0, 0.02)
   -- discriminator.modules[#discriminator.modules].bias:fill(0)
discriminator:add(nn.LeakyReLU(0.2, true))
discriminator:add( nn.SpatialConvolution( 64, 128, 4, 4, 2, 2, 1, 1 ) )         -- 64x32x32 -> 128x16x16
   -- discriminator.modules[#discriminator.modules].weight:normal(0, 0.02)
   -- discriminator.modules[#discriminator.modules].bias:fill(0.1)
discriminator:add( nn.SpatialBatchNormalization( 128 ) )
   -- discriminator.modules[#discriminator.modules].weight:normal(1.0, 0.02)
   -- discriminator.modules[#discriminator.modules].bias:fill(0)
discriminator:add(nn.LeakyReLU(0.2, true))
discriminator:add( nn.SpatialConvolution( 128, 256, 4, 4, 2, 2, 1, 1 ) )        -- 128x16x16 -> 256x8x8
   -- discriminator.modules[#discriminator.modules].weight:normal(0, 0.02)
   -- discriminator.modules[#discriminator.modules].bias:fill(0.1)
discriminator:add( nn.SpatialBatchNormalization( 256 ) )
   -- discriminator.modules[#discriminator.modules].weight:normal(1.0, 0.02)
   -- discriminator.modules[#discriminator.modules].bias:fill(0)
discriminator:add(nn.LeakyReLU(0.2, true))
discriminator:add( nn.SpatialConvolution( 256, 512, 4, 4, 2, 2, 1, 1 ) )        -- 256x8x8 -> 512x4x4
   -- discriminator.modules[#discriminator.modules].weight:normal(0, 0.02)
   -- discriminator.modules[#discriminator.modules].bias:fill(0.1)
discriminator:add( nn.View( 512*4*4 ) )
discriminator:add( nn.Linear(512*4*4, 1) )
discriminator:add( nn.Sigmoid() )


--converting to cuda

--classifier:cuda()
generator:cuda()
discriminator:cuda()
