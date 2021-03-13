using Base.Iterators: partition
using Flux
using Flux.Optimise: update!
using Flux.Losses: logitbinarycrossentropy
using Images
using ImageMagick
using MLDatasets
using Statistics
using Parameters: @with_kw
using Random
using Printf
using CUDA
using Zygote
using FileIO
using Plots
using BSON:@save

if has_cuda()		# Check if CUDA is available
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

@with_kw struct HyperParams
    batch_size::Int = 128
    latent_dim::Int = 100
    epochs::Int = 25
    verbose_freq::Int = 100
    checkpoint_freq::Int = 1000
    noise_reduc::Float32 = 0.001 # reduce by this amount every verbose_freq iterations

    # Discriminator
    lr_dscr::Float64 = 0.0004
    beta_dscr::Float64 = 0.5
    beta2_dscr::Float64 = .99
    dropout_disc::Float64 = 0.5

    # Generator
    lr_gen::Float64 = 0.0001
    beta_gen::Float64 = 0.5
    beta2_gen::Float64 = .9999
    dropout_gen::Float64 = 0.5
    #dropout_gen::Float64 = 0.0

    # Data
    trainPath::String = "../minimapPortraits/train/"
    champPath::String = "../preped_champ/"
    outDims::Int = 6
end

struct Generator
    g_latent          # Submodel to take latent_dims as input and convert it to shape of (7, 7, 128, batch_size)
    g_model    
end

function generator(args)
    g_latent = Chain(Dense(args.latent_dim, 4*4*1024), x-> leakyrelu.(x, 0.2f0), x-> reshape(x, 4, 4, 1024, size(x, 2))) |> gpu
    g_model = Chain(
                ConvTranspose((4,4), 1024=>512; init = Flux.glorot_uniform, stride=2, pad=2),
                BatchNorm(512),
                x-> leakyrelu.(x, 0.2f0),   
                Dropout(args.dropout_gen),
                ConvTranspose((4,4), 512=>256; init = Flux.glorot_uniform, stride=2, pad=2),
                BatchNorm(256),
                x-> leakyrelu.(x, 0.2f0),
                Dropout(args.dropout_gen),
                ConvTranspose((4,4), 256=>128; init = Flux.glorot_uniform, stride=2, pad=3),
                BatchNorm(128),
                x-> leakyrelu.(x, 0.2f0),
                Dropout(args.dropout_gen),
                ConvTranspose( (4,4), 128=>3; init = Flux.glorot_uniform, stride=2, pad=3),
                BatchNorm(3, tanh) ) |> gpu
    Generator(g_latent, g_model)
end

function (m::Generator)(x, y)
    #return (m.g_model(m.g_latent(x)) + y)
    return (m.g_model(m.g_latent(x)))
end

struct Discriminator
    d_model  
end

function discriminator(args)
    d_model = Chain(Conv((4,4), 3=>128, pad=3, init = Flux.glorot_uniform, stride=2),
                    BatchNorm(128),
                    x-> leakyrelu.(x, 0.2f0), 
                    Dropout(args.dropout_disc),      
                    Conv((4,4), 128=>256, pad=3, init = Flux.glorot_uniform, stride=2),
                    BatchNorm(256),
                    x-> leakyrelu.(x, 0.2f0),
                    Conv((4,4), 256=>512, pad=2, init = Flux.glorot_uniform, stride=2),
                    BatchNorm(512),
                    x-> leakyrelu.(x, 0.2f0),
                    #Dropout(args.dropout_disc),     
                    Conv((4,4), 512=>1024, pad=2, init = Flux.glorot_uniform, stride=2),
                    BatchNorm(1024),
                    x-> leakyrelu.(x, 0.2f0),
                    #Dropout(args.dropout_disc),     
                    x-> reshape(x, :, size(x, 4)),
                    Dense(16*1024, 1)) |> gpu
    Discriminator(d_model)
end

function (m::Discriminator)(x)
    return m.d_model(x)
end

function load_data(hparams)
    # load all images
    #println(readdir(hparams.trainPath))
    paths = readdir(hparams.trainPath)
    images = zeros((28,28,3,length(paths)))
    i = 1
    for path in paths
        images[:,:,:,i] = permutedims(channelview(load(string(hparams.trainPath, path))), (2,3,1))
        i = i + 1
    end
    # break into batches
    data = [images[:,:,:,r] for r in partition(1:length(paths),hparams.batch_size)]
    #println(size(data))
    paths = readdir(hparams.champPath)
    champs = zeros((28,28,3,length(paths)))
    i = 1
    for path in paths
        champs[:,:,:,i] = permutedims(channelview(load(string(hparams.champPath, path))), (2,3,1))
        i = i + 1
    end
    return (data, champs)
end

function load_paths(hparams)
    return(readdir(hparams.trainPath), readdir(hparams.champPath))
end

function load_champs(hparams, champPaths)
    champs = zeros((28,28,3,length(champPaths)))
    i = 1
    for path in champPaths
        champs[:,:,:,i] = permutedims(channelview(load(string(hparams.champPath, path))), (2,3,1)).*2f0 .- 1f0
        i = i + 1
    end
    return champs
end

function load_batch_data!(hparams, batchPaths, batch, image_noise, noise_ratio)
    i = 1
    randn!(image_noise)

    for path in batchPaths
        batch[:,:,:,i] = permutedims(channelview(load(string(hparams.trainPath, path))), (2,3,1)).*2 .-1
        i += 1
    end
    batch = noise_ratio.* image_noise .+ batch
end

# Loss functions
function discr_loss(real_output, fake_output)
    return (logitbinarycrossentropy(real_output, 0.9f0) + logitbinarycrossentropy(fake_output, 0.1f0))
end

generator_loss(fake_output) = logitbinarycrossentropy(fake_output, 0.9f0)

function train_discr(discr, fake_data, original_data, opt_discr, disParams)
    loss, back = Zygote.pullback(disParams) do
        discr_loss(discr(original_data), discr(fake_data))
    end
    update!(opt_discr, disParams, back(1f0))
    return loss
end

Zygote.@nograd train_discr

function train_gan(gen, discr, original_data, opt_gen, opt_discr, genParams, disParams, noise, champs)
    # Random Gaussian Noise and Labels as input for the generator
    loss = Dict()####
    randn!(noise)
    loss["gen"], back = Zygote.pullback(genParams) do
            fake = gen(noise, champs)
            #add_portraits!(fake,champs)
            loss["discr"] = train_discr(discr, fake, original_data, opt_discr, disParams)
            generator_loss(discr(fake))
    end
    update!(opt_gen, genParams, back(1f0))
    return loss
end

function add_portraits!(fakes, champs)
    for i in 1:size(fakes)[ndims(fakes)]
        fakes[:,:,:,i] = fakes[:,:,:,i] + champs[:,:,:,rand(1:size(champs)[ndims(champs)])]
    end
end

function get_portraits!(champs, portraits, champNums)
    #shuffle!(champNums)
    for i in 1:size(portraits)[ndims(portraits)]
        portraits[:,:,:,i] = champs[:,:,:,Int32(rand(1:size(champNums)[ndims(champNums)]))]
    end
end

function shufflePaths!(hparams, paths, batchPaths)
    shuffle!(paths)
    i = 1
    for batch in partition(paths,hparams.batch_size)
        batchPaths[i] = batch
        i += 1
    end
end

function train(; kws...)
    hparams = HyperParams(kws...)

    # load data (1 batch at a time)
    paths, champPaths = load_paths(hparams)
    # break into batches
    batchPaths = [paths[r] for r in partition(1:length(paths),hparams.batch_size)]
    champs = load_champs(hparams, champPaths) |> gpu
    noise_ratio = 1f0

    # data containers
    fixed_noise = randn(hparams.latent_dim, hparams.outDims * hparams.outDims)  |> gpu
    noise = randn!(similar(fixed_noise, (hparams.latent_dim, hparams.batch_size))) |> gpu
    image_array = colorview(RGB, permutedims(zeros(28*hparams.outDims, 28*hparams.outDims, 3),(3,1,2)))
    batch = similar(fixed_noise, (28,28,3,hparams.batch_size))#zeros((28,28,3,hparams.batch_size))
    image_noise = randn!(similar(batch, (28,28,3,hparams.batch_size)))
    champNums = [i for i in 1:length(champPaths)]
    portraits = similar(batch,(28,28,3,hparams.batch_size)) # [champs[:,:,:,i] for i in 1:hparams.batch_size]
    fixed_portraits = champs[:,:,:,1:hparams.outDims * hparams.outDims] 
    @info "Data Containers Initialized"

    # Models
    dscr = discriminator(hparams) 
    gen =  generator(hparams)

    # Parameters
    genParams = params(gen.g_latent, gen.g_model)
    disParams = params(dscr.d_model) 
    @info "Network Initialized"


    # Check if the `output` directory exists or needed to be created
    isdir("output")||mkdir("output")
    isdir("output/checkpoints")||mkdir("output/checkpoints")
    isdir("output/images")||mkdir("output/images")

    # Optimizers
    opt_dscr = ADAM(hparams.lr_dscr, (hparams.beta_dscr, hparams.beta2_dscr))
    opt_gen = ADAM(hparams.lr_gen, (hparams.beta_gen, hparams.beta2_gen))

    @info "Starting Training"
    # Training
    genLoss = []
    dscLoss = []
    train_steps = 0
    img_seen = 0
    for ep in 1:hparams.epochs
        @info "Epoch $ep"
        for batchPath in batchPaths
            # Update discriminator and generator
            if length(batchPath) < hparams.batch_size
                continue
            end
            get_portraits!(champs, portraits, champNums)
            load_batch_data!(hparams, batchPath, batch, image_noise, noise_ratio)
            batch |> gpu
            loss = train_gan(gen, dscr, batch, opt_gen, opt_dscr, genParams, disParams, noise, portraits)

            img_seen += size(batch)[ndims(batch)]
            @info("Epoch:$(ep) step:$(train_steps), Discriminator loss = $(loss["discr"]), Generator loss = $(loss["gen"]), Seen $(img_seen) images")
            if train_steps % hparams.verbose_freq == 0  ## verbose frequency
                # Save generated fake image
                create_output_image!(gen, fixed_noise, fixed_portraits, hparams, image_array)
                save(@sprintf("output/images/mGAN%06d.png", train_steps), image_array)
                #save(@sprintf("output/masks/mGAN%06d.png", train_steps), image_array)
                # save model weights in checkpoint
                @save "output/checkpoints/GENcheckpoint.bson" genParams opt_gen
                @save "output/checkpoints/DIScheckpoint.bson" disParams opt_dscr
                #display(plot([x for x in 1:train_steps], [dscLoss, genLoss], title = "Loss vs Iterations", label = ["Discriminator" "Generator"]))

                noise_ratio = max(0f0, noise_ratio - hparams.noise_reduc)
            end

            if train_steps % hparams.checkpoint_freq == 0
                # save model weights as train steps
                @save (@sprintf("output/checkpoints/GEN_%06d.bson", train_steps)) genParams
                @save (@sprintf("output/checkpoints/DIS_%06d.bson", train_steps)) disParams
            end
            append!(genLoss, loss["gen"])
            append!(dscLoss, loss["discr"])
            train_steps += 1
            
            batch |> cpu
        end
        shufflePaths!(hparams, paths, batchPaths)
    end

    create_output_image!(gen, fixed_noise, fixed_portraits, hparams, image_array)
    save("output/mGAN_final.png", image_array)
    @save "output/GEN_Final.bson" genParams
    @save "output/DIS_Final.bson" disParams
    return 
end    


function create_output_image!(gen, fixed_noise, fixed_portraits, hparams, image_array)
    @eval Flux.istraining() = false
    fake_images = gen(fixed_noise, fixed_portraits) |>cpu
    @eval Flux.istraining() = true
    for i in 1:hparams.outDims
        for j in 1:hparams.outDims
            image_array[i*28-27:i*28,j*28-27:j*28] .= clamp01!(colorview(RGB, permutedims((fake_images[:,:,:,hparams.outDims*(i-1)+j].+1f0)./2f0,(3,1,2))))
        end
    end
    return Nothing
end

cd(@__DIR__)
fixed_labels = train()
println("Done")

# TO-DO list 
# 11) Add way to view gradients
# 13) Add noise to labels