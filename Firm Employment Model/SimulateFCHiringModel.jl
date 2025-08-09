# Simulate the dynamics of firm hiring using the fixed cost of labour adjustment model.
using Distributions
using PythonPlot
using Random
using HDF5, JLD

include("BilinearInterp2D.jl")
include("LinearInterp1D.jl")

solve_sep = 0
if solve_sep == 1
    struct ParameterValues
        r :: Float64
        w :: Float64
        eta :: Float64
        alpha :: Float64
        rho :: Float64
        phi :: Float64
    end

    struct SolutionValues
        V :: Array{Float64,2}
        J :: Array{Float64,2}
        DRH :: Array{Float64,2}

        rhoz :: Float64
        sez :: Float64
        mZ :: Float64
        PIZ :: Array{Float64,2}
        Zt :: Array{Float64,1}

        NVec :: Array{Float64,2}
        ZVec :: Array{Float64,2}
        NPVGrid :: Array{Float64,1}
        NPJGrid :: Array{Float64,1}
        nsize :: Int64
        npsize :: Int64
        nstate :: Int64
    end
    SolutionResults, Params = load("/Users/jacobwong/Dropbox/Macro IV Computer Codes/Julia Codes/Post-Version 1.0/Firm Employment Model/LabourAdjModelFCwSepHiWHiSE.jld",
                                    "SolutionResults","Params")

    function SimulateModelwSep(T :: Int64, ChainZ :: Array, statevec :: Array, SolutionResults :: SolutionValues, Params :: ParameterValues)
        NVec = SolutionResults.NVec
        ZVec = SolutionResults.ZVec
        DRH  = SolutionResults.DRH
        nsize = SolutionResults.nsize
        nstate = SolutionResults.nstate
        delta = Params.delta    

        NMat = reshape(NVec,nsize,nstate)
        ZMat = reshape(ZVec,nsize,nstate)
        DRHM = reshape(DRH,nsize,nstate)
        Hires = zeros(T,1)
        Labour = zeros(T,1)
        Labour[1] = minimum(NVec)
        for t in 1 : T
            ht = LinearInterp1D(NMat[:,statevec[t]],DRHM[:,statevec[t]],Labour[t])
            #ht = BilinearInterp2D(NMat,ZMat,DRHM,Labour[t],ChainZ[t])
            Hires[t] = ht
            if t < T
                Labour[t+1] = (1-delta)Labour[t] + ht
            end 
        end       
        return Hires, Labour
    end
else
    struct SolutionValues
        V :: Array{Float64,2}
        J :: Array{Float64,2}
        DRH :: Array{Float64,2}

        rhoz :: Float64
        sez :: Float64
        mZ :: Float64
        PIZ :: Array{Float64,2}
        Zt :: Array{Float64,1}

        NVec :: Array{Float64,2}
        ZVec :: Array{Float64,2}
        NPGrid :: Array{Float64,1}
        nsize :: Int64
        npsize :: Int64
        nstate :: Int64
    end    
    SolutionResults = load("/Users/jacobwong/Dropbox/Macro IV Computer Codes/Julia Codes/Post-Version 1.0/Firm Employment Model/LabourAdjModelFCHiWHiSE.jld",
                                    "SolutionResults")

    function SimulateModel(T :: Int64, ChainZ :: Array, statevec :: Array, SolutionResults :: SolutionValues)
        NVec = SolutionResults.NVec
        ZVec = SolutionResults.ZVec
        DRH  = SolutionResults.DRH
        nsize = SolutionResults.nsize
        nstate = SolutionResults.nstate    

        NMat = reshape(NVec,nsize,nstate)
        ZMat = reshape(ZVec,nsize,nstate)
        DRHM = reshape(DRH,nsize,nstate)
        Hires = zeros(T,1)
        Labour = zeros(T,1)
        Labour[1] = minimum(NVec)
        for t in 1 : T
            ht = LinearInterp1D(NMat[:,statevec[t]],DRHM[:,statevec[t]],Labour[t])
            #ht = BilinearInterp2D(NMat,ZMat,DRHM,Labour[t],ChainZ[t])
            Hires[t] = ht
            if t < T
                Labour[t+1] = Labour[t] + ht
            end 
            println(t)
        end    
        return Hires, Labour
    end    
end

function SimulateProductivity(T :: Int64, SolutionResults :: SolutionValues)
    PIZ = SolutionResults.PIZ
    Zt  = SolutionResults.Zt
    nstateZ = size(PIZ,1)
    # First simulate a sequence for the productivity.
    cumPI   = zeros(nstateZ,nstateZ)
    for k in 1 : nstateZ
        tmp = 0
        for kk in 1 : nstateZ
            tmp         = tmp + PIZ[k,kk]
            cumPI[k,kk] = tmp
        end
    end

    s0      = convert(Int64,ceil(nstateZ/2))  # Set the initial state to equal the state with the mean of the
                                             # TFP process. (There are nstate states.)
    Brn     = 0
    nsim    = Brn+T         # Total number of periods to simulate.
    Random.seed!(123)       # Reset the random number generator to use the seed given by "seed"
    dU      = Uniform(0,1)
    p       = rand(dU,nsim) # Draw nsim realizations of a random variable that is uniformly distributed over the
                            # [0,1] interval.  These are treated as probabilities.
    drw     = convert(Array{Int64,1},dropdims(zeros(nsim,1),dims=2)) # The j's will be the TFP realizations.
    drw[1]  = s0
    for k in 2 : nsim
        drw[k]    = minimum(findall(cumPI[drw[k-1],:] .> p[k]))
    end
    ChainZ  = Zt[drw]
    # Now construct a vector indicating the state for TFP in each period
    statevec    = zeros(nsim,1)
    for kk in 1 : nsim
        for k in 1 : nstateZ
            if ChainZ[kk] == Zt[k]
                statevec[kk]   = k
            end
        end            
    end
    return ChainZ, statevec, Brn
end

function Construct_pdfs(T :: Int64, statevec :: Array, Hires :: Array, Labour :: Array, SolutionResults :: SolutionValues)
    # Construct the histogram grids
    nsize = SolutionResults.nsize
    nstate = SolutionResults.nstate
    NGrid = SolutionResults.NVec[1:nsize,1]
    NBins = NGrid[1:end-1]
    minH  = minimum(Hires)
    maxH  = maximum(Hires)
    hsize = 101
    HGrid = collect(range(minH,stop=maxH,length=hsize))
    HBins = HGrid[1:end-1]
    NHist = zeros(nsize-1,1)
    HHist = zeros(hsize-1,1)
    for t in 1 : T
        i = searchsortedlast(NBins,Labour[t])
        j = searchsortedlast(HBins,Hires[t])
        NHist[i] += 1
        HHist[j] += 1
    end
    NHist  = NHist./T
    HHist  = HHist./T
    ndelta = NBins[2] - NBins[1]
    hdelta = HBins[2] - HBins[1]
    n_cdf  = [0 ; cumsum(NHist,dims=1)]
    h_cdf  = [0 ; cumsum(HHist,dims=1)]
    n_pdf  = (n_cdf[2:end] - n_cdf[1:end-1])./ndelta
    h_pdf  = (h_cdf[2:end] - h_cdf[1:end-1])./hdelta
    return n_pdf, h_pdf
end

#---------------------
# Simulate the Model:
#---------------------

T   = 1000000
ChainZ, statevec, Brn = SimulateProductivity(T, SolutionResults)
# Drop the burn in periods t = 1 through Brn:
ChainZ = ChainZ[Brn+1:Brn+T]
statevec = convert(Array{Int64,1},statevec[Brn+1:Brn+T])
# Simulate the model to produce a time-series for hires and the firm's labour force.
if solve_sep == 0
    Hires, Labour = SimulateModel(T, ChainZ, statevec, SolutionResults)
else
    Hires, Labour = SimulateModelwSep(T, ChainZ, statevec, SolutionResults,Params)
end
n_pdf, h_pdf = Construct_pdfs(T, statevec, Hires, Labour, SolutionResults)

nsize = SolutionResults.nsize
hsize = 101
minH  = minimum(Hires)
maxH  = maximum(Hires)
HGrid = collect(range(minH,stop=maxH,length=hsize))
HMidPt = 0.5*(HGrid[1:hsize-1] + HGrid[2:hsize])
h_delta = HGrid[2] - HGrid[1]

plot_figs = 1
if plot_figs == 1
    # Define Font Dictionaries:
    font1 = Dict("family"=>"serif",
    # "name" => "times",
    "name" => "Times New Roman",
    #"color"=>"darkred",
    #"weight"=>"normal",
    "size"=>14)
    #xlabel("Time",fontdict=font1)

    font2 = Dict("family"=>"serif",
    # "name" => "times",
    "name" => "Times New Roman",
    #"color"=>"darkred",
    #"weight"=>"normal",
    "size"=>16)
    #xlabel("Time",fontdict=font1)

    font3 = Dict("family"=>"serif",
    # "name" => "times",
    "name" => "Times New Roman",
    #"color"=>"darkred",
    #"weight"=>"normal",
    "size"=>12)
    #xlabel("Time",fontdict=font1)

    hire_plot = 0
    if hire_plot == 1
        fig = figure("Hires_Plot")
        subplot(3,1,1)
        plot(collect(1:T),Hires[1:T],linewidth = 1, color = "b")
        axis([1,T,minimum(Hires),maximum(Hires)])
        title("Hires")
        subplot(3,1,2)
        plot(collect(1:T),Labour[1:T],linewidth = 1, color = "b")
        axis([1,T,0,1])
        title("Firm's Labour Force")
        subplot(3,1,3)
        plot(collect(1:T),ChainZ[1:T],linewidth = 1, color = "b")
        axis([1,T,minimum(SolutionResults.Zt),maximum(SolutionResults.Zt)])
        xlabel("Period")
        title("TFP")
    end

    pdf_plot = 1
    if pdf_plot == 1
        # Plot Histogram:
        nsize = SolutionResults.nsize
        nstate = SolutionResults.nstate
        NGrid = SolutionResults.NVec[1:nsize,1]
        NMidPt = 0.5*(NGrid[1:nsize-1] + NGrid[2:nsize])
        n_delta = NGrid[2] - NGrid[1]

        fig = figure("Surface_Decision_Rule")
        subplot(1,2,1)
        bar(NMidPt,n_pdf,n_delta)
        #plot(NMidPt,n_pdf)
        xlabel("Employment",fontdict=font1)
        ylabel("p.d.f.",fontdict=font1)
        axis([minimum(NMidPt),maximum(NMidPt),0,maximum(n_pdf)])
        subplot(1,2,2)
        bar(HMidPt,h_pdf,h_delta)
        #plot(HMidPt,h_pdf)
        xlabel("Hires",fontdict=font1)
        ylabel("p.d.f.",fontdict=font1)
        axis([minimum(HMidPt),maximum(HMidPt),0,maximum(h_pdf)])
    end
end

println("All Done")
