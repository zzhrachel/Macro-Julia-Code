# THIS CODE NEEDS TO BE FIXED!  SEE 2 STATE MODEL.  THIS CODE NEEDS TO CALCULATE THE WAGE BANDS FOR EACH STATE... I'VE ONLY CALCULATED LOWER BANDS!

using Distributions
using PyPlot, PyCall
include("rouwenhorst.jl")
include("fcsolve.jl")

type Model_Parameters
    y :: Real
    B :: Float64
    chi :: Real
    rho :: Float64
    sigma :: Float64


    n  :: Int64
    UGrid :: Array{Float64,1}
    PIU :: Array{Float64,2}
    Ut :: Array{Float64,1}
end

function util(c :: Float64, chi :: Real)
    u = (c^(1-chi)-1)/(1-chi)
    return u
end

function util(c :: Array, chi :: Real)
    u = (c.^(1-chi)-1)./(1-chi)
    return u
end

function SolvePeriodThree(PV3 :: Array{Float64,1}, Params :: Model_Parameters)
    B   = Params.B
    chi = Params.chi
    n   = Params.n
    PIU = Params.PIU
    Ut  = Params.Ut

    U3 = Ut[3]
    V1 = PV3[1]
    V2 = PV3[2]
    w  = ((1-chi)*(U3 - B*PIU[3,1]*V1 - B*PIU[3,2]*V2 - B*PIU[3,3]*U3)+1)^(1/(1-chi))
    # Define residuals:
    res1 = V1 - util(w,chi) - B*PIU[1,1]*V1 - B*PIU[1,2]*V2 - B*PIU[1,3]*U3
    res2 = V2 - util(w,chi) - B*PIU[2,1]*V1 - B*PIU[2,2]*V2 - B*PIU[2,3]*U3
    res = [res1 ; res2]
    return res
end

function SolvePeriodTwo(PV2 :: Array{Float64,1}, Params :: Model_Parameters)
    B   = Params.B
    chi = Params.chi
    n   = Params.n
    PIU = Params.PIU
    Ut  = Params.Ut

    U3  = Ut[3]
    U2  = Ut[2]
    V1  = PV2[1,1]
    w   = ((1-chi)*(U2 - B*PIU[2,1]*V1 - B*PIU[2,2]*U2 - B*PIU[2,3]*U3)+1)^(1/(1-chi))
    # Define residuals:
    res = [V1 - util(w,chi) - B*PIU[1,1]*V1 - B*PIU[1,2]*U2 - B*PIU[1,3]*U3]  # res is of type Array{Float64,1}
    return res
end

function InitialWage(V0 :: Float64, Ut :: Array{Float64,1}, PIU :: Array{Float64,2}, Params :: Model_Parameters)
    n = Params.n
    # Find interval bounds sandwiching the initial promised utility
    i = searchsortedlast(Ut,V0)
    println("The initial state is state: $(i)")
    if i == 1    
        Vsum = V0 - B*PIU[1,1]*V0
        for j in 2 : n
            Vsum -= B*PIU[i,j]*Ut[j]
        end
        w = ((1-chi)*Vsum+1)^(1/(1-chi))
    else
        error("This code is for a 3-state model with an initial promised utilities below U(2)!")
    end
    return w
end

function SimulateUStates(T :: Int64, PI :: Array{Float64,2}, V0 :: Float64, Ut :: Array{Float64,1})
    s0 = searchsortedlast(Ut,V0)  # Initial state of wage intervals
    n  = Params.n
    # First simulate a sequence for the income.
    cumPI   = zeros(n,n)
    for k in 1 : n
        tmp = 0
        for kk in 1 : n
            tmp         = tmp + PI[k,kk]
            cumPI[k,kk] = tmp
        end
    end
    srand(127)             # Reset the random number generator to use the seed given by "seed"
    dU      = Uniform(0,1)
    p       = rand(dU,T)    # Draw T realizations of a random variable that is uniformly distributed 
                            # over the [0,1] interval.  These can be treated as probabilities.

    drw     = convert(Array{Int64,1},squeeze(zeros(T,1),2)) # The j's will be the TFP realizations.
    drw[1]  = s0
    for k in 2 : T
        drw[k]    = minimum(find(cumPI[drw[k-1],:] .> p[k]))
    end
    ChainU  = Ut[drw]
    # Now construct a vector indicating the state for income in each period
    statevec    = zeros(T,1)
    for kk in 1 : T
        for k in 1 : n
            if ChainU[kk] == Ut[k]
                statevec[kk]   = k
            end
        end            
    end
    return ChainU, statevec
end

#----------------------------------------------------------------------------------------------------------

########################
#   Solve the Model    #
########################

# Define parameter values:
y = 1       # Period output
B = 0.99    # Discount factor (beta in the notes)
chi = 2     # Curvature in CRRA utility function

# Set-up stochastic process as a 3-state Markov Chain:
rho = 0.99
sigma = 0.1
n = 3
UGrid, PIU = rouwenhorst(rho,sigma,n)
Ut  = exp.(UGrid)

Params = Model_Parameters(y,B,chi,rho,sigma,n,UGrid,PIU,Ut)
PV3_0 = squeeze(zeros(2,1),2) # fcsolve.jl is written to accept initial value arrays as Array{Float64,1}

options = [1000 ; 1e-6 ; 1e-8 ; 1e-3; 1]
PV3 = fcsolve(SolvePeriodThree,PV3_0,[],Params)
w3  = ((1-chi)*(Ut[3] - B*PIU[3,1]*PV3[1] - B*PIU[3,2]*PV3[2] - B*PIU[3,3]*Ut[3])+1)^(1/(1-chi))
PV2_0 = [-1.0]  # Constructed this way, PV2_0 is of type Array{Float64,1}
PV2 = fcsolve(SolvePeriodTwo,PV2_0,[],Params)
w2  = ((1-chi)*(Ut[2] - B*PIU[2,1]*PV2[1] - B*PIU[2,2]*Ut[2] - B*PIU[2,3]*Ut[3])+1)^(1/(1-chi))
w1  = ((1-chi)*(Ut[1] - B*PIU[1,1]*Ut[1] - B*PIU[1,2]*Ut[2] - B*PIU[1,3]*Ut[3])+1)^(1/(1-chi))
WageBounds = [w1 ; w2 ; w3]

println("-----------------------------------------------------------")
println("                                                           ")
println("The wage bounds are: $(WageBounds)                         ")
println("The promised utilities in state 3 are: $(vcat(PV3,Ut[3]))  ")
println("The promised utilities in state 2 are: $(vcat(PV2,Ut[2:3]))")
println("The promised utilities in state 1 are: $(Ut)               ")
println("                                                           ")
println("-----------------------------------------------------------")


# Provide a simulation:
T   = 250  # Number of periods in the simulation
V0  = mean(Ut[1:2])  # Initial value of Promised Utility
ChainU, statevec = SimulateUStates(T, PIU, V0, Ut)
statevec = convert(Array{Int64,1},squeeze(statevec,2))
Wages = zeros(T,1)
w0  = InitialWage(V0, Ut, PIU, Params)
println([V0,w0])
Wages[1] = w0
for i in 2 : T
    if statevec[i] > statevec[i-1]
        Wages[i] = WageBounds[statevec[i]]
    else
        Wages[i] = Wages[i-1]
    end
end

fig_plot = 1
if fig_plot == 1
    ## Plot Figure:

    # Set up font dictionaries:
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


    fig = figure("Implicit_Contracts",figsize=(14*1.05,8*1.05))
    if T <= 25
        plot(collect(1:T),Wages,linewidth=2,linestyle="-",marker="o",color="b")
    else
        plot(collect(1:T),Wages,linewidth=2,linestyle="-",color="b")
    end
    plot(collect(1:T),ones(T,1)*WageBounds[1],linestyle="-.",linewidth=1,color="r")
    plot(collect(1:T),ones(T,1)*WageBounds[2],linestyle="-.",linewidth=1,color="r")
    plot(collect(1:T),ones(T,1)*WageBounds[3],linestyle="-.",linewidth=1,color="r")
    xlabel("Period",fontdict=font3)
    ylabel("Wage",fontdict=font3)
    title("Simulated Wage Dynamics",fontdict=font1)
    axis([1,T,WageBounds[1]-0.01,WageBounds[3]+0.01])

end
println("All Done")

