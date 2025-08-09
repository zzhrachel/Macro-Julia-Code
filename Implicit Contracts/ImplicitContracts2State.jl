# Purpose: This code solves a simple 2-state implicit contract model in order to simulate some wage dynamics.

using Distributions
using PythonPlot
using Random
include("rouwenhorst.jl")
include("fcsolve.jl")

println("Loaded")

struct Model_Parameters
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

function util(c :: Real, chi :: Real)
    u = ((1/c)^(chi-1)-1)/(1-chi)
    return u
end

function util(c :: Float64, chi :: Real)
    u = (c^(1-chi)-1)/(1-chi)
    return u
end

function util(c :: Array, chi :: Real)
    u = (c.^(1-chi)-1)./(1-chi)
    return u
end

function State2WageBounds(S2Vec :: Array{Float64,1}, Params :: Model_Parameters)
    B   = Params.B
    chi = Params.chi
    n   = Params.n
    PIU = Params.PIU
    Ut  = Params.Ut

    maxV2 = S2Vec[1]
    maxV1 = S2Vec[2]
    minV1 = S2Vec[3]
    U2    = Ut[2]

    minw  = ((1-chi)*(U2 - B*PIU[2,1]*minV1 - B*PIU[2,2]*U2)+1)^(1/(1-chi))    
    # Set maximum state-2 wage to equal y.
    resmaxV2 = maxV2 - util(y,chi) - B*PIU[2,1]*maxV1 - B*PIU[2,2]*maxV2
    resmaxV1 = maxV1 - util(y,chi) - B*PIU[1,1]*maxV1 - B*PIU[1,2]*maxV2
    resminV1 = minV1 - util(minw,chi) - B*PIU[1,1]*minV1 - B*PIU[1,2]*U2

    res = [resmaxV2 ; resmaxV1 ; resminV1]
end

function State1WageBounds(Params :: Model_Parameters)
    B   = Params.B
    chi = Params.chi
    n   = Params.n
    PIU = Params.PIU
    Ut  = Params.Ut

    U2  = Ut[2]
    U1  = Ut[1]
    maxw  = ((1-chi)*(U2 - B*PIU[1,1]*U2 - B*PIU[1,2]*U2)+1)^(1/(1-chi))
    minw  = ((1-chi)*(U1 - B*PIU[1,1]*U1 - B*PIU[1,2]*U2)+1)^(1/(1-chi))
    return minw, maxw
end

function InitialWage(V0 :: Float64, Ut :: Array{Float64,1}, PIU :: Array{Float64,2}, Params :: Model_Parameters)
    n = Params.n
    # Find interval bounds sandwiching the initial promised utility
    i = searchsortedlast(Ut,V0)
    println("The initial state is state: $(i)")
    if i == 1    
        Vsum = V0 - B*PIU[1,1]*V0 - B*PIU[1,2]*Ut[2]
        w0 = ((1-chi)*Vsum+1)^(1/(1-chi))
    else
        error("This code is for a 1-state model with an initial promised utilities below U(2)!")
    end
    return w0
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
    Random.seed!(123)             # Reset the random number generator to use the seed given by "seed"
    dU      = Uniform(0,1)
    p       = rand(dU,T)    # Draw T realizations of a random variable that is uniformly distributed 
                            # over the [0,1] interval.  These can be treated as probabilities.
    drw     = convert(Array{Int64,1},dropdims(zeros(T,1),dims=2)) # The j's will be the TFP realizations.
    drw[1]  = s0
    for k in 2 : T
        drw[k]    = minimum(findall(cumPI[drw[k-1],:] .> p[k]))
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
y = 10.0    # Period output
B = 0.99    # Discount factor (beta in the notes)
chi = 2     # Curvature in CRRA utility function

# Set-up stochastic process as a 2-state Markov Chain:
rho = 0.9
sigma = 0.01
n = 2
UGrid, PIU = rouwenhorst(rho,sigma,n)
Ut  = util(0.6*y,chi)/(1-B)*exp.(UGrid)  # Note: Ensure that y > 1 or else utilities are negative and the order
                                         # of utilites from highest to lowest is backward in functions above.

Params = Model_Parameters(y,B,chi,rho,sigma,n,UGrid,PIU,Ut)

# Solve for state 2 wage bounds:
S2Vec = dropdims(2*ones(3,1),dims=2) # fcsolve.jl is written to accept initial value arrays as Array{Float64,1}
options = [1000 ; 1e-6 ; 1e-8 ; 1e-3; 1]
S2soln = fcsolve(State2WageBounds,S2Vec,[],Params)
maxV2 = S2soln[1]
maxV1 = S2soln[2]
minV1 = S2soln[3]
maxw2 = y
minw2 = ((1-chi)*(Ut[2] - B*PIU[1,2]*minV1 - B*PIU[2,2]*Ut[2])+1)^(1/(1-chi))
minw1, maxw1 = State1WageBounds(Params)

println("-----------------------------------------------------------------------------------")
println("                                                                                   ")
println("The wage bounds in State 2 are: $([minw2,y])                                       ")
println("The wage bounds in State 1 are: $([minw1,maxw1])                                   ")
println("The promised utilities in state 2 at the upper wage bound are: $([maxV1,maxV2])    ")
println("The promised utilities in state 2 at the lower wage bound are: $([minV1,Ut[2]])    ")
println("The promised utilities in state 1 at the upper wage bound is: $(Ut[1])             ")
println("The promised utilities in state 1 at the lower wage bound is: $(Ut[2])             ")
println("                                                                                   ")
println("-----------------------------------------------------------------------------------")


# Provide a simulation:
WageBounds = [minw1 maxw1 ; minw2 y]
T   = 100  # Number of periods in the simulation
V0  = mean(Ut[1:2])  # Initial value of Promised Utility
ChainU, statevec = SimulateUStates(T, PIU, V0, Ut)
statevec = convert(Array{Int64,1},dropdims(statevec,dims=2))
Wages = zeros(T,1)
w0  = InitialWage(V0, Ut, PIU, Params)
Wages[1] = w0
for i in 2 : T
    if statevec[i] .> statevec[i-1]
        if Wages[i-1].< WageBounds[2,1]
            Wages[i] = WageBounds[2,1]
        else
            Wages[i] = Wages[i-1]
        end
    else
        Wages[i] = Wages[i-1]
    end
end

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
subplot(2,1,1)
if T <= 25
    plot(collect(1:T),Wages,linewidth=2,linestyle="-",marker="o",color="b")
else
    plot(collect(1:T),Wages,linewidth=2,linestyle="-",color="b")
end
plot(collect(1:T),ones(T,1)*WageBounds[1,1],linestyle="-.",linewidth=1,color="r")
plot(collect(1:T),ones(T,1)*WageBounds[1,2],linestyle="-.",linewidth=1,color="r")
plot(collect(1:T),ones(T,1)*WageBounds[2,1],linestyle="-.",linewidth=1,color="g")
plot(collect(1:T),ones(T,1)*WageBounds[2,2],linestyle="-.",linewidth=1,color="g")
#xlabel("Period",fontdict=font3)
ylabel("Wage",fontdict=font3)
title("Wage Dynamics",fontdict=font1)
axis([1,T,minimum(WageBounds)-0.5,maximum(WageBounds)+0.5])
subplot(2,1,2)
plot(collect(1:T),Ut[statevec],linewidth=2,linestyle="-",color="b")
xlabel("Period",fontdict=font3)
ylabel("Value of Outside Option",fontdict=font3)
title("Dynamics of Outside Option",fontdict=font1)
axis([1,T,minimum(Ut)-1,maximum(Ut)+0.5])

println("All Done")

