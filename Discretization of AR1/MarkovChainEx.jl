using Distributions
using Random
using PythonPlot
using LaTeXStrings
include("rouwenhorst.jl")

#-----------------------------------------------
# PURPOSE: This code uses various routines to construct a Markov-Chain approximation
#          to an AR(1) process.  Let the AR(1) process be
#
#          ln(x') = (1-rho)ln(mX) + rho ln(x) + e(t).
#
#----------------------------------------------

#---------------------------------
# Constructing the Markov Chain : 
#---------------------------------

rho     = 0.80    # Persistance of x
se      = 0.01    # Standard deviation of shocks to x
mX      = 0       # Mean of x
nstate  = 5       # Number of states for x
Xgrid, PI = rouwenhorst(rho,se,nstate)
rX  = Xgrid .+ mX
eXt = exp.(rX)

# Set up a matrix of the CMF (cumulative mass function) for the transition probabilities in PI
cumPI   = zeros(nstate,nstate)
for k   = 1 : nstate
    tmp = 0
    for kk = 1 : nstate
        tmp = tmp + PI[k,kk]
        cumPI[k,kk] = tmp
    end
end


## Simulate a time series for X
s0      = ceil(nstate/2)   # Set the initial state to equal the state with the mean of the x process. 
                           # (There are nstate states.)
Brn     = 0                # Number of "burn-in" periods to use in simulations
Nt      = Brn + 100000     # Total number of periods to simulate.
Random.seed!(27213)        # Store the seed for the vector of random numbers to be used in the simulations
dU      = Uniform(0,1)     # Assign to dU the Distributions type Uniform(0,1)
p       = rand(Nt,1);      # Draw T realizations of a random variable that is uniformly distributed over
                           # the [0,1] interval.
drw     = convert(Array{Int64,1},dropdims(zeros(Nt,1),dims=2)) # The drw will be the indices of the draws
                                                               # for the state.
drw[1]  = s0
for k in 2 : Nt
    # Final all entries in the row of cumPI corresponding to the index of the current value of the state
    # variable that are less than the kth draw from the uniform distribution.  Store the index of the 
    # element in the row of cumPI that is the lowest value greater than p[k].
    drw[k]    = minimum(findall(cumPI[drw[k-1],:] .> p[k])) 

end
xChain  = eXt[drw]

# Drop the Burn-in periods:
xChain  = xChain[Brn+1:Nt]
T       = Nt-Brn

## Compare the properties of the Markov-chain to an AR(1) process.

# Draw a sequence of shocks, e(t), of length Nt.
Random.seed!(12345)     # Store the seed for the vector of random numbers to be used in the simulations
dN      = Normal(0,se)
evec    = rand(dN,Nt)   # Draw Nt values from the Normal distributions with mean zero and standard deviation se. 
xvec    = zeros(Nt,1)
xvec[1] = mX
for t = 2 : Nt
    xvec[t] = (1-rho)*mX + rho*xvec[t-1] + evec[t]  # Note: ln(x(t)) = (1-rho)ln(mX) + rho ln(x(t-1)) + e(t)
end
xvec    = exp.(xvec[Brn+1:Nt])
corrAR1 = cor(xvec[1:end-1],xvec[2:end])
stdAR1  = std(xvec)
corrMC  = cor(xChain[1:end-1],xChain[2:end])
stdMC   = std(xChain)

println(" ")
println("-----------------------------------------------------------------------")
println("                     RESULTS                                           ")
println("-----------------------------------------------------------------------")
println(" ")
println("The standard devation of the AR(1) process is : $(stdAR1)")
println("The standard devation of the Markov-chain is : $(stdMC)")
println("The relative standard deviation of the processes is : $(stdMC/stdAR1)")
println("The autocorrelation coefficient of the AR(1) process : $(corrAR1)")
println("The autocorrelation coefficient of the Markov-chain : $(corrMC)")
println("The relative autocorrelation of the processes is : $(corrMC/corrAR1)")
println(" ")
println("-----------------------------------------------------------------------")


# Plot simulated data:
plotx = 1
if plotx == 1
    figure
    subplot(1,2,1)
    plot(range(1,stop=T,length=T),xvec)
    xlabel("Period",fontname="times",fontsize="16")
    ylabel(L"e^{x_{t}}",fontname="times",fontsize="16")
    title("A Sample Run: AR(1)",fontname="times",fontsize="16")
    subplot(1,2,2)
    plot(range(1,stop=T,length=T),xChain)
    xlabel("Period",fontname="times",fontsize="16")
    ylabel(L"e^{x_{t}}",fontname="times",fontsize="16")
    title("A Sample Run: Markov-Chain",fontname="times",fontsize="16")
end