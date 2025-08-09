# This code solves for the value functions and the decision rules of the baseline Mortensen-Pissarides 
# search model using the Den Haan, Ramey and Watson (AER) channel system matching function.  In this 
# version we use the standard linear utility function but solve using a BFGS hill-climbing routine.  
# This model should replicate Hagedorn and Manovskii's paper (AER, 2008).

using LaTeXStrings
using PythonPlot
using Random

struct Parameter_Values
    rho :: Real
    beta :: Real
    delta :: Real
    alpha :: Real
    z :: Real
    b :: Real
    mA :: Real
    mD :: Real
    nstateA :: Int64
    nstateD :: Int64
    eAt :: Array{Float64}
    eDt :: Array{Float64}
    PIA :: Array
    PID :: Array
end

# Defined Parameter_Values composite type before including .jl files because some of them call for the Parameter_Values composite type and if it is not
# defined then we have an error message.
include("rouwenhorst.jl")
include("fcsolve.jl")
include("DMP_DRWSS.jl")
include("DMP_DRW_TRes.jl")
include("DMP_DRW_SSS.jl")
include("DMP_DRW_Simulate.jl")


## Parameters :
# This Shimer calibration gets the job finding probability to be 40% and the job filling probability
# to be 71% in the stochastic steady state. (rho = 0.9992, beta = 0.895, delta = 0.0081, alpha = 1.2, z = 0.12, b = 0.4)
rho   = 0.996;  # Discount factor
beta  = 0.895;  # Worker's bargaining weight (0.895)
delta = 0.0081; # Job separation rate (results in a quarterly job separation rate of approximatly 0.0929)
alpha = 1.2;    # Curvature parameter in matching function
z     = 0.12;   # Cost of opening a vacancy
mA    = 0;
mD    = 0;
b     = 0.40*exp(mA); # Unemployment consumption

#--------------------------------------------------------
# Constructing the Markov Chain for Labour Productivity :
#--------------------------------------------------------

rhoA    = 0.9895 # Persistance in productivity
seA     = 0.0034 # Volatility in labour productivity
mA      = 0      # Mean of labour productivity (At = exp(A))
nstateA = 11     # Number of states for labour productivity
Agrid, PIA = rouwenhorst(rhoA,seA,nstateA)
eAt        = exp.(Agrid .+ mA)

#------------------------------------------------------------
# Constructing the Markov Chain for the Job Sepration Rate :
#------------------------------------------------------------

# From Fujita and Ramey, the monthly job separation rate has AR(1) parameter of rho = 0.8346 and the residuals have a standard deviation of 0.0088.
rhoD    = 0.8346 # Persistance in job separation rate
seD     = 0.0088 # Volatility in job separation
mD      = 0      # Mean of job separations (Dt = exp(D)*delta)
nstateD = 5      # Number of states for job separations
Dgrid, PID = rouwenhorst(rhoD,seD,nstateD)
eDt        = exp.(Dgrid)*delta;

Param = Parameter_Values(rho,beta,delta,alpha,z,b,mA,mD,nstateA,nstateD,eAt,eDt,PIA,PID)

# Define useful inline functions :
util(c) = c # Define the function "util" using an inline or assignment form declaration.


##
#----------------------------------------
# Solving for Steady State (see notes) :
#----------------------------------------

x0_ss   = 1
xss     = fcsolve(DMP_DRWSS,x0_ss,[],Param)
thss    = exp(xss)
qss     = 1/((1+thss^-alpha)^(1/alpha))
uss     = exp(mD)*delta/(exp(mD)*delta + thss*qss)
wss     = beta*exp(mA) + (1-beta)*b + beta*z*thss
Jss     = (exp(mA)-wss)/(1-rho*(1-exp(mD)*delta))
Uss     = b/(1-rho) + beta/(1-beta)*z*thss/(1-rho)
Wss     = beta/(1-beta)*Jss + Uss
Sss     = Jss + Wss - Uss
hss     = thss*qss*uss
vss     = thss*uss
jfndss  = 1/((1+thss^(-alpha))^(1/alpha))
jfllss  = 1/((1+thss^(alpha))^(1/alpha))


## Initiate matrices for match surplus and market tightness variables :
scratch = 1
if scratch == 1
    # If starting from scratch :
    #SMat    = ones(nstateA,nstateD)*Sss;
    x0      = log.(ones(nstateA,nstateD)*thss)
    # In the solution routine we will set theta = exp(x0).  This ensures
    # that theta will always be strictly positive.
else
    # If using stored solution values :
    load(file)
    #SMat    = MPDRW.SMat
    THETA   = MPDRW.THETA
    x0      = log.(THETA)
end

solve = 1;
if solve == 1
    # If starting from scratch choose x0 = 2*ones(nstateA*nstateD,1);
    x0        = x0[:]
    options   = [100 ; 1e-8 ; 1e-7 ; 1e-3 ; 1]
    TMat      = fcsolve(DMP_DRW_TRes,x0,[],Param)
    TMat      = reshape(TMat,nstateA,nstateD)
    THETA     = exp.(TMat)  # This ensures that theta is always strictly positive.
else
    x0        = TMat[:]
end


# Solve for equilibrium value of variables in the stochastic steady state
usss, vsss, hsss, jfndsss, jfllsss, wsss = StochasticSSDRW(THETA,Param)

## Print results to the command window (nicely).
println(" ")
println("---------------------------------------------------------------------------")
println("             (STOCHASTIC) STEADY STATE RESULTS                             ")
println("---------------------------------------------------------------------------")
println(" ") 
println("The results from solving the DMP economy are :")
println(" ")
println("The non-stochastic steady state unemployment rate is : $(uss)")
println("The non-stochastic steady state level of job vacancies is : $(vss)")
println("The non-stochastic steady state level of the v-u ratio is : $(thss)")
println("The non-stochastic steady state level of job hires is : $(hss)")
println("The non-stochastic steady state level of the job finding rate is : $(jfndss)",)
println("The non-stochastic steady state level of the job filling rate is : $(jfllss)",)
println(" ")
println("The stochastic steady state unemployment rate is : $(usss)")
println("The stochastic steady state level of job vacancies is : $(vsss)")
println("The stochastic steady state level of the v-u ratio is : $(vsss/usss)")
println("The stochastic steady state level of job hires is : $(hsss)")
println("The stochastic steady state level of the job finding rate is : $(jfndsss)")
println("The stochastic steady state level of the job filling rate is : $(jfllsss)")
println(" ")
println("----------------------------------------------------------------------------")


# Define Font Dictionaries:
font1 = Dict("family"=>"serif",
# "name" => "times",
"name" => "Times New Roman",
#"color"=>"darkred",
#"weight"=>"normal",
"size"=>10)
#xlabel("Time",fontdict=font1)

font2 = Dict("family"=>"serif",
# "name" => "times",
"name" => "Times New Roman",
#"color"=>"darkred",
#"weight"=>"normal",
"size"=>12)
#xlabel("Time",fontdict=font1)

font3 = Dict("family"=>"serif",
# "name" => "times",
"name" => "Times New Roman",
#"color"=>"darkred",
#"weight"=>"normal",
"size"=>14)
#xlabel("Time",fontdict=font1)

plotopt = 2
if plotopt == 1
    ## Plotting Impulse Response to Productivity Shock:
    T    = 48  # Number of periods to plot impulse responses.
    shck = 8   # Index of level of labour productivity with which to plot impulse response.
    sA0  = ceil(nstateA/2)  # Initial State for labour productivity
    sD0  = ceil(nstateD/2)  # Initial State for job separation rate
    statevecA = dropdims(convert(Array{Int64,2},ones(T,1)*sA0),dims=2)
    statevecA[2:10] = dropdims(convert(Array{Int64,2},ones(9,1)*shck),dims=2)
    ChainA  = eAt[statevecA]
    statevecD = dropdims(convert(Array{Int64,2},ones(T,1)*sD0),dims=2)
    ChainD  = eDt[statevecD]

    uvec    = zeros(T,1)
    thvec   = zeros(T,1)
    vvec    = zeros(T,1)
    wvec    = zeros(T,1)
    fndvec  = zeros(T,1)
    fllvec  = zeros(T,1)
    hvec    = zeros(T,1)
    lpvec   = ChainA
    dvec    = ChainD
    uvec[1] = usss # the nonstochastic steady state UR rate for Manovskii-Hagedorn model is 8.01#
                   # The stochastic steady state unemployment rate is 5.32#
    for t in 1 : T
        ut      = uvec[t]
        at      = ChainA[t]
        stateAt = statevecA[t]
        dt      = ChainD[t]
        stateDt = statevecD[t]

        tht     = THETA[stateAt,stateDt]
        qt      = 1/((1+tht^alpha)^(1/alpha))
        ht      = ut*tht*qt
        ut1     = ut + dt*(1-ut) - tht*qt*ut
        ut1     = max(ut1,0)
        ut1     = min(ut1,1)
        thvec[t]    = tht
        vvec[t]     = tht*ut 
        fndvec[t]   = tht*qt
        fllvec[t]   = qt
        hvec[t]     = ht
        wvec[t]     = beta*at + (1-beta)*b + beta*z*tht
        if t < T
            uvec[t+1]  = ut1
        end
    end
    figure("Labour Productivity IRF")
    subplot(2,3,1)
    plot(collect(1:T),(log.(lpvec) .- log(exp(mA)))*100,linewidth=1,color="b",label="a")
    plot(collect(1:T),(log.(dvec) .- log(exp(mD)*delta))*100,linestyle="-.",linewidth=1,color="r",label=L"\delta")
    title("Productivity and Job Destrubtion",fontdict=font2)
    legend(loc=1)
    xlabel("Quarters",fontdict=font1)
    ylabel("% Deviation from Stochastic Steady State",fontdict=font1)
    subplot(2,3,2)
    plot(collect(1:T),log.(uvec./usss),color="b")
    title("Unemployment Rate",fontdict=font2)
    xlabel("Quarters",fontdict=font1)
    #ylabel("# Deviation from Stochastic Steady State",fontdict=font1)
    # set(gca,"xlim",[1,T],"fontname","times","fontsize",12);
    subplot(2,3,3)
    plot(collect(1:T),log.(fndvec./jfndsss),color="b",label=L"\mu")  # 0.1442 is the HM job finding rate in the stochastic steady state
    plot(collect(1:T),log.(fllvec./jfllsss),color="r",label="q")  # 0.2254 is the HM job filling rate in the stochastic steady state
    legend(loc=1)
    title("Job Finding and Filling Probability",fontdict=font2)
    xlabel("Quarters",fontdict=font1);
    #ylabel("# Deviation from Stochastic Steady State",fontdict=font1)
    #set(gca,"xlim",[1,T],"fontname","times","fontsize",12);
    subplot(2,3,4);
    plot(collect(1:T),log.(wvec./wsss),color="b")  # 0.2254 is the HM job filling rate in the stochastic steady state
    title("Wages",fontdict=font2)
    #title("v-u Ratio","fontname","times","fontsize",12);
    xlabel("Quarters",fontdict=font1)
    ylabel("# Deviation from Stochastic Steady State",fontdict=font1)
    #set(gca,"xlim",[1,T],"fontname","times","fontsize",12);
    subplot(2,3,5)
    plot(collect(1:T),log.(vvec./(1 .- uvec + vvec)) .- log(vsss/(1-usss+vsss)),linewidth=1,color="b",label="v(t)")
    plot(collect(1:T),log.(hvec./(1 .- uvec)) .- log(hsss/(1-usss)),linewidth=1,color="r",label="h(t)")
    legend()
    title("Vacancy Rate vs Hiring Rate",fontdict=font2)
    xlabel("Quarters",fontdict=font1)
    #ylabel("# Deviation from Stochastic Steady State",fontdict=font1)
    subplot(2,3,6)
    plot(uvec[1:11],vvec[1:11]./(1 .- uvec[1:11] .+ vvec[1:11]),linewidth=1,color="b");
    #hold on
    plot(uvec[11:T],vvec[11:T]./(1 .- uvec[11:T] .+ vvec[11:T]),linestyle="-.",linewidth=1,color="r")
    #hold off
    title("The Beveridge Curve",fontdict=font2)
    xlabel("Unemployment Rate",fontdict=font1)
    ylabel("Vacancy Rate",fontdict=font1)
    
elseif plotopt == 2

    ## Plotting Impulse Response to Job Destruction Shock:
    T    = 48  # Number of periods to plot impulse responses.
    shck = 4   # Index of level of labour productivity with which to plot impulse response.
    sA0  = ceil(nstateA/2)  # Initial State for labour productivity
    sD0  = ceil(nstateD/2)  # Initial State for job separation rate
    statevecA = dropdims(convert(Array{Int64,2},ones(T,1)*sA0),dims=2)
    ChainA  = eAt[statevecA]
    statevecD = dropdims(convert(Array{Int64,2},ones(T,1)*sD0),dims=2)
    statevecD[2:10] = dropdims(convert(Array{Int64,2},ones(9,1)*shck),dims=2)
    ChainD  = eDt[statevecD]

    uvec    = zeros(T,1)
    thvec   = zeros(T,1)
    vvec    = zeros(T,1)
    wvec    = zeros(T,1)
    fndvec  = zeros(T,1)
    fllvec  = zeros(T,1)
    hvec    = zeros(T,1)
    lpvec   = ChainA
    dvec    = ChainD
    uvec[1] = usss # the nonstochastic steady state UR rate for Manovskii-Hagedorn model is 8.01#
                   # The stochastic steady state unemployment rate is 5.32#
    for t in 1 : T
        ut      = uvec[t]
        at      = ChainA[t]
        stateAt = statevecA[t]
        dt      = ChainD[t]
        stateDt = statevecD[t]

        tht     = THETA[stateAt,stateDt]
        qt      = 1/((1+tht^alpha)^(1/alpha))
        ht      = ut*tht*qt
        ut1     = ut + dt*(1-ut) - tht*qt*ut
        ut1     = max(ut1,0)
        ut1     = min(ut1,1)
        thvec[t]    = tht
        vvec[t]     = tht*ut 
        fndvec[t]   = tht*qt
        fllvec[t]   = qt
        hvec[t]     = ht
        wvec[t]     = beta*at + (1-beta)*b + beta*z*tht
        if t < T
            uvec[t+1]  = ut1
        end
    end
    figure("Job Separation IRF")
    subplot(2,3,1)
    plot(collect(1:T),(log.(lpvec) .- log(exp(mA)))*100,linewidth=1,color="b",label="a")
    plot(collect(1:T),(log.(dvec) .- log(exp(mD)*delta))*100,linestyle="-.",linewidth=1,color="r",label=L"\delta")
    title("Productivity and Job Destrubtion",fontdict=font2)
    legend(loc=1)
    xlabel("Quarters",fontdict=font1)
    ylabel("% Deviation from Stochastic Steady State",fontdict=font1)
    subplot(2,3,2)
    plot(collect(1:T),log.(uvec./usss),color="b")
    title("Unemployment Rate",fontdict=font2)
    xlabel("Quarters",fontdict=font1)
    #ylabel("# Deviation from Stochastic Steady State",fontdict=font1)
    # set(gca,"xlim",[1,T],"fontname","times","fontsize",12);
    subplot(2,3,3)
    plot(collect(1:T),log.(fndvec./jfndsss),color="b",label=L"\mu")  # 0.1442 is the HM job finding rate in the stochastic steady state
    plot(collect(1:T),log.(fllvec./jfllsss),color="r",label="q")  # 0.2254 is the HM job filling rate in the stochastic steady state
    legend(loc=1)
    title("Job Finding and Filling Probability",fontdict=font2)
    xlabel("Quarters",fontdict=font1);
    #ylabel("# Deviation from Stochastic Steady State",fontdict=font1)
    #set(gca,"xlim",[1,T],"fontname","times","fontsize",12);
    subplot(2,3,4);
    plot(collect(1:T),log.(wvec./wsss),color="b")  # 0.2254 is the HM job filling rate in the stochastic steady state
    title("Wages",fontdict=font2)
    #title("v-u Ratio","fontname","times","fontsize",12);
    xlabel("Quarters",fontdict=font1)
    ylabel("# Deviation from Stochastic Steady State",fontdict=font1)
    #set(gca,"xlim",[1,T],"fontname","times","fontsize",12);
    subplot(2,3,5)
    plot(collect(1:T),log.(vvec./(1 .- uvec + vvec)) .- log(vsss/(1-usss+vsss)),linewidth=1,color="b",label="v(t)")
    plot(collect(1:T),log.(hvec./(1 .- uvec)) .- log(hsss/(1-usss)),linewidth=1,color="r",label="h(t)")
    legend()
    title("Vacancy Rate vs Hiring Rate",fontdict=font2)
    xlabel("Quarters",fontdict=font1)
    #ylabel("# Deviation from Stochastic Steady State",fontdict=font1)
    subplot(2,3,6)
    plot(uvec[1:11],vvec[1:11]./(1 .- uvec[1:11] .+ vvec[1:11]),linewidth=1,color="b");
    #hold on
    plot(uvec[11:T],vvec[11:T]./(1 .- uvec[11:T] .+ vvec[11:T]),linestyle="-.",linewidth=1,color="r")
    #hold off
    title("The Beveridge Curve",fontdict=font2)
    xlabel("Unemployment Rate",fontdict=font1)
    ylabel("Vacancy Rate",fontdict=font1)
    
elseif plotopt == 3
    Brn  = 1000   # Number of Burn-in Periods to discard in simulation.
    nsim = 1000;   # Total number of periods to simulate (not inluding Burn-in periods!).
    SSSvec = [usss ; vsss ; hsss ; jfndsss ; jfllsss]

    uvec, vvec, thvec, hvec, fndvec, fllvec, wvec, ChainA, ChainD = DMP_DRW_Simulate(nsim, Brn, THETA, SSSvec, Param)

    T = nsim

    figure
    subplot(2,3,1)
    plot(collect(1:T),(log.(ChainA) .- log(exp(mA)))*100,linewidth=1,color="b",label="a")
    plot(collect(1:T),(log.(ChainD) .- log(exp(mD)*delta))*100,linestyle="-.",linewidth=1,color="r",label=L"\delta")
    title("Productivity and Job Destrubtion",fontdict=font2)
    legend(loc=1)
    xlabel("Quarters",fontdict=font1)
    ylabel("% Deviation from Stochastic Steady State",fontdict=font1)
    subplot(2,3,2)
    plot(collect(1:T),log.(uvec./usss))
    title("Unemployment Rate",fontdict=font2)
    xlabel("Quarters",fontdict=font1)
    #ylabel("# Deviation from Stochastic Steady State",fontdict=font1)
    # set(gca,"xlim",[1,T],"fontname","times","fontsize",12);
    subplot(2,3,3)
    plot(collect(1:T),log.(fndvec./jfndsss))  # 0.1442 is the HM job finding rate in the stochastic steady state
    title("Job Finding Probability",fontdict=font2)
    xlabel("Quarters",fontdict=font1);
    #ylabel("# Deviation from Stochastic Steady State",fontdict=font1)
    #set(gca,"xlim",[1,T],"fontname","times","fontsize",12);
    subplot(2,3,4);
    plot(collect(1:T),log.(fllvec./jfllsss))  # 0.2254 is the HM job filling rate in the stochastic steady state
    title("Job Filling Probability",fontdict=font2)
    #title("v-u Ratio","fontname","times","fontsize",12);
    xlabel("Quarters",fontdict=font1)
    ylabel("# Deviation from Stochastic Steady State",fontdict=font1)
    #set(gca,"xlim",[1,T],"fontname","times","fontsize",12);
    subplot(2,3,5)
    plot(collect(1:T),log.(vvec./(1 .- uvec + vvec)) .- log(vsss/(1-usss+vsss)),linewidth=1,color="b",label="v(t)")
    plot(collect(1:T),log.(hvec./(1 .- uvec)) .- log(hsss/(1-usss)),linewidth=1,color="r",label="h(t)")
    legend()
    title("Vacancy Rate vs Hiring Rate",fontdict=font2)
    xlabel("Quarters",fontdict=font1)
    #ylabel("# Deviation from Stochastic Steady State",fontdict=font1)
    subplot(2,3,6)
    scatter(uvec,vvec,facecolors="none",edgecolors="blue")

    # plot(uvec[1:11],vvec[1:11]./(1 .- uvec[1:11] .+ vvec[1:11]),linewidth=1,color="b");
    # #hold on
    # plot(uvec[11:T],vvec[11:T]./(1 .- uvec[11:T] .+ vvec[11:T]),linestyle="-.",linewidth=1,color="r")
    # #hold off
    title("The Beveridge Curve",fontdict=font2)
    xlabel("Unemployment Rate",fontdict=font1)
    ylabel("Vacancy Rate",fontdict=font1)

end

println("All Done!")