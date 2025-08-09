using Distributions
include("fcsolve.jl")


#---------------------------------------------------------------------------------------------------------
# PURPOSE: Solve for the steady state level of v-u ratio in the Diamond-Mortensen-Pissarides model.
# The matching function is chosen to be CES (channel matching function).
#---------------------------------------------------------------------------------------------------------

struct Parameter_Values
    rho :: Float64
    beta :: Float64
    delta :: Float64
    alpha :: Real
    kappa :: Real
    mA :: Real
    mD :: Real
    b :: Float64
end

function DMPSteadyStateSolve(x0 :: Real, Param :: Parameter_Values)
    #-----------------------------------------------------------------------------------------------
    # PURPOSE: This function solves for the steady state v-u ratio.
    #-----------------------------------------------------------------------------------------------

    # Retrieve parameter values:
    rho     = Param.rho
    beta    = Param.beta
    delta   = Param.delta
    alpha   = Param.alpha
    kappa   = Param.kappa
    mA      = Param.mA
    mD      = Param.mD
    b       = Param.b

    # Retrieve current iteration of market tightness:
    thss    = x0
    # Define job filling probability:
    qss     = 1/((1+thss^alpha)^(1/alpha))
    # Calculate deviation of equilibrium equation from holding at current iteration of market tightness:
    res     = kappa - rho*qss*((1-beta)*(exp(mA)-b) - beta*kappa*thss)/(1-rho*(1-exp(mD)*delta))
    return res
end


## Define parameter values:
cal = 1
if cal == 1
    # For the baseline Hagedorn-Manovskii set-up use alpha = 0.407 and z = 0.584.  Using alpha= 1.2 and 
    # z = 1.025 gets the job finding probability to be 41# and the job filling probability to be 71# in
    # the stochastic steady state.
    rho   = 0.9992; # Discount factor (0.9992 in HM"s paper)
    beta  = 0.052;  # Worker"s bargaining weight (0.052 for HM, 0.5 for Shimer)
    delta = 0.0081; # Job separation rate
    alpha = 1.2;    # Curvature parameter in matching function
    kappa = 1.025;  # Cost of opening a vacancy
    mA    = 0;      # Steady state labour productivity
    mD    = 0;      # Steady state separation rate
    b     = 0.955*exp(mA); # Unemployment consumption (0.955*exp(mA) for HM, 0.4*exp(mA) for Shimer)
else
    # This Shimer calibration gets the job finding probability to be 40# and the job filling probability
    # to be 71# in the stochastic steady state. (rho = 0.9992, beta = 0.895, delta = 0.0081, alpha = 1.2, z = 0.12, b = 0.4)
    rho   = 0.9992; # Discount factor
    beta  = 0.895;  # Worker"s bargaining weight (0.895)
    delta = 0.0081; # Job separation rate (results in a quarterly job separation rate of approximatly 0.0929)
    alpha = 1.2;    # Curvature parameter in matching function
    z     = 0.12;   # Cost of opening a vacancy
    mA    = 0;
    mD    = 0;
    b     = 0.40*exp(mA); # Unemployment consumption
end

# Store parameter values in a structure:
Param = Parameter_Values(rho,beta,delta,alpha,kappa,mA,mD,b)

##
#----------------------------------------------
# Solving for Steady State (see class notes) :
#----------------------------------------------

x0      = 1    # Initial guess for steady state v-u ratio.
options = []   # Use default setting for fscolve.
thss    = fcsolve(DMPSteadyStateSolve,x0,[],Param)   # Solve for steady state v-u ratio.

# Recover all other steady state values for endogenous variables:
qss     = 1/((1+thss^-alpha)^(1/alpha))
uss     = exp(mD)*delta/(exp(mD)*delta + thss*qss)
wss     = beta*exp(mA) + (1-beta)*b + beta*kappa*thss
Jss     = (exp(mA)-wss)/(1-rho*(1-exp(mD)*delta))
Uss     = b/(1-rho) + beta/(1-beta)*kappa*thss/(1-rho)
Wss     = beta/(1-beta)*Jss + Uss
Sss     = Jss + Wss - Uss
hss     = thss*qss*uss
vss     = thss*uss
jfndss  = 1/((1+thss^(-alpha))^(1/alpha))
jfllss  = 1/((1+thss^(alpha))^(1/alpha))

println(" ") ;
println("-----------------------------------------------------") ;
println("                     RESULTS") ;
println("-----------------------------------------------------") ;
println(" ")
println("The steady state v-u ratio is: $(thss)")
println("The steady state unemployment rate is: $(uss)")
println("The steady state vacancy rate is: $(vss/(1-uss+vss))")
println("The steady state wage is: $(wss)")
println("The steady state job finding probability is: $(jfndss)")
println("The steady state job filling probability is: $(jfllss)")
println(" ") ;
println("-----------------------------------------------------") ;
