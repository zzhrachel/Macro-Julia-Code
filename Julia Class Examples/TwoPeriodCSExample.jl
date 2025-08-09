using Distributions
using Random
include("fcsolve.jl")

#---------------------------------------------------------------------------------------------------------
# PURPOSE: Solve for the optimal level of savings in a 2-period consumption-savings model under certainty.
#---------------------------------------------------------------------------------------------------------

struct Parameter_Values
    beta :: Real
    r    :: Real
    y0   :: Real
    y1   :: Real
    sigma :: Real
end

function TwoPeriodCSSolve(x0 :: Real, Param :: Parameter_Values)
    #------------------------------------------------------------------------------
    # PURPOSE: Find the value of savings that makes the optimality condition hold:
    #------------------------------------------------------------------------------
    # Retrieve parameters:
    beta = Param.beta
    r    = Param.r
    y0   = Param.y0
    y1   = Param.y1
    sigma = Param.sigma
    # Get current iterate for savings:
    s   = x0
    # Define residual of Euler equation at current savings iterate:
    res = (y1 + (1+r)*s)^sigma - beta*(1+r)*(y0 - s)^sigma
    return res
end


## Define parameters:
beta  = 0.9  # Subjective discount factor.
r     = 0.1  # Choose r close to 1/beta-1.  If r = 1/beta-1 optimal consumption
             # is equal across periods and we can solve for consumption analytically.
y0    = 1.1  # Period 0 income.
y1    = 1.25 # Period 1 income.
sigma = 4    # Curvature parameter in CRRA utility function.


# Store parameter values in a structure:
Param = Parameter_Values(beta,r,y0,y1,sigma)

# Define parameters of zero-finding routine: (These are the default values)
itmax = 1000    # max no. of iterations
crit  = 1e-6	# sum of abs. values small enough to be a solution
delta = 1e-8	# differencing interval for numerical gradient
alpha = 1e-3	# tolerance on rate of descent
dispo = 1       # partial printlnlay (printlno = 0 for no printlnlay)


x0   = 1;    # Initial guess for optimal savings.
#options = [itmax ; crit ; delta ; alpha ; dispo]; # Entering options = [] means fcsolve uses its 
                                                  # default setting. Make sure options are entered
                                                  # in the correct order.
options = [];                                                  
x    = fcsolve(TwoPeriodCSSolve,x0,[],Param);

println(" ")
println("-----------------------------------------------------")
println("                     RESULTS") ;
println("-----------------------------------------------------")
println(" ")
println("Optimal saving level is: $(x)")
println("Optimal consumption in the first period is: $(y0-x)")
println("Optimal consumption in the second period is: $(y1+(1+r)*x)")
println(" ")
println("-----------------------------------------------------")

