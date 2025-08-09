using Distributions
using Random
include("fcsolve.jl")

#------------------------------------------------------------------------------------------
# PURPOSE : Find the maximum of z = -0.5*(x-a)^2 -0.5*(y-b)^2 using first-order conditions:
#------------------------------------------------------------------------------------------

struct Parameter_Values
    a :: Real
    b :: Real
end

function MaxBowlFONCSolve(x0 :: Vector, Param :: Parameter_Values)
    #---------------------------------------------------------------------------
    # PURPOSE: Find the values of x and y that maximize z = -0.5*x^2 - 0.5*y^2
    #---------------------------------------------------------------------------

    # Retrieve parameter values:
    a   = Param.a
    b   = Param.b

    # Retrieve current interation values of (x,y)
    x   = x0[1]
    y   = x0[2]

    res1 = x - a
    res2 = y - b

    res = [res1 ; res2]
    return res
end


# Set parameter values:
a   = 5
b   = 7

Param = Parameter_Values(a,b)

## Set an initial guess at optimal values:
x0 = [1 ; 3] # Initial guess for x and y at maximum.

# Define parameters of zero-finding routine: (These are the default values)
itmax =1000     # max no. of iterations
crit  =1e-6		# sum of abs. values small enough to be a solution
delta =1e-8 	# differencing interval for numerical gradient
alpha =1e-3 	# tolerance on rate of descent
dispo =1        # partial printlnlay (printlno = 0 for no printlnlay)
options = [itmax ; crit ; delta ; alpha ; dispo]  # Entering options = [] means fcsolve uses its 
                                                  # default setting. Make sure options are entered
                                                  # in the correct order.
                                                  
xx   = fcsolve(MaxBowlFONCSolve,x0,[],Param);
x    = xx[1]
y    = xx[2]
z    = -0.5*(x-a)^2 -0.5*(y-b)^2;

## Print results to the command window (nicely).
println(" ")
println("---------------------------------------------------------------------------")
println("                             RESULTS                                       ")
println("---------------------------------------------------------------------------")
println(" ")
println("The results from finding the maximum of z = -0.5*(x-a)^2 -0.5*(y-b)^2 are :")
println(" ")
println("The value of x at the maximum is : $(x)")
println("The value of y at the maximum is : $(y)")
println("The value of z at the maximum is : $(z)")
println(" ")
println("----------------------------------------------------------------------------")


