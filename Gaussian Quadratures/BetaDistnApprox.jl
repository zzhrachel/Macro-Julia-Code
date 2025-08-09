using Distributions
#include("lgwt.jl")
include("gauss_leg.jl")
function trsf(x :: Array, a :: Real, b :: Real)
    # maps [a,b] into [0,1]
    y = (x .- a)/(b-a)
    return y
end

function trsf(x :: Real, a :: Real, b :: Real)
    # maps [a,b] into [0,1]
    y = (x .- a)/(b-a)
    return y
end

#------------------------------------
# Integration for Beta Distribution:
#------------------------------------

# Define the shape parameters for the beta distribution:
a = 1
b = 2
dist_ei = Beta(a,b)                            # Define the distribution that we will use.
eint    = 25                                   # # of nodes in the quadrature
#z, q    = lgwt(eint,-1,1)                     # nodes and weights for Gauss-Legendre quadrature (Mathworks Code (more accurate))
z, q    = gauss_leg(eint)                      # nodes and weights using Fabrice's codes (rewritten from Matlab to Julia) 
ei      = trsf(z,-1,1)                         # transforms nodes from [-1,1] to [0,1]
# Test the beta distribution approximation using the gauss-legendre quadrature:
# Approximate the mean of the beta distribution:
mean_approx = sum(0.5*q.*ei.*pdf.(dist_ei,ei))
beta_mean = a/(a+b)

# Approximate the mean of the variance distribution:
variance_approx = sum(0.5*q.*((ei .- mean_approx).^2).*pdf.(dist_ei,ei))
beta_variance = (a*b)/((1+a+b)*(a+b)^2)

println("The true mean and its approximation is $([beta_mean mean_approx])")
println("The true variance and its approximation are $([beta_variance variance_approx])")

