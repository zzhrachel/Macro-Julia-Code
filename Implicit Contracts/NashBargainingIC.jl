# Solve the Nash Bargaining analogue to the Implicit contract model.
using Distributions
using PyPlot

include("fcsolve.jl")
include("rouwenhorst.jl")

type Model_Parameters
    y :: Real
    B :: Float64
    chi :: Real
    xi :: Float64
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

function NashBargainingSoln(x0 :: Array{Float64,1}, Params :: Model_Parameters)
    xi  = Params.xi
    B   = Params.B
    chi = Params.chi
    y   = Params.y

    Ut  = Params.Ut
    PIU = Params.PIU


    VmU = exp.(x0[1:nstate])
    J   = exp.(x0[nstate+1:2*nstate])
    V   = VmU+Ut

    # #w = ((1-xi)/xi*(V - Ut)./J).^(-1/xi)
    # w  = (xi/(1-xi)*J./(V-Ut)).^(1/xi)
    # TV = util(w,chi) + B*PIU*V
    # TJ = y - w + B*PIU*J
    # resV = V - TV
    # resJ = J - TJ

    resV = (1-chi)*V.*((1-xi)*(V-Ut)).^((1-chi)/chi) - (xi*J).^((1-chi)/chi) + ((1-xi)*(V-Ut)).^((1-chi)/chi) - 
                ((1-xi)*(V-Ut)).^((1-chi)/chi)*(1-chi).*(B*PIU*V)
    resJ = J.*((1-xi)*(V-Ut)).^(1/chi) - y*((1-xi)*(V-Ut)).^(1/chi) + (xi*J).^(1/chi) + ((1-xi)*(V-Ut)).^(1/chi).*(B*PIU*J)
    res = [resV./V ; resJ./J]
    return res
end

# Define parameter values:
y = 10      # Period output
B = 0.99    # Discount factor (beta in the notes)
chi = 2     # Curvature in CRRA utility function
xi = 0.5    # Worker's Nash Bargaining weight

# Set-up stochastic process as a 3-state Markov Chain:
rho = 0.99
sigma = 0.1
nstate = 2
UGrid, PIU = rouwenhorst(rho,sigma,nstate)
Ut  = exp.(UGrid)

# V0  = util([0.75 ; 0.85 ; 0.9],chi)/(1-B)
# J0  = (y - [0.75 ; 0.85 ; 0.9])/(1-B)
# x0  = [V0;J0]
#x0  = [27.1247 ; 29.9659 ; 33.0757 ; -0.288135 ; -0.48135 ; -0.688135]
x0  = dropdims(3*ones(2*nstate,1),dims=2)
Params = Model_Parameters(y,B,chi,xi,rho,sigma,nstate,UGrid,PIU,Ut)
options = [100 ; 1e-8 ; 1e-8 ; 1e-3; 1]
xx  = fcsolve(NashBargainingSoln,x0,[],Params)

# # Provide a simulation:
# WageBounds = [minw1 maxw1 ; minw2 y]
# T   = 100  # Number of periods in the simulation
# V0  = mean(Ut[1:2])  # Initial value of Promised Utility
# ChainU, statevec = SimulateUStates(T, PIU, V0, Ut)
# statevec = convert(Array{Int64,1},squeeze(statevec,2))
# Wages = zeros(T,1)
# w0  = InitialWage(V0, Ut, PIU, Params)
# Wages[1] = w0
# for i in 2 : T
#     if statevec[i] .> statevec[i-1]
#         if Wages[i-1].< WageBounds[2,1]
#             Wages[i] = WageBounds[2,1]
#         else
#             Wages[i] = Wages[i-1]
#         end
#     else
#         Wages[i] = Wages[i-1]
#     end
# end

