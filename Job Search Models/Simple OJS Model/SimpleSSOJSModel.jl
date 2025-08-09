using Distributions
using PythonPlot
using Random
using HDF5, JLD

include("rouwenhorst.jl")
include("fcsolve.jl")

struct ParameterValues
    rho :: Float64
    chi :: Float64
    b :: Float64
    kappa :: Float64
    delta :: Float64
    xi :: Float64

    rhoz :: Float64
    sez :: Float64
    mz :: Float64
    nstate :: Int64
end

struct SolutionValues
    U :: Float64
    W :: Array{Float64,2}
    J :: Array{Float64,2}
    w :: Array{Float64,2}
    theta :: Float64
end

function SimulateProductivity(T :: Int64, PIZ :: Array, Zt :: Array)
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
    Brn     = 1000
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

function InitialZ_Distn(T :: Int64, nstate :: Int64, statevec :: Array)
    z_bins = collect(1:nstate)
    z_count = zeros(nstate,1)
    for t in 1 : T
        i = searchsortedlast(z_bins,statevec[t])
        z_count[i] += 1
    end
    z_hist  = z_count./T
    z_cdf   = cumsum(z_hist,dims=1)
    return z_hist, z_cdf
end

function SolveEqmTheta(x0 :: Real, nstate :: Int64, z0_pdf :: Array, Soln_Values :: SolutionValues,  Params :: ParameterValues)
    rho = Params.rho
    chi = Params.chi
    b   = Params.b
    kappa = Params.kappa
    delta = Params.delta
    xi  = Params.xi

    theta = exp(x0)

    mu = 1/((1+theta^(-xi))^(1/xi))
    eta = mu/theta 

    U = Soln_Values.U
    W = Soln_Values.W
    J = Soln_Values.J
    w = Soln_Values.w

    tol = 1e-6
    crt = 1
    while crt > tol
        SW  = zeros(nstate,1)
        SU  = zeros(nstate,1)
        Fw  = zeros(nstate,1)
        for i in 1 : nstate
            if i < nstate
                SW[i] = sum(z0_pdf[i+1:nstate].*W[i+1:nstate])
            end
            SU[i] = sum(z0_pdf[1:i])*U
            Fw[i] = sum(z0_pdf[1:i])
        end
        # w  = chi*Zt + (1-chi)*b + chi*kappa + (1-chi)*rho*mu*sum(z0_pdf.*W) - (1-chi)*mu*rho*SW - (1-chi)*mu*rho*SU
        w  = chi*Zt .+ (1-chi)*b .+ chi*kappa .+ (1-chi)*rho*(mu-eta)*sum(z0_pdf.*W) .- (1-chi)*mu*rho*SW .- (1-chi)*mu*rho*SU .+ (1-chi)*eta*rho*U
        TU = b + rho*mu*sum(z0_pdf.*W) + rho*(1-mu)*U
        TW = w + rho*mu*SW + rho*(1-delta)*(1 .- mu*(1 .- Fw)).*PIZ*W + rho*delta*(1 .- mu*(1 .- Fw))*U
        TJ = Zt - w + rho*(1-delta)*(1 .- mu*(1 .- Fw)).*PIZ*J

        resT = U - TU
        resW = W - TW
        resJ = J - TJ
        res  = [resT ; resW ; resJ]
        crt  = maximum(abs.(res)./(1+maximum(abs.(res))))
        
        U   = TU
        W   = TW
        J   = TJ
        #println(crt)
    end
    Soln_Values = SolutionValues(U,W,J,w,theta)
    res = kappa - eta*rho*sum(z0_pdf.*J)
    return res
end

function ReconstructSolution(xx :: Float64, nstate :: Int64, z0_pdf :: Array, Params :: ParameterValues)
    rho = Params.rho
    chi = Params.chi
    b   = Params.b
    kappa = Params.kappa
    delta = Params.delta
    xi  = Params.xi

    theta = exp(xx)

    mu = 1/((1+theta^(-xi))^(1/xi))
    eta = mu/theta

    U = 0
    W = zeros(nstate,1)
    J = zeros(nstate,1)
    w = zeros(nstate,1)

    tol = 1e-6
    crt = 1
    while crt > tol
        SW  = zeros(nstate,1)
        SU  = zeros(nstate,1)
        Fw  = zeros(nstate,1)
        for i in 1 : nstate
            if i < nstate
                SW[i] = sum(z0_pdf[i+1:nstate].*W[i+1:nstate])
            end
            SU[i] = sum(z0_pdf[1:i])*U
            Fw[i] = sum(z0_pdf[1:i])
        end
        w  = chi*Zt .+ (1-chi)*b .+ chi*kappa .+ (1-chi)*rho*(mu-eta)*sum(z0_pdf.*W) - (1-chi)*mu*rho*SW - (1-chi)*mu*rho*SU .+ (1-chi)*eta*rho*U
        TU = b .+ rho*mu*sum(z0_pdf.*W) .+ rho*(1-mu)*U
        TW = w + rho*mu*SW + rho*(1-delta)*(1 .- mu*(1 .- Fw)).*PIZ*W + rho*delta*(1 .- mu*(1 .- Fw))*U
        TJ = Zt - w + rho*(1-delta)*(1 .- mu*(1 .- Fw)).*PIZ*J

        resT = U - TU
        resW = W - TW
        resJ = J - TJ
        res  = [resT ; resW ; resJ]
        crt  = maximum(abs.(res)./(1+maximum(abs.(res))))
        
        U   = TU
        W   = TW
        J   = TJ
        # println(crt)
    end
    return U, W, J, w, theta   
end

function JobFindingProb(theta :: Float64, Params :: ParameterValues)
    xi = Params.xi
    mu = 1/((1+theta^(-xi))^(1/xi))
end

# Define Parameter Values:
rho  = 0.9959   # Discount Factor
chi  = 0.5      # Worker's bargaining power
b    = 0.4      # At b = 0.4 and nstate = 25, the minimum(Zt) > b (minimum(Zt)= )
kappa = 1.1     # Vacancy creation cost
delta = 0.0081  # Exogenous job separation probability
xi   = 1.2      # Curvature of CES matching function

rhoz   = 0.95   # Persistance of productivity.
sez    = 0.075  # Volatility of labour productivity.  High volatility sez = 0.01, low volatiliy sez = 0.001
mZ     = 0.0    # mean
nstate = 15     # Number of states for labour productivity

Params = ParameterValues(rho,chi,b,kappa,delta,xi,rhoz,sez,mZ,nstate)

Zgrid, PIZ = rouwenhorst(rhoz,sez,nstate)
Zt  = exp.(Zgrid .+ mZ)

T   = 100000
ChainZ, statevec, Brn = SimulateProductivity(T, PIZ, Zt)
# Drop the burn in periods t = 1 through Brn:
ChainZ = ChainZ[Brn+1:Brn+T]
statevec = convert(Array{Int64,1},statevec[Brn+1:Brn+T])
z0_pdf, z0_cdf = InitialZ_Distn(T, nstate, statevec)

# Initiate the composite type "Soln_Values"

load_sol = 0
if load_sol == 1
    Soln_Values = load("/Users/jacobwong/Dropbox/Macro IV Computer Codes/Julia Codes/Post-Version 1.0/Job Search Models/Simple OJS Model/SimpleSSOJSModelSoln.jld","Soln_Values")
    U   = Soln_Values.U
    W   = Soln_Values.W
    J   = Soln_Values.J
    w   = Soln_Values.w
    theta = Soln_Values.theta
else
    U   = 0
    W   = zeros(nstate,1)
    J   = zeros(nstate,1)
    w   = zeros(nstate,1)
    theta = 0.6
end
Soln_Values = SolutionValues(U,W,J,w,theta)

Solve_Model = 1
if Solve_Model == 1
    x0  = log(theta)
    @time xx = fcsolve(SolveEqmTheta,x0,[],nstate,z0_pdf,Soln_Values,Params)
    U, W, J, w, theta = ReconstructSolution(xx, nstate, z0_pdf, Params)
    Soln_Values = SolutionValues(U,W,J,w,theta)
end
theta = Soln_Values.theta
mu = JobFindingProb(theta, Params)
eta = mu/theta

function SimulateEconomy(nstate :: Int64, theta :: Float64, z0_pdf :: Array{Float64,2}, PIZ :: Array{Float64,2}, Params :: ParameterValues)
    delta = Params.delta

    mu = JobFindingProb(theta, Params)

    Fw  = zeros(nstate,1)
    for i in 1 : nstate
        if i < nstate
            Fw[i] = sum(z0_pdf[i+1:nstate])  # Probability of worker leaving by OJS conditional on having an offer.
        end
    end
    # Initiate guess at unemployment rate and the distribution of workers across firm types, psi.
    tld_psi = 1/nstate*ones(1,nstate)
    tld_delta = (1 .- mu*tld_psi*Fw)*delta
    uss  = tld_delta./(mu .+ tld_delta)
    u    = uss[1]
    psi_t = 1/nstate*ones(1,nstate)*(1-u)
    tol  = 1e-8
    crt  = 1
    bean = 0
    while crt > tol
        u_sv = u
        # Construct a vector of inflows to each state from OJS transitions:
        OJS_TRNS = zeros(nstate,nstate)
        for i in 2 : nstate
            for j in 1 : i-1
                OJS_TRNS[i,j] += mu*psi_t[j]*z0_pdf[i]
            end
        end
        OJS_Inflows = sum(OJS_TRNS*(1-u),dims=2)
        # Construct a vector of inflows to each employment state from unemployment
        U2Z_Inflows = mu*z0_pdf*u
        # State-to-State Transitions by firm productivity
        S2S_TRNS = (1-delta)*((1 .- mu*Fw').*psi_t*PIZ)
        # Construct 
        psi_p = S2S_TRNS' + U2Z_Inflows + OJS_Inflows 
        psi_t = psi_p'
        u     = 1 - sum(psi_t)
        crt   = abs(u - u_sv)
        # println(crt)
    end
    # Re-construct a vector of inflows to each state from OJS transitions:
    OJS_TRNS = zeros(nstate,nstate)
    for i in 2 : nstate
        for j in 1 : i-1
            OJS_TRNS[i,j] += mu*psi_t[j]*z0_pdf[i]
        end
    end
    OJS_Inflows = sum(OJS_TRNS*(1-u),dims=2)
    Job2JobRate = sum(OJS_Inflows)
    return psi_t, Job2JobRate
end

psi, Job2JobRate = SimulateEconomy(nstate, theta, z0_pdf, PIZ, Params)
uss = 1 - sum(psi)
tld_psi = psi/(1-uss)

save_sol = 0
if save_sol == 1
    save("/Users/jacobwong/Dropbox/Macro IV Computer Codes/Julia Codes/Post-Version 1.0/Job Search Models/Simple OJS Model/SimpleSSOJSModelSoln.jld","Soln_Values",Soln_Values,"Params",Params)
end

mean_wage = sum(tld_psi*w)
mean_min_ratio = log(mean_wage/minimum(w))

println("----------------------------------------------------------------------")
println("In the steady state, the job finding probability is: $(mu)            ")
println("In the steady state, the job filling probability is: $(eta)           ")
println("In the steady state, the job-to-job transition rate is: $(Job2JobRate)")
println("In the steady state, the mean-min wage ratio is: $(mean_min_ratio)    ")
println("----------------------------------------------------------------------")


plot_figs = 1
if plot_figs == 1
    # Define Font Dictionaries:
    font1 = Dict("family"=>"serif",
    # "name" => "times",
    "name" => "Times New Roman",
    #"color"=>"darkred",
    #"weight"=>"normal",
    "size"=>12)
    #xlabel("Time",fontdict=font1)

    font2 = Dict("family"=>"serif",
    # "name" => "times",
    "name" => "Times New Roman",
    #"color"=>"darkred",
    #"weight"=>"normal",
    "size"=>14)
    #xlabel("Time",fontdict=font1)

    font3 = Dict("family"=>"serif",
    # "name" => "times",
    "name" => "Times New Roman",
    #"color"=>"darkred",
    #"weight"=>"normal",
    "size"=>16)
    #xlabel("Time",fontdict=font1)


    fig = figure("Value_Function_Plots")
    subplot(2,2,1)
    plot(Zt,U*ones(nstate,1),linewidth=2,color="r")
    xlabel(L"$Z$",fontdict=font1)
    ylabel(L"$U$",fontdict=font1)
    title("The Value of Unemployment",fontdict=font2)
    axis([Zt[1],Zt[nstate],U-0.1*U,U+0.1*U])
    subplot(2,2,2)
    plot(Zt,W,linewidth=2,color="b")
    xlabel(L"$Z$",fontdict=font1)
    ylabel(L"$W$",fontdict=font1)
    title("The Value of Employment",fontdict=font2)
    axis([Zt[1],Zt[nstate],minimum(W),maximum(W)])    
    subplot(2,2,3)
    plot(Zt,J,linewidth=2,color="b")
    xlabel(L"$Z$",fontdict=font1)
    ylabel(L"$J$",fontdict=font1)
    title("The Value of the Firm",fontdict=font2)
    axis([Zt[1],Zt[nstate],minimum(J),maximum(J)])
    subplot(2,2,4)
    plot(Zt,w,linewidth=2,color="b")
    xlabel(L"$Z$",fontdict=font1)
    ylabel(L"$w$",fontdict=font1)
    title("Wages",fontdict=font2)
    axis([Zt[1],Zt[nstate],minimum(w),maximum(w)])

    fig = figure("Distribution_of_Workers")
    plot(Zt,psi',linewidth=2,color="b",label="Distribution of Workers")
    plot(Zt,z0_pdf,linewidth=2,color="r",label="f(z) of Start-ups")
    legend(loc=1)
    xlabel("z")
    axis([Zt[1],Zt[nstate],0,max(maximum(z0_pdf),maximum(psi'))])

end

println("All Done")

