using PythonPlot
using HDF5, JLD
using Distributions
using Random

# This code solves for a benchmark consumption-savings problem when the individual is subject to a borrowing rate that is weakly greater 
# than that of the borrowing rate.  I will assume, in this code, that the subjective discount factor is less than or equal to the inverse of
# the gross lending rate (so that beta*Rl <= 1).

#type CSBaselineVFISoln
struct CSBaselineVFISoln
	beta :: Real
	Rb :: Real
	Rl :: Real
  	sigma :: Real
	pref :: Int64
	rhoY :: Real
	seY :: Real
	mY :: Real
	nstateY :: Int64
	Ygrid :: Array
	PI :: Array
	Yt :: Array
	nx :: Int64
	nxp :: Int64
	xgrid :: Array
	xmin :: Real
	xmax :: Real
	XPMat :: Array
	CMat :: Array
	V :: Array
	DRx :: Array
	DRc :: Array
	SDRx :: Array
	SDRc :: Array
end

# Define Parameters:
Rl    = 1.02
Rb    = 1.15
beta  = 0.96	# Make sure that beta <= 1/Rl!

pref  = 1;
if pref == 0
    util(x)   = log(x)
else
    util(x,sigma)  = (x.^(1-sigma) .- 1)/(1-sigma) .- 1/(1-sigma)       # Utility function
    sigma   = 2.0;
end

#----------------------------------------------------------------------------------------------------
# Constructing the Markov Chain (Tauchen-Hussey, 1991 following Floden's code (Economics Letters)) :
#----------------------------------------------------------------------------------------------------

rhoY    = 0.8    # Persistance of productivity.
seY     = 0.1    # Volatility of labour productivity
mY      = 0.0    # mean
nstateY = 5      # Number of states for labour productivity

include("rouwenhorst.jl")
Ygrid, PI = rouwenhorst(rhoY,seY,nstateY)
Yt  = exp.(Ygrid .+ mY)

# Define Grid for savings.
xmin = -0.99*Yt[1]/(Rb-1)           # Minimum value for x.  Must be very careful with this setting of xmin or results will be nonsensical
xmax = Yt[nstateY]/(Rl-1)           # Maximum value for x.  
nx   = 151                          # Number of gridpoints for x grid.
nxp  = 1501                         # Number of gridpoints for x' grid.
# xgrid = collect(linspace(xmin,xmax,nx))    # Construct linear grid for x and store in a column vector.
xgrid = collect(range(xmin,stop=xmax,length=nx))    # Construct linear grid for x and store in a column vector.

#---------------------------------------------------------------------------------------------------------------------------
# Define matrices for consumption and continuation savings.
cmin = 1e-6

function ConstructCandX(cmin :: Real, xgrid :: Array)
	# For each level of savings I need to construct a matrix for consumption and continuation savings where each row is for a 
	# given level of saving and each column is for a given level of consumption.
	chk  = findall(xgrid .<= 0)  # Finds values of xgrid that are less than zero.

	xneg = xgrid .< 0
	xnonneg = xgrid .>= 0
	# cmax  = repmat(Rb*xneg.*xgrid+Rl*xnonneg.*xgrid,nstateY,1) + kron(Yt,ones(nx,1)) - xmin
	cmax  = repeat(Rb*xneg.*xgrid+Rl*xnonneg.*xgrid,nstateY,1) + kron(Yt,ones(nx,1)) .- xmin

	CMat     = zeros(nx*nstateY,nxp)
	for i in 1 : nx*nstateY
	    #CMat[i,:] = linspace(cmin,cmax[i],nxp)
	    CMat[i,:] = range(cmin,stop=cmax[i],length=nxp)
	end
	#XPMat = repmat(Rb*xneg.*xgrid+Rl*xnonneg.*xgrid,nstateY,nxp) + kron(Yt,ones(nx,nxp)) - CMat;
	XPMat = repeat(Rb*xneg.*xgrid+Rl*xnonneg.*xgrid,nstateY,nxp) + kron(Yt,ones(nx,nxp)) - CMat;
	# Get rid of values in XPMat where x' > xmax by imposing the upper bound on x'.
	ind = findall(XPMat .> xmax)
	if ~isempty(ind) 
		XPMat[ind] .= xmax
	end
	ind = findall(XPMat .< xmin)
	if ~isempty(ind) 
		XPMat[ind] .= xmin
	end
	return CMat, XPMat
end

include("LinearInterp1D.jl")

function ConstructVP(nx :: Int64, nxp :: Int64, nstateY :: Int64, xgrid :: Array, XPMat :: Array, V :: Array)
    VP  = zeros(nx*nstateY,nxp);
    for i in 1 : nstateY
        vptmp = zeros(nx,nxp);
        for j = 1 : nx
            VPvec = LinearInterp1D(xgrid,V[(i-1)*nx+1:i*nx],collect(XPMat[(i-1)*nx+j,:]'))
            vptmp[j,:] = VPvec'
        end
        VP[(i-1)*nx+1:i*nx,:] = vptmp;
    end
    return VP
end

function ConstructEV(nx :: Int64, nxp :: Int64, nstateY :: Int64, PI :: Array, VP :: Array)
    EV  = zeros(nx*nstateY,nxp)
    for i in 1 : nstateY
        evp  = zeros(nx,nxp)
        for k in 1 : nstateY
            evp  = evp + PI[i,k]*VP[(k-1)*nx+1:k*nx,:];
        end
        EV[(i-1)*nx+1:i*nx,:] = evp
    end
    return EV      
end

function ConstructTV(pref :: Int64, c :: Array, EV :: Array, sigma :: Real)
    if pref == 0
        TV, ind = findmax(util(c) + beta*EV,dims=2)
    else
        TV,ind  = findmax(util(c,sigma) + beta*EV,dims=2)
    end
    return TV, ind
end

CMat, XPMat = ConstructCandX(cmin, xgrid)
loadsoln = 0
if loadsoln == 0
	global V	= ones(nx*nstateY,1)
	global VP  = repeat(V,1,nxp)
	global DRx = zeros(nx*nstateY,1)
	global DRc = zeros(nx*nstateY,1)
else
	CSBaselineSoln = load("/Users/jacobwong/Dropbox/Jake's Julia Codes/Post-Version 1.0/Consumption Savings Codes/CSBaselineVFISol.jld","CSBaselineSoln")
	V0	= CSBaselineSoln.V
	xgrid0 = CSBaselineSoln.xgrid
	nx0	= CSBaselineSoln.nx
	nstateY0 = CSBaselineSoln.nstateY
	global V	= zeros(nx*nstateY,1)
	for i in 1 : nstateY
		V[(i-1)*nx+1:i*nx,:] = LinearInterp1D(xgrid0,V0[(i-1)*nx0+1:i*nx0,:],xgrid)
	end
	global VP  = repeat(V,1,nxp)
	global DRx = zeros(nx*nstateY,1)
	global DRc = zeros(nx*nstateY,1)
end
crt	= 1
tol	= 1e-6
@time while crt >= tol
	global VP  = ConstructVP(nx, nxp, nstateY, xgrid, XPMat, V)
	EVP = ConstructEV(nx, nxp, nstateY, PI, VP)
	TV, ind = ConstructTV(pref, CMat, EVP, sigma)
	global DRx	= XPMat[ind]
	global DRc = CMat[ind]
	global 	crt	= maximum(abs.(V - TV)./(1+maximum(abs.(V))))
    println(crt)
    global V   = copy(TV)
end

#----------------------------
# Smooth the Decision Rules:
#----------------------------
# Use the following transformation when smoothing the decision rules using chebychev polynomials.  
# Write functions as multiple dispatch to allow for number of vector inputs
function trsfc(x :: Real, a :: Real, b :: Real)
	# maps [a,b] into [-1,1]
	zd	= 2*(x-a)/(b-a)-1
	return zd
end

function trsfc(x :: Array, a :: Real, b :: Real)
	# maps [a,b] into [-1,1]
	zd	= 2*(x .- a)/(b-a) .- 1
	return zd
end

function itrsfc(x :: Real, a :: Real, b :: Real)
	# maps [-1,1] into [a,b]
	xd	= 0.5*(b-a)*(1+x)+a
	return xd
end

function itrsfc(x :: Array, a :: Real, b :: Real)
	# maps [-1,1] into [a,b]
	xd	= 0.5*(b-a)*(1 .+ x) .+ a
	return xd
end

# Run a simple regression to "smooth out the decision rule" for use in simulations.
include("chebychev.jl")
nd      = 25;                      # # of chebychev polynomials used in chebychev polynomial approximations
zd      = trsfc(xgrid,xmin,xmax);
TD      = chebychev(zd,nd);

SDRx    = zeros(nx*nstateY,1);
SDRc    = zeros(nx*nstateY,1);
for i = 1 : nstateY
    B       = TD\DRx[(i-1)*nx+1:i*nx,1]
    Bc      = TD\DRc[(i-1)*nx+1:i*nx,1]
    SDRx[(i-1)*nx+1:i*nx,1] = TD*B      # Smoothed decision rule for savings
    SDRc[(i-1)*nx+1:i*nx,1] = TD*Bc		# Smoothed decision rule for savings
end

CSBaselineSoln = CSBaselineVFISoln(beta,Rb,Rl,sigma,pref,rhoY,seY,mY,nstateY,Ygrid,PI,Yt,nx,nxp,xgrid,xmin,xmax,XPMat,CMat,V,DRx,DRc,SDRx,SDRc)

savesoln = 1
if savesoln == 1
    desktop = 0
    if desktop == 0
    	save("/Users/jacobwong/Dropbox/Jake's Julia Codes/Post-Version 1.0/Consumption Savings Codes/CSBaselineVFISol.jld","CSBaselineSoln",CSBaselineSoln)
    else
		save("/Users/a1159308/Dropbox/Jake's Julia Codes/Post-Version 1.0/Consumption Savings Codes/CSBaselineVFISol.jld","CSBaselineSoln",CSBaselineSoln)
	end
end

fignum = 1
## Plotting:
plotDR = 1
if plotDR == 1
	# fig = figure()
	# ax  = fig[:gca](projection = "3d")
	# ax[:plot_surface](X,Y,Z,cmap = ColorMap("jet"))
	# ax[:set_xlabel]("x")
	# ax[:set_ylabel]("y")
	# ax[:set_zlabel]("z")

	fig = figure("Consumption Decision Rule")
	subplot(1,2,1)
	plot(xgrid,DRc[1:nx,:],"b")
	plot(xgrid,DRc[nx+1:2*nx,:],"r")
	plot(xgrid,DRc[2*nx+1:3*nx,:],"k")
	plot(xgrid,DRc[3*nx+1:4*nx,:],"m")
	plot(xgrid,DRc[4*nx+1:5*nx,:],"g")
	axis([minimum(xgrid), maximum(xgrid),minimum(DRc),maximum(DRc)])
    title("Consumption Decision Rule")
	subplot(1,2,2)
	plot(xgrid,SDRc[1:nx,:],"b")
	plot(xgrid,SDRc[nx+1:2*nx,:],"r")
	plot(xgrid,SDRc[2*nx+1:3*nx,:],"k")
	plot(xgrid,SDRc[3*nx+1:4*nx,:],"m")
	plot(xgrid,SDRc[4*nx+1:5*nx,:],"g")
	axis([minimum(xgrid), maximum(xgrid),minimum(SDRc),maximum(SDRc)])
    title("Smoothed Consumption Decision Rule")
end


## Simulate Model
sim_model = 1
if sim_model == 1
    #------------------------
    # Simulating the Model : 
    #------------------------

    # First simulate a sequence for the income.
    cumPI   = zeros(nstateY,nstateY)
    for k in 1 : nstateY
        tmp = 0
        for kk in 1 : nstateY
            tmp         = tmp + PI[k,kk]
            cumPI[k,kk] = tmp
        end
    end
    s0      = ceil(nstateY/2)  # Set the initial state to equal the state with the mean of the income process. (There are nstateY states.)
    Brn     = 500
    T       = Brn+5000      # Total number of periods to simulate.
    Random.seed!(123)		        # Reset the random number generator to use the seed given by "seed"
    dU   	= Uniform(0,1)
    p       = rand(dU,T)    # Draw T realizations of a random variable that is uniformly distributed over the [0,1] interval.  These
                            # can be treated as probabilities.
    drw     = convert(Array{Int64,1},dropdims(zeros(T,1),dims=2))    # The j's will be the TFP realizations.
    drw[1]  = s0
    for k in 2 : T
        drw[k]    = minimum(findall(cumPI[drw[k-1],:] .> p[k]))
    end
    ChainY	= Yt[drw]

    # Now construct a vector indicating the state for income in each period
    statevec    = convert(Array{Int64,1},dropdims(zeros(T,1),dims=2))
    for kk in 1 : T
        for k in 1 : nstateY
            if ChainY[kk] == Yt[k]
                statevec[kk]   = k
            end
        end            
    end
    SDRx	= dropdims(SDRx,dims=2)
    SDRc	= dropdims(SDRc,dims=2)
    # Construct time series for all variables:
    Xvec    = zeros(T,1);
    Cvec    = zeros(T,1);
    Xvec[1] = 20;
    for t = 1 : T
        i       = statevec[t]
        xt      = Xvec[t]
        xp      = LinearInterp1D(xgrid,SDRx[(i-1)*nx+1:i*nx,:],xt)
        ct		= LinearInterp1D(xgrid,SDRc[(i-1)*nx+1:i*nx,:],xt)
        Cvec[t] = ct
        if t < T
            if xp <= xmin
                Xvec[t+1] = xmin  
            else
                Xvec[t+1] = xp
            end
        end
    end
    println("Hi")

    fig = figure("Simulated Time-Series")
    subplot(2,1,1)
    plot(collect(1:T),ChainY[1:T],"b", label="y(t)")
    plot(collect(1:T), Cvec, "r", label="c(t)")
    legend()
    axis([1, T,minimum(ChainY)-1,maximum(ChainY)+1])
    axis([1, T, minimum([minimum(Cvec),minimum(ChainY)]),maximum([maximum(Cvec),maximum(ChainY)])])
    xlabel("t")
    ylabel(L"$y_{t}$")
    title("Income")
    subplot(2,1,2)
    plot(collect(1:T), Xvec, "b", label="x(t)")
    axis([1, T, minimum(Xvec), maximum(Xvec)])
    xlabel("t")
    ylabel(L"$c_{t}, x_{t}$")
    title("Consumption and Savings")
end