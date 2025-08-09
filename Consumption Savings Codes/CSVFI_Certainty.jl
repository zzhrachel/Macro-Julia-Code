using PythonPlot
using HDF5, JLD
include("LinearInterp1D.jl")
include("chebychev.jl")

# This code solves for a benchmark consumption-savings problem when the individual is subject to a borrowing rate that is weakly greater 
# than that of the borrowing rate.  I will assume, in this code, that the subjective discount factor is less than or equal to the inverse of
# the gross lending rate (so that beta*Rl <= 1).  There is no income risk in this version of the model.

struct CSVFI_Soln
	beta :: Real
	Rb :: Real
	Rl :: Real
	y  :: Real
  	sigma :: Real
	pref :: Int64
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
y     = 1

pref  = 1;
if pref == 0
    util(x)   = log(x)
else
    util(x,sigma)  = (x.^(1-sigma) .- 1)/(1-sigma) .- 1/(1-sigma)       # Utility function
    sigma   = 2.0;
end

# Define Grid for savings.
xmin = -0.99*y/(Rb-1)               # Minimum value for x.  Must be very careful with this setting of xmin or results will be nonsensical
xmax = y/(Rl-1)                     # Maximum value for x.  
nx   = 251                          # Number of gridpoints for x grid.
nxp  = 1501                         # Number of gridpoints for x' grid.
# xgrid = collect(linspace(xmin,xmax,nx))    # Construct linear grid for x and store in a column vector.
xgrid = collect(range(xmin,stop=xmax,length=nx))    # Construct linear grid for x and store in a column vector.

# Define matrices for consumption and continuation savings.
cmin = 1e-6

function ConstructCandX(cmin :: Real, xgrid :: Vector)
	# For each level of outut I need to construct a matrix for consumption and continuation savings where each row is for a 
	# given level of saving and each column is for a given level of consumption.
	chk  = findall(xgrid .<= 0)  # Finds values of xgrid that are less than zero.

	xneg = xgrid .< 0
	xnonneg = xgrid .>= 0
	cmax  = Rb*xneg.*xgrid+Rl*xnonneg.*xgrid + y*ones(nx,1) .- xmin

	CMat     = zeros(nx,nxp)
	for i in 1 : nx
	    #CMat[i,:] = linspace(cmin,cmax[i],nxp)
	    CMat[i,:] = range(cmin,stop=cmax[i],length=nxp)
	end
	XPMat = repeat(Rb*xneg.*xgrid+Rl*xnonneg.*xgrid,1,nxp) + y*ones(nx,nxp) - CMat;
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

function ConstructVP(nx :: Int64, nxp :: Int64, xgrid :: Array, XPMat :: Array, V :: Array)
    VP  = zeros(nx,nxp)
    # vptmp = zeros(nx,nxp);
    for j = 1 : nx
        VPvec = LinearInterp1D(xgrid,V,collect(XPMat[j,:]'))
        VP[j,:] = VPvec'
    end
    return VP
end

function ConstructTV(pref :: Int64, c :: Array, VP :: Array, sigma :: Real)
    if pref == 0
        TV, ind = findmax(util(c) + beta*VP,dims=2)
    else
        TV,ind  = findmax(util(c,sigma) + beta*VP,dims=2)
    end
    return TV, ind
end

#-------------------------
# Run the VFI Algorithm :
#-------------------------

CMat, XPMat = ConstructCandX(cmin, xgrid)

loadsoln = 0
if loadsoln == 0
	global V   = ones(nx,1)
	global VP  = repeat(V,1,nxp)
	global DRx = zeros(nx,1)
	global DRc = zeros(nx,1)
else
	CSVFISoln = load("/Users/jacobwong/Dropbox/Jake's Julia Codes/Posted Class Codes f1.8/Consumption Savings Codes/CSVFI_CertaintySol.jld","CSVFICertainty_Soln")
	V0	= CSVFI_Soln.V
	xgrid0 = CSVFI_Soln.xgrid
	nx0	= CSVFI_Soln.nx
	global V = zeros(nx,1)
	V = LinearInterp1D(xgrid0,V0,xgrid)  # Using saved solution, if size of nx grid is different than that of the saved solution (nx0), 
										 # interpolate into the grid used in this application of the code.
	global VP  = repeat(V,1,nxp)
	global DRx = zeros(nx*nstateY,1)
	global DRc = zeros(nx*nstateY,1)
end

crt	= 1
tol	= 1e-6
@time while crt >= tol
	global VP  = ConstructVP(nx, nxp, xgrid, XPMat, V)
	TV, ind = ConstructTV(pref, CMat, VP, sigma)
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
nd      = 51        # Number of chebychev polynomials used in chebychev polynomial approximations
zd      = trsfc(xgrid,xmin,xmax)
TD      = chebychev(zd,nd)

SDRx    = zeros(nx,1)
SDRc    = zeros(nx,1)
B       = TD\DRx
Bc      = TD\DRc
SDRx    = TD*B      # Smoothed decision rule for savings
SDRc    = TD*Bc		# Smoothed decision rule for savings

# Save the solution in a structure:
CSVFICertainty_Soln = CSVFI_Soln(beta,Rb,Rl,y,sigma,pref,nx,nxp,xgrid,xmin,xmax,XPMat,CMat,V,DRx,DRc,SDRx,SDRc)


savesoln = 0 # set to 1 if wanting to save the solution.
desktop  = 0 # set to 1 if using university desktop.
if savesoln == 1
    desktop = 0
    if desktop == 0
    	save("/Users/jacobwong/Dropbox/Jake's Julia Codes/Posted Class Codes v1.8/Consumption Savings Codes/CSBaselineVFISol.jld","CSBaselineSoln",CSBaselineSoln)
    else
		save("/Users/a1159308/Dropbox/Jake's Julia Codes/Posted Class Codes v1.8/Consumption Savings Codes/CSBaselineVFISol.jld","CSBaselineSoln",CSBaselineSoln)
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
	plot(xgrid,DRc,"b")
	axis([minimum(xgrid), maximum(xgrid),minimum(DRc),maximum(DRc)])
    title("Consumption Decision Rule")
	subplot(1,2,2)
	plot(xgrid,SDRc,"b")
	axis([minimum(xgrid), maximum(xgrid),minimum(SDRc),maximum(SDRc)])
    title("Smoothed Consumption Decision Rule")
end


## Simulate Model
sim_model = 1
if sim_model == 1
    #------------------------
    # Simulating the Model : 
    #------------------------
    SDRx	= dropdims(SDRx,dims=2)
    SDRc	= dropdims(SDRc,dims=2)
    T       = 100  # Number of periods in simulation
    # Construct time series for all variables:
    Xvec    = zeros(T,1)
    Cvec    = zeros(T,1)
    Xvec[1] = 0  # Set value of initial wealth
    for t = 1 : T
        xt      = Xvec[t]
        xp      = LinearInterp1D(xgrid,SDRx,xt)
        ct		= LinearInterp1D(xgrid,SDRc,xt)
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
    plot(collect(1:T), Cvec, "r", label="c(t)")
    axis([1, T, minimum(Cvec),maximum(Cvec)])
    # xlabel("t")
    ylabel(L"$c_{t}$")
    title("Consumption")
    subplot(2,1,2)
    plot(collect(1:T), Xvec, "b", label="x(t)")
    axis([1, T, minimum(Xvec), maximum(Xvec)])
    xlabel("t")
    ylabel(L"$x_{t}$")
    title("Consumption and Savings")
end
println("All Done")
