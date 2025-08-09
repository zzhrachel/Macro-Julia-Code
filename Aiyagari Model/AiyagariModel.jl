using PythonPlot
using Colors
using LaTeXStrings
using JLD
using Distributed
num_procs = 3
addprocs(num_procs)
worker_pool = workers()
@everywhere using Distributions
@everywhere using Random
@everywhere using SharedArrays
include("chebychev.jl")
include("rouwenhorst.jl")
#@everywhere include("LinearInterp1D.jl")

# This code solves for a version of Aiyagari's QJE model.  It is not identical but similar.  Y = z*K^alpha*L^(1-alpha)

#################################
#								#
#		Solution Notes:			#
#								#
#################################

# 1) Higher persistence in labour market states requires a larger value for amax.  At rho = 0.6, amax = w0*Lt[nstateL]/r0	 
# 	 works well with r0 = rho - 0.005 and rho = 1/beta - 1.

@everywhere struct Parameters
	lambda :: Real
	beta :: Real
	alpha :: Real
	delta :: Real
	z :: Real
  	sigma :: Real
	pref :: Int64
	rhoL :: Real
	seL :: Real
	mL :: Real
	nstateL :: Int64
	Lgrid :: Array
	PI :: Array
	Lt :: Array
	Lss :: Real
	na :: Int64
	nap :: Int64
	agrid :: Array
	amin :: Real
	amax :: Real
	APMat :: Array
	cmin :: Real
	nedges :: Int64
	nbins :: Int64
	bin_edges :: Array
	bin_width :: Real
	bin_midpt :: Array	
end	

@everywhere struct Aiyagari_Struct
	V :: Array
	DRa :: Array
	DRc :: Array
	SDRa :: Array
	SDRc :: Array
	a_hist :: Array
	a_dens :: Array
	a_cdf :: Array
	r :: Real
	w :: Real
end

# @everywhere struct RLow_Struct
# 	V_lo :: Array
# 	DRa_lo :: Array
# 	DRc_lo :: Array
# 	SDRa_lo :: Array
# 	SDRc_lo :: Array
# 	ahist_lo :: Array
# 	adens_lo :: Array
# 	acdf_lo :: Array
# end

# @everywhere struct RHi_Struct
# 	V_hi :: Array
# 	DRa_hi :: Array
# 	DRc_hi :: Array
# 	SDRa_hi :: Array
# 	SDRc_hi :: Array
# 	ahist_hi :: Array
# 	adens_hi :: Array
# 	acdf_hi :: Array
# end

#-------------------
# Define functions:
#-------------------

@everywhere function LinearInterp1D(x :: Array, y :: Array, z :: Array)
	nz	= length(z)
	nx	= length(x)
	intvals = zeros(nz,1)
	for i in 1 : nz
		zi  = z[i]
		lw	= searchsortedlast(x,zi)
		if lw == 1
			intvals[i] = y[1]
		elseif lw == nx
			intvals[i] = y[nx]
		else 
			xl  = x[lw]
			xh  = x[lw+1]
			slp = (y[lw+1] - y[lw])/(xh - xl)
			intvals[i] = y[lw] + slp*(zi - xl)
		end
	end
	return intvals
end

@everywhere function LinearInterp1D(x :: Array, y :: Array, z :: Real)
	nz	= length(z)
	nx	= length(x)
	intvals = 0	
	for i in 1 : nz
		zi  = z[i]
		lw	= searchsortedlast(x,zi)
		if lw == 1 && zi == x[1]
			intvals = y[1]
		elseif lw == nx
			intvals = y[nx]
		else 
			xl  = x[lw]
			xh  = x[lw+1]
			slp = (y[lw+1] - y[lw])/(xh - xl)
			intvals = y[lw] + slp*(zi - xl)
		end
	end
	return intvals
end


function LabourDistribution(Lt :: Array{Float64,1}, PI :: Array{Float64,2})
    #----------------------------------------
    # Simulating an individual's experience: 
    #----------------------------------------
    # First simulate a sequence for the income.
    cumPI   = zeros(nstateL,nstateL)
    for k in 1 : nstateL
        tmp = 0
        for kk in 1 : nstateL
            tmp         = tmp + PI[k,kk]
            cumPI[k,kk] = tmp
        end
    end
    s0      = ceil(nstateL/2)  # Set the initial state to equal the state with the mean of the income process. (There are nstateY states.)
    Brn     = 5000
    T       = Brn+100000     # Total number of periods to simulate.
    #Random.seed!(123);	    # Reset the random number generator to use the seed given by "seed"
    dU   	= Uniform(0,1)
    p       = rand(dU,T)    # Draw T realizations of a random variable that is uniformly distributed over the [0,1] interval.  These
                            # can be treated as probabilities.
    drw     = convert(Array{Int64,1},dropdims(zeros(T,1),dims=2))    # The j's will be the labour time realizations.
    drw[1]  = s0
    for k in 2 : T
        drw[k]    = convert(Int64,minimum(findall(cumPI[drw[k-1],:] .> p[k])))
    end
    ChainL	= Lt[drw]

    # Now construct a vector indicating the state for income in each period
    statevec    = convert(Array{Int64,1},dropdims(zeros(T,1),dims=2))
    for kk in 1 : T
        for k in 1 : nstateL
            if ChainL[kk] == Lt[k]
                statevec[kk]   = k
            end
        end            
    end
    # Calculate the fraction of observations in each labour state:
    L_hist = zeros(nstateL,1)
    for i in 1 : nstateL
    	L_hist[i] = size(findall(statevec .== i),1)
    end
    Lss = sum(Lt.*L_hist)/T
    return Lss, L_hist/T
end

function ConstructVP(na :: Int64, nap :: Int64, nstateL :: Int64, agrid :: Array, APMat :: Array, V :: Array)
    VP  = zeros(na*nstateL,nap);
    for i in 1 : nstateL
        vptmp = zeros(na,nap);
        for j = 1 : na
            VPvec = LinearInterp1D(agrid,V[(i-1)*na+1:i*na],collect(APMat[(i-1)*na+j,:]'))
            vptmp[j,:] = VPvec'
        end
        VP[(i-1)*na+1:i*na,:] = vptmp;
    end
    return VP
end

function ConstructEV(na :: Int64, nap :: Int64, nstateL :: Int64, PI :: Array, VP :: Array)
    EV  = zeros(na*nstateL,nap)
    for i in 1 : nstateL
        evp  = zeros(na,nap)
        for k in 1 : nstateL
            evp  = evp + PI[i,k]*VP[(k-1)*na+1:k*na,:];
        end
        EV[(i-1)*na+1:i*na,:] = evp
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

function VFI(Params :: Parameters, V0 :: Array{Float64,2}, r :: Real, w :: Real)
	Lt = Params.Lt
	sigma = Params.sigma
	pref = Params.pref
	na = Params.na
	nap = Params.nap
	nstateL = Params.nstateL
	amin = Params.amin
	amax = Params.amax
	agrid = Params.agrid
	APMat = Params.APMat
	PI  = Params.PI

	CMat = repeat((1+r)*agrid,nstateL,nap) + kron(w*Lt,ones(na,nap)) - APMat
	# Replace negative values of c with cmin
	neg_c = findall(CMat .<= cmin)
	CMat[neg_c] .= cmin

	Vt  = V0
	DRa = zeros(na*nstateL,1)
	DRc = zeros(na*nstateL,1)
	crt	= 1
	tol	= 1e-6
	while crt >= tol
		VPt  = ConstructVP(na, nap, nstateL, agrid, APMat, Vt)
		EVP  = ConstructEV(na, nap, nstateL, PI, VPt)
		TV, ind = ConstructTV(pref, CMat, EVP, sigma)
		DRa  = APMat[ind]
		DRc  = CMat[ind]
		crt	 = maximum(abs.(Vt - TV)./(1+maximum(abs.(Vt))))
	    #println(crt)
	    Vt   = copy(TV)
	end
	# Run a simple regression to "smooth out the decision rule" for use in simulations.
	nda     = 3;                      # # of chebychev polynomials used in chebychev polynomial approximations
	zda     = trsfc(agrid,amin,amax);
	TDa     = chebychev(zda,nda);
	ndc     = 12;                      # # of chebychev polynomials used in chebychev polynomial approximations
	zdc     = trsfc(agrid,amin,amax);
	TDc     = chebychev(zdc,ndc);

	SDRa    = zeros(na*nstateL,1);
	SDRc    = zeros(na*nstateL,1);
	for i = 1 : nstateL
	    Ba      = TDa\DRa[(i-1)*na+1:i*na,1]
	    Bc      = TDc\DRc[(i-1)*na+1:i*na,1]
	    SDRa[(i-1)*na+1:i*na,1] = TDa*Ba     # Smoothed decision rule for savings
	    SDRc[(i-1)*na+1:i*na,1] = TDc*Bc		# Smoothed decision rule for savings
	end
	return Vt, DRa, DRc, SDRa, SDRc
end

@everywhere function SimulateIndividualHistory(Params :: Parameters, AiyagariSoln :: Aiyagari_Struct)
	nstateL = Params.nstateL
	Lt = Params.Lt
	PI = Params.PI
	na = Params.na
	amin = Params.amin
	amax = Params.amax
	agrid = Params.agrid
	bin_edges = Params.bin_edges
	bin_midpt = Params.bin_midpt

	SDRa = AiyagariSoln.SDRa
	a_cdf = AiyagariSoln.a_cdf

    #----------------------------------------
    # Simulating an individual's experience: 
    #----------------------------------------
    # First simulate a sequence for the income.
    cumPI   = zeros(nstateL,nstateL)
    for k in 1 : nstateL
        tmp = 0
        for kk in 1 : nstateL
            tmp         = tmp + PI[k,kk]
            cumPI[k,kk] = tmp
        end
    end
    s0      = ceil(nstateL/2)  # Set the initial state to equal the state with the mean of the income process. (There are nstateY states.)
    Brn     = 500
    T       = Brn+1000      # Total number of periods to simulate.
    #Random.seed!(123);	    # Reset the random number generator to use the seed given by "seed"
    dU   	= Uniform(0,1)
    p       = rand(dU,T)    # Draw T realizations of a random variable that is uniformly distributed over the [0,1] interval.  These
                            # can be treated as probabilities.
    drw     = convert(Array{Int64,1},dropdims(zeros(T,1),dims=2))    # The j's will be the labour time realizations.
    drw[1]  = s0
    for k in 2 : T
        drw[k]    = convert(Int64,minimum(findall(cumPI[drw[k-1],:] .> p[k])))
    end
    ChainL	= Lt[drw]

    # Now construct a vector indicating the state for income in each period
    statevec    = convert(Array{Int64,1},dropdims(zeros(T,1),dims=2))
    for kk in 1 : T
        for k in 1 : nstateL
            if ChainL[kk] == Lt[k]
                statevec[kk]   = k
            end
        end            
    end

    ## Construct the bin movements of the individual:    
    bin_vec = zeros(T,1)
    avec    = zeros(T,1)
    # Draw an initial bin:
    ai_bin  = convert(Int64,minimum(findall(a_cdf[:,statevec[1]] .> rand(dU))))
    avec[1] = bin_midpt[ai_bin]
    for t = 1 : T
        i       = statevec[t]
        at      = avec[t]
        ap      = LinearInterp1D(agrid,SDRa[(i-1)*na+1:i*na,:],at)
        if t < T
            if ap <= amin
                avec[t+1] = amin  
            elseif ap >= amax
            	avec[t+1] = agrid[na-1]
            else
                avec[t+1] = ap
            end
        end
    end
    i_vec = [avec[end] statevec[end]]
	return i_vec
end 

function SimulateDistribution(Params :: Parameters, AiyagariSoln :: Aiyagari_Struct)
	nstateL = Params.nstateL
	nbins = Params.nbins
	bin_edges = Params.bin_edges
	bin_width = Params.bin_width
	a_cdf = AiyagariSoln.a_cdf

	N_ind = 100000
	Ind_Histories = SharedArray{Float64,2}(zeros(N_ind,2))
	#a_histogram = SharedArray{Float64,2}(zeros(nbins,nstateL))
	@time @sync @distributed for i in 1 : N_ind
		 Ind_Histories[i,:] = SimulateIndividualHistory(Params,AiyagariSoln)
	end
	# Bin Individuals by Wealth:
	wealth_vec = Ind_Histories[:,1]
	l_vec = convert(Array{Int,1},Ind_Histories[:,2])
	a_histogram = zeros(Int64,nbins,nstateL)
	for i in 1 : N_ind
		i_bin = maximum(findall(bin_edges .<= wealth_vec[i]),dims=1)
		a_histogram[i_bin[1],l_vec[i]] += 1
	end
	a_hist = a_histogram/N_ind
	a_dens = (a_hist/bin_width)
	a_cdf  = cumsum(a_dens,dims=1)./sum(a_dens,dims=1)    # Set-up conditional cdf for each labour type.
	return a_hist, a_dens, a_cdf
end

function ConstructAggregatesAndPrices(Params :: Parameters, bin_width :: Real, bin_midpt :: Array, a_dens :: Array)
	alpha   = Params.alpha
	z		= Params.z
	delta   = Params.delta
	nstateL = Params.nstateL
	Lt 	    = Params.Lt

	l_dist = zeros(nstateL,1)
	for i in 1 : nstateL
		l_dist[i] = sum(a_dens[:,i]*bin_width)
	end
	# Construct aggregate labour supply:
	l_dist = l_dist/sum(l_dist)
	L = sum(l_dist.*Lt)
	# Construct aggregate capital stock:
	#K = sum(a_dens.*bin_midpt*bin_width)
	K = 0
	for j in 1 : nstateL
		K += sum(a_dens[:,j].*bin_midpt*bin_width)*l_dist[j]
	end
	wp = (1-alpha)z*(K/L)^alpha
	rp = alpha*z*(L/K)^(1-alpha) - delta	
	return wp, rp			
end

function UpdateDistribution(Params :: Parameters, AiyagariSoln :: Aiyagari_Struct, a_hist1 :: Array{Float64,2}, a_dens1 :: Array{Float64,2})
	lambda  = Params.lambda
	alpha   = Params.alpha
	z		= Params.z
	delta   = Params.delta
	nstateL = Params.nstateL
	Lt 	    = Params.Lt
	bin_width = Params.bin_width
	bin_midpt = Params.bin_midpt

	a_dens0   = AiyagariSoln.a_dens
	a_hist0   = AiyagariSoln.a_hist

	hist_update = lambda*a_hist1 + (1-lambda)*a_hist0
	dens_update = lambda*a_dens1 + (1-lambda)*a_hist0

	a_hist = hist_update
	a_dens = dens_update
	a_cdf  = cumsum(a_dens,dims=1)./sum(a_dens,dims=1)    # Set-up conditional cdf for each labour type.

	l_dist = zeros(nstateL,1)
	for i in 1 : nstateL
		l_dist[i] = sum(a_dens[:,i]*bin_width)
	end
	# Construct aggregate labour supply:
	l_dist = l_dist/sum(l_dist)
	L = sum(l_dist.*Lt)
	# Construct aggregate capital stock:
	#K = sum(a_dens.*bin_midpt*bin_width)
	K = 0
	for j in 1 : nstateL
		K += sum(a_dens[:,j].*bin_midpt*bin_width)*l_dist[j]
	end
	w = (1-alpha)z*K^alpha*L^(-alpha)
	r = alpha*z*K^(alpha-1)*L^(1-alpha) - delta
	# println("Steady state labour supply is approximately: $(L)")
	# println("Steady state capital stock is approximately: $(K)")
	# println("Steady state wage is approximately: $(wp)")
	# println("Steady state rental rate of capital is approximately: $(rp)")

	return a_hist, a_dens, a_cdf, w, r
end


function EqmSolve(Params :: Parameters, AiyagariSoln0 :: Aiyagari_Struct)
	# Solve for steady state r by starting with a guess r0.  Using r0 calculate the optimal savings decision rule of the individual.
	# With the optimal decision rule, simulate a distribution for asset savings.  Using this simulated distribution, calculate the 
	# interest rate that is consistent with market clearing.  Call this interest rate r1.  Save this r1.  Now we have two interest
	# rates, (r0,r1).  Without loss of generality, suppose that r1 > r0 and call r_low = r0 and r_hi = r1.  Take the midpoint
	# r_mid = (r_low + r_hi)/2.  Using r_mid, calculate the optimal savings decision rule and simulate a distribution of savings
	# that is then used to calculate the market clearing interest rate, r_new.  If r_new > r_mid, replace r_hi with r_new.  Otherwise
	# replace r_low with r_new.  Now repeat the exercise.  Keep doing this until the r_new converges to an interest rate that does
	# not change across iterations.
	alpha = Params.alpha
	z 	  = Params.z
	Lss   = Params.Lss
	nedges = Params.nedges
	nbins = Params.nbins
	bin_edges = Params.bin_edges
	bin_width = Params.bin_width
	bin_midpt = Params.bin_midpt
	a_hist = AiyagariSoln0.a_hist
	a_dens = AiyagariSoln0.a_dens
	a_cdf  = AiyagariSoln0.a_cdf

	# Initialize initial interest rate, r0:
	V0 = AiyagariSoln0.V
	r0 = AiyagariSoln0.r
	w0 = AiyagariSoln0.w

	# Construct r1:
	V1, DRa1, DRc1, SDRa1, SDRc1 = VFI(Params, V0, r0, w0)
	# With decision rules in hand, now simulate a distribution of individuals and update until the wage is unchanged (note r0, w0)
	AiyagariSoln1 = Aiyagari_Struct(V1,DRa1,DRc1,SDRa1,SDRc1,a_hist,a_dens,a_cdf,r0,w0)
	a_hist1, a_dens1, a_cdf1 = SimulateDistribution(Params,AiyagariSoln1)
	w1, r1 = ConstructAggregatesAndPrices(Params, bin_width, bin_midpt, a_dens)

	# Now I have my initial pair of interest rates, r0 and r1.
	if r0 <= r1
		r_lo = r0
		w_lo = w0
		r_hi = r1
		w_hi = w1
	else
		r_lo = r1
		w_lo = w1
		r_hi = r0
		w_hi = w0
	end
	K_lo = Lss*(alpha*z/r_lo)^(1/(1-alpha))
	K_hi = Lss*(alpha*z/r_hi)^(1/(1-alpha))

	r_mid = (r_hi + r_lo)/2
	# Construct the capital stock and wage consistent with a steady state interest rate of r_mid
	K_mid = Lss*(alpha*z/r_mid)^(1/(1-alpha))
	w_mid = (1-alpha)*z*(K_mid/Lss)^alpha

	V_m, DRa_m, DRc_m, SDRa_m, SDRc_m = VFI(Params, V0, r_mid, w_mid)
	AiyagariSoln = Aiyagari_Struct(V_m,DRa_m,DRc_m,SDRa_m,SDRc_m,a_hist1,a_dens1,a_cdf1,r_mid,w_mid)
	bean = 1
	diff = 1
	while diff >= 1e-4
		V_m, DRa_m, DRc_m, SDRa_m, SDRc_m = VFI(Params, V_m, r_mid, w_mid)
		# With decision rules in hand, now simulate a distribution of individuals and update until the wage is unchanged:
		AiyagariSoln = Aiyagari_Struct(V_m,DRa_m,DRc_m,SDRa_m,SDRc_m,a_hist1,a_dens1,a_cdf1,r_mid,w_mid)
		a_hist_m, a_dens_m, a_cdf_m = SimulateDistribution(Params,AiyagariSoln)
		wp, rp = ConstructAggregatesAndPrices(Params,bin_width,bin_midpt,a_dens_m)
		# println("Steady state labour supply is approximately: $(L)")
		# println("Steady state capital stock is approximately: $(K)")
		# println("Steady state wage is approximately: $(wp)")
		# println("Steady state rental rate of capital is approximately: $(rp)")
		if rp > r_mid
			r_lo = r_mid
			w_lo = w_mid
			println("rp > r_mid")
		else
			r_hi = r_mid
			w_hi = w_mid
			println("rp < r_mid")
		end
		diff = abs(r_mid - (r_hi + r_lo)/2)/r_mid
		println("The interest rates during iteration $(bean) are r_mid=$(r_mid) and rp=$(rp)")		
		println("The stopping criterion during iteration $(bean) is currently: $(diff)")
		# Update r_mid:
		r_mid = (r_hi + r_lo)/2
		# Construct the capital stock and wage consistent with a steady state interest rate of r_mid
		K_mid = Lss*(alpha*z/r_mid)^(1/(1-alpha))
		w_mid = (1-alpha)*z*(K_mid/Lss)^alpha
		# Update AiyagariSoln_mid with the distributions and prices:
		AiyagariSoln = Aiyagari_Struct(V_m,DRa_m,DRc_m,SDRa_m,SDRc_m,a_hist_m,a_dens_m,a_cdf_m,r_mid,w_mid)
		bean += 1
	end
	return AiyagariSoln
end	


#--------------------
# Define Parameters:
#--------------------
lambda = 0.25   # distribution adjustment parameter
beta  = 0.96	# Subjective discount factor.  Make sure that beta <= 1!
delta = 0.08	# Depreciation rate of capital
alpha = 0.36	# Capital's share of aggregate output.
z     = 1		# TFP.

pref  = 1;
if pref == 0
    util(x)   = log(x)
else
    util(x,sigma)  = (x.^(1-sigma).-1)/(1-sigma) .- 1/(1-sigma)       # Utility function
    sigma   = 3.0;
end

#----------------------------------------------------------------------------------------------------
# Constructing the Markov Chain (Tauchen-Hussey, 1991 following Floden's code (Economics Letters)) :
#----------------------------------------------------------------------------------------------------

rhoL     = 0.9    				# Persistance of labour time
seL      = 0.4*sqrt(1-rhoL^2)   # Volatility of labour time
mL       = 0.0    				# mean
nstateL  = 7      				# Number of states for labour time

Lgrid, PI = rouwenhorst(rhoL,seL,nstateL)  # Markov chain grid and transition probability matrix
Lt  = exp.(Lgrid .+ mL)*(35/(7*24))		   # Transformed Markov chain grid into labour supply units.

rho = 1/beta - 1 		  # Subjective rate of time preference. 
# Make an initial guess of r and w to construct the grid for a.
r0  = rho - 0.005  		  # If I set this to 0.98/beta - 1 the maximum a' < amax.  For Bisection Method of Aiyagari, start rss < 1/beta-1
Lss, L_hist = LabourDistribution(Lt, PI)  # Construct the steady state aggregate labour supply (which is exogenous in this model)
K0  = Lss*(alpha*z/(r0 + delta))^(1/(1-alpha)) # Capital stock consistent with initial guess of steady state interest rate, r0.
w0  = (1-alpha)*z*(alpha*z/(r0+delta))^(alpha/(1-alpha)) # Wage rate consistent with initial guess of steady state interest rate, r0.

## Define Grid for savings.
amin = 0 #-0.99*wss*Lt[1]/rss # Minimum value for a.  Must be very careful with this setting of amin or results will be nonsensical
amax = 2*w0*Lt[nstateL]/r0		# Maximum value for a.
na   = 151					 	# Number of gridpoints for a grid.
nap  = 10*na                 	# Number of gridpoints for a' grid.
agrid = collect(range(amin,stop=amax,length=na))
apgrid = collect(range(amin,stop=amax,length=nap))
APMat = kron(repeat(apgrid',na,1),ones(nstateL,1))

# Initialize histogram:
nedges = na           # Number of bin edges for a grid of histogram bins 
nbins = nedges-1	  # Number of bins for a grid of histogram bins	
bin_edges = collect(range(amin,stop=amax,length=nedges))
bin_width = bin_edges[2] - bin_edges[1]
nbins = size(bin_edges,1)-1
bin_midpt = 0.5*(bin_edges[2:nbins+1] + bin_edges[1:nbins])

cmin = 1e-8
Params = Parameters(lambda,beta,alpha,delta,z,sigma,pref,rhoL,seL,mL,nstateL,Lgrid,PI,Lt,Lss,na,nap,agrid,amin,amax,APMat,cmin,
					nedges,nbins,bin_edges,bin_width,bin_midpt)

loadsoln = 1
if loadsoln == 0
	V_init   = ones(na*nstateL,1)

	r0  = rho - 0.005  		  # If I set this to 0.98/beta - 1 the maximum a' < amax.  For Bisection Method of Aiyagari, start rss < 1/beta-1
	Lss, L_hist = LabourDistribution(Lt, PI)  # Construct the steady state aggregate labour supply (which is exogenous in this model)
	K0  = Lss*(alpha*z/(r0 + delta))^(1/(1-alpha)) # Capital stock consistent with initial guess of steady state interest rate, r0.
	w0  = (1-alpha)*z*(alpha*z/(r0+delta))^(alpha/(1-alpha)) # Wage rate consistent with initial guess of steady state interest rate, r0.

	println("Steady state labour supply is approximately: $(Lss)")
	println("Initial guess at steady state capital stock is approximately: $(K0)")
	println("Initial guess at steady state interest rate is approximately: $(r0)")
	println("Initial guess at steady state wage rate is approximately: $(w0)")

	# Initiate wealth histogram, conditional density and conditional cdf:
	a_hist0 = 1/(nbins*nstateL).*ones(nbins,nstateL)
	a_dens0 = (a_hist0/bin_width)
	a_cdf0  = cumsum(a_dens0,dims=1)./sum(a_dens0,dims=1)    # Set-up conditional cdf for each labour type.
else
	Params0, AiyagariSoln0 = load("/Users/jacobwong/Dropbox/Macro IV Computer Codes/Julia Codes/Post-Version 1.0/Aiyagari Model/AiyagariModelSoln.jld","Params","AiyagariSoln")
	agrid0 	 = Params0.agrid
	na0		 = Params0.na
	nstateL0 = Params0.nstateL
	V0		 = AiyagariSoln0.V
	V_init   = zeros(na*nstateL,1)
	for i in 1 : nstateL
		V_init[(i-1)*na+1:i*na,:] = LinearInterp1D(agrid0,V0[(i-1)*na0+1:i*na0,:],agrid)
	end
	r0 = AiyagariSoln0.r
	w0 = AiyagariSoln0.w
	a_hist0 = AiyagariSoln0.a_hist
	a_dens0 = AiyagariSoln0.a_dens
	a_cdf0  = AiyagariSoln0.a_cdf
end


V0, DRa0, DRc0, SDRa0, SDRc0 = VFI(Params, V_init, r0, w0)

AiyagariSoln0 = Aiyagari_Struct(V0,DRa0,DRc0,SDRa0,SDRc0,a_hist0,a_dens0,a_cdf0,r0,w0)
@time AiyagariSoln  = EqmSolve(Params, AiyagariSoln0)

savesoln = 1
if savesoln == 1
    desktop = 0
    if desktop == 0
    	save("/Users/jacobwong/Dropbox/Jake's Julia Codes/Post-Version 1.0/Aiyagari Model/AiyagariModelSoln.jld","Params",Params,"AiyagariSoln",AiyagariSoln)
    else
    	save("/Users/a1159308/Dropbox/Jake's Julia Codes/Post-Version 1.0/Aiyagari Model/AiyagariModelSoln.jld","Params",Params,"AiyagariSoln",AiyagariSoln)
    end
end

#####################
#					#
#	Create Plots	#
#					#
#####################

# Define font dictionaries:
font1 = Dict("family"=>"serif",
#"color"=>"darkred",
#"weight"=>"normal",
"size"=>14)
#xlabel("Time",fontdict=font1)

font2 = Dict("family"=>"serif",
"name" => "times",
#"color"=>"darkred",
#"weight"=>"normal",
"size"=>16)
#xlabel("Time",fontdict=font1)

figure("Distribution Plots")
plot(bin_midpt,AiyagariSoln.a_dens[:,1],"blue",label=L"l_{1}")
plot(bin_midpt,AiyagariSoln.a_dens[:,2],"cyan",label=L"l_{2}")
plot(bin_midpt,AiyagariSoln.a_dens[:,3],"green",label=L"l_{3}")
plot(bin_midpt,AiyagariSoln.a_dens[:,4],"orange",label=L"l_{4}")
plot(bin_midpt,AiyagariSoln.a_dens[:,5],"red",label=L"l_{5}")
plot(bin_midpt,AiyagariSoln.a_dens[:,6],"purple",label=L"l_{6}")
plot(bin_midpt,AiyagariSoln.a_dens[:,7],"black",label=L"l_{7}")
legend(loc=1)
axis([bin_midpt[1], bin_midpt[end], 0, maximum(AiyagariSoln.a_dens)])
xlabel("Wealth",fontdict=font1)
title(L"Wealth Density Conditional on Labour Supply ($\rho = 0.9$)",fontdict=font2)

# Calculate the wealth distribution that is NOT conditional on labour supply:
a_dens = AiyagariSoln.a_dens
normalized_adens = a_dens./sum(a_dens*bin_width,dims=1)  # Sum of the densities along dimension 1 (down each column) equal 1.
unconditional_adens = sum(normalized_adens.*repeat(transpose(L_hist),nbins,1),dims=2)
unconditional_acdf = cumsum(unconditional_adens*bin_width,dims=1)

figure("Wealth Density Plot")
plot(bin_midpt,unconditional_adens,"r")
axis([bin_midpt[1], bin_midpt[end], 0, maximum(unconditional_adens)])
xlabel("Wealth",fontdict=font1)
title(L"Wealth Density ($\rho = 0.9$)",fontdict=font2)


####################
#				   #	
#   Lorenz Curve   #
#				   #
####################

# To construct the Lorenz Curve, I want a grid on [0,1] for my horizontal axis and the fraction of total wealth held on the vertical axis.
# What I have is the cdf for the wealth distribution which has bin_midpt on the horizontal axis and the measure of individuals with wealth less 
# than the bin_midpt bin on the vertical axis.  Therefore, I need to invert the relationship onto a grid on [0,1].

Agg_Wealth_Dist = cumsum(bin_midpt.*unconditional_adens*bin_width,dims=1)  # Cumulated wealth evaluated at each level of capital in
																	       # bin_midpt grid.  This lines up with the measure of 
																		   # individuals with wealth less than or equal to bin_midpt
																		   # as measured in a_cdf.
# Add extra points to ensure no extrapolation is required:
Agg_Wealth_Dist = vcat(0,Agg_Wealth_Dist)/Agg_Wealth_Dist[end]
# Normalize agrid:
pop_edges = collect(range(0,stop=1,length=nedges))
mid_pop   = vcat(0,(pop_edges[2:end] + pop_edges[1:end-1])/2)


Lorenz_curve = LinearInterp1D(dropdims(vcat(0,unconditional_acdf),dims=2),dropdims(Agg_Wealth_Dist,dims=2),mid_pop)

# Construct Gini Coefficient:
u_hist = 1/(nbins).*ones(nbins,1)
u_dens = (u_hist/bin_width)
u_cdf  = cumsum(u_dens*bin_width,dims=1)
u_cdf  = vcat(0,u_cdf/u_cdf[end])

Gini_Coeff = sum((u_cdf - Lorenz_curve)*bin_width,dims=1)



figure("Lorenz Curve")
plot(mid_pop,u_cdf,"b",label="Equal Wealth Distribution")
plot(mid_pop,Lorenz_curve,"r",label="Lorenz Curve")
axis([0, mid_pop[end], 0, 1])
xlabel("Fraction of Population",fontdict=font1)
ylabel("Fraction of Wealth",fontdict=font1)
title(L"The Lorenz Curve ($\rho = 0.9$)",fontdict=font2)


rmprocs(worker_pool)
println("All Done")


