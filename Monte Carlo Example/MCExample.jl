using Distributions
using PythonPlot
using StatsBase

# Set parameter values:
rho = 0.95      # AR(1) coefficient (persistence of AR(1) process)
mu  = 0.0       # Mean of shocks to AR(1) process   
sigma = 0.029   # Standard deviation of shocks to AR(1) process

T    = 1000     # Length of one sample
nsim = 300000   # Number of simulations in an ensemble

function CEnsemble(rho :: Float64, mu :: Float64, sigma :: Float64, T :: Int64, nsim :: Int64)
	EnsRho = zeros(nsim,1)   # Vector to save mean of each sample
	EnsStd = zeros(nsim,1)   # Vector to save std of shocks
	for i in 1 : nsim
		dist = Normal(mu,sigma)
	    evec = rand(dist,T)  # Generate vector of shocks to AR(1) process for sample i of the ensemble
	    yvec = zeros(T,1)    # Initiate vector to save values of y(t) in sample i
	    y0   = 0    		 # Initial condition for y(t).
	    for t in 1 : T
	        if t == 1
	            yvec[1] = rho*y0 + evec[1]
	        else
	            yvec[t] = rho*yvec[t-1] + evec[t];
	        end
	    end
	    # Run OLS regression to retrieve an estimate of rho and sigma given sample i (current vector of yvec)
	    Y   = yvec[2:end,:]
	    X   = yvec[1:end-1,:]
	    rhohat = X\Y    # X\Y returns the same result as inv(X'*X)*X'*Y!  This is the sample estimate for rho
	    resid  = Y - rhohat.*X  # Construct OLS residuals
	    sigmahat = std(resid)   # Calculate sample estimate of sigma.
	    EnsRho[i] = rhohat[1]  # rhohat is a Vector type even though it is a 1 x 1 vector by construction.  Need to extract 
	    						 # its value in a Float64 type.
	    EnsStd[i] = sigmahat
	    println(i)
	end
	return EnsRho, EnsStd
end

function MidPoints(binedges :: Array)
	mdpnts = (binedges[2:end] + binedges[1:end-1])/2
	return mdpnts
end

## Create a plot of density functions using the hist command:
# Create a function that takes the data, "d" and creates a maximum and minimum range.  This function uses the number of bins and their edges.
function pdfbarplot_bins(d :: Array{Float64,1}, dedges :: Array{Float64,1}, N :: Int64)
    # Old Line: v0.4.3:
    # dhist  = hist(d,dedges)
    dhistogram = fit(Histogram,d,dedges,closed=:right)
    dhist  = dhistogram.weights
    dhist_norm = dhist./N
    cdfdh  = [0 ; cumsum(dhist_norm)]
    delta  = dedges[2] - dedges[1]
    dpdf   = (cdfdh[2:end] - cdfdh[1:end-1])./delta
    return dpdf
end

@time EnsRho, EnsStd = CEnsemble(rho, mu, sigma, T, nsim)

EnsRho = sort(dropdims(EnsRho,dims=2))  # Sort the values in the vector EnsRho (just for fun)
EnsStd = sort(dropdims(EnsStd,dims=2))  # Sort the values in the vector EnsStd (just for fun)

# Plot pdf and cdf (not histogram) for EnsRho and EnsStd: 
# Need to normalize by the area under the histogram when plotting pdf.
nbinsRho = 501  # Number of bin edges to be used in histogram of rho estimates.
rhobins = collect(range(0.7,stop=1,length=nbinsRho))   # Create the edges of the bins to be used in the histogram
rdelta  = rhobins[2] - rhobins[1]    # Length of each bin
# Old Line: v.0.4.3
# rhobins, rcounts = hist(EnsRho,rhobins) # rcounts includes the number of observations at the last bin edge
rhist   = fit(Histogram,EnsRho,rhobins,closed=:right) # Note that intervals can be closed on the left or right when choosing to construct histogram.
													  # If we are worried about observations at the edges of the first AND last bins then we must do something
													  # slightly different.
rcounts = rhist.weights 
rmidpnt = MidPoints(rhobins) # Calculate midpoints of bins (for plot)

sbinsRho = 501   # Number of bin edges to be used in histogram of std. dev. estimates.
stdbins = collect(range(minimum(EnsStd),stop=maximum(EnsStd),length=sbinsRho)) # Create the edges of the bins to be used in the histogram
sdelta  = stdbins[2] - stdbins[1]    # Length of each bin
# Old Line: v.0.4.3
# stdbins, scounts = hist(EnsStd,stdbins) # scounts includes the number of observations at the last bin edge
shist   = fit(Histogram,EnsStd,stdbins,closed=:right)
smidpnt = MidPoints(stdbins)  # Calculate midpoints of bins (for plot)


rhopdf = pdfbarplot_bins(EnsRho, rhobins, nsim)
stdpdf = pdfbarplot_bins(EnsStd, stdbins, nsim)

rhocdf = cumsum(rhopdf*rdelta)
stdcdf = cumsum(stdpdf*sdelta)


fig, ax = subplots(1)
subplot(2,2,1)
bar(rmidpnt, rhopdf, rhobins[2:end]-rhobins[1:end-1])
#plot(rmidpnt,rhopdf,"r")
axis([minimum(rmidpnt[:,1]), maximum(rmidpnt[:,1]), 0, maximum(rhopdf)])
xlabel(L"$\rho$")
title(L"p.d.f. of $\rho$")
subplot(2,2,2)
#plot(rmidpnt,rhocdf,"r")
bar(rmidpnt, rhocdf, rhobins[2:end]-rhobins[1:end-1])
axis([minimum(rmidpnt[:,1]), maximum(rmidpnt[:,1]), 0, 1])
xlabel(L"$\rho$")
title(L"c.d.f. of $\rho$")
subplot(2,2,3)
bar(smidpnt, stdpdf, stdbins[2:end]-stdbins[1:end-1])
#plot(smidpnt,stdpdf,"r")
axis([minimum(smidpnt[:,1]), maximum(smidpnt[:,1]), 0, maximum(stdpdf)])
xlabel(L"$\sigma$")
title(L"p.d.f. of $\sigma$")
subplot(2,2,4)
bar(smidpnt, stdcdf, stdbins[2:end]-stdbins[1:end-1])
#plot(smidpnt,stdcdf,"r")
axis([minimum(smidpnt[:,1]), maximum(smidpnt[:,1]), 0, 1])
xlabel(L"$\sigma$")
title(L"c.d.f. of $\sigma$")

println("All Done")