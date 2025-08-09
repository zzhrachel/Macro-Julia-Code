function rouwenhorst(rho :: Real,sigma_eps :: Real,n :: Int64)
	# rouwenhorst.jl
	#
	# zgrid, P = rouwenhorst(rho, sigma_eps, n)
	#
	# rho is the 1st order autocorrelation
	# sigma_eps is the standard deviation of the error term
	# n is the number of points in the discrete approximation


	mu_eps = 0

	q = (rho+1)/2
	nu = ((n-1)/(1-rho^2))^(1/2) * sigma_eps

	P = [q 1-q;1-q q]


	for i in 2:n-1	
	   	P = q*[P zeros(i,1);zeros(1,i+1)] + (1-q)*[zeros(i,1) P;zeros(1,i+1)] + (1-q)*[zeros(1,i+1); P zeros(i,1)] + q*[zeros(1,i+1); zeros(i,1) P];
		P[2:i,:] = P[2:i,:]/2;
	end
	#zgrid = collect(linspace(mu_eps/(1-rho)-nu,mu_eps/(1-rho)+nu,n))  # linspace() is deprecated since version 1.0.  Use "range()".
	zgrid = collect(range(mu_eps/(1-rho)-nu,stop=mu_eps/(1-rho)+nu,length=n))

	return zgrid, P
end
