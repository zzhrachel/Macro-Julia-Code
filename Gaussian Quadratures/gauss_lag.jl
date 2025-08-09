#-------------------------------------------------------------------------------
# Compute the coefficients of weights and nodes of the Gauss-Hermite quadrature
#-------------------------------------------------------------------------------

function gauss_lag(n :: Int64)
	d       = -collect(1:n-1)
	f       = collect(1:2:2*n-1)
	fc      = 1

	J       = diagm(-1 => d) + diagm(0 => f) + diagm(1 => d)
	v, u    = eigen(J)
	j       = sortperm(v)

	x		= v[j]
	w       = (fc*u[1,:].^2)
	w       = w[j]
	#w 		= squeeze(w,2)
	return x, w
end

function gauss_lag(n :: Int64,FUN,VarArgIn...)
	d       = -collect(1:n-1)
	f       = collect(1:2:2*n-1)
	fc      = 1

	J       = diagm(-1 => d) + diagm(0 => f) + diagm(1 => d)
	v, u    = eigen(J)
	j       = sortperm(v)

	x		= v[j]
	w       = (fc*u[1,:].^2)
	w       = w[j]
	
    int		= w'*apply(FUN,x,varargin{:});
    #w 		= squeeze(w,2)
	return x, w, int
end
