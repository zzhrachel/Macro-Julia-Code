#-------------------------------------------------------------------------------
# Compute the coefficients of weights and nodes of the Gauss-Hermite quadrature
#-------------------------------------------------------------------------------

function gauss_lag(n :: Int64)
	d       = -collect(1:n-1)
	f       = collect(1:2:2*n-1)
	fc      = 1

	J       = diagm(d,-1) + diagm(f) + diagm(d,1)
	v, u    = eig(J)
	j       = sortperm(v)

	x		= v[j]
	w       = (fc*u[1,:].^2)
	w       = w[j]
	return x, w
end

# function gauss_lag(n :: Int64,FUN,VarArgIn...)
# 	d       = -[1:n-1]
# 	f       = [1:2:2*n-1]
# 	fc      = 1

# 	J       = diagm(d,-1) + diagm(f) + diagm(d,1)
# 	v, u    = eig(J)
# 	j       = sortperm(v)
# 	x		= v[j]
# 	w       = (fc*u[1,:].^2)';
# 	w       = w[j];
	
#     int		=w'*apply(FUN,x,varargin{:});
#     w 		= squeeze(w,2)
# 	return x, w, int
# end

x, w = gauss_lag(16)

println(x)
println(w)