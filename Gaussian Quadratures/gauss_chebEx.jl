function gauss_cheb(a :: Real, b :: Real, n :: Int64);
	x	= (2*collect(1:n)' .- 1)*pi/(2*n)
	x	= cos.(x)
	w	= sqrt.(1 .- x.*x)
	x	= a .+ 0.5*(1 .+ x)*(b-a)
	return x', w'
end

function gauss_cheb(a :: Real, b :: Real,n :: Int64, f, VarArgIn...);
	x	= (2*collect(1:n)' .- 1)*pi/(2*n)
	x	= cos.(x)
	w	= sqrt.(1 .- x.*x)
	x	= a .+ 0.5*(1 .+ x)*(b-a)
	y	= feval(f,x,VarArgIn)
	int = pi*(b-a)*(y'*w)/(2*n)
	return x, w, int
end

x, w = gauss_cheb(-1,1,7)