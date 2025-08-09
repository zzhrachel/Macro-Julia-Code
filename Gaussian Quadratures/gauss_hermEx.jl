#----------------------------------------------------------------------
# Compute the coefficients of Hermite polynomials using the recursion
#
# H(n+1)=2H(n)-2nH(n-1)
#
#----------------------------------------------------------------------

using Polynomials

function gauss_herm(n :: Int64)
	p0	= 1
	p1	= [2 ; 0]
	p	= zeros(n,1)
	for i in 1 : n-1
	   p	= 2*[p1 ; 0] - 2*i*[0 ; 0 ; p0]
	   p0	= p1
	   p1	= p
	end
	# Using the polynomial coefficients in p, construct the polynomial pp using the Polynomials package.  This will allow us to use the "roots()" command.
	# Note: Matlab defines the polynomial of [4,2,-1] as 4x^2 + 2 - 1 = 0 whereas the "Polynomials" package defines the polynomial on the same vector as
	# 4 + 2x - x^2 = 0.  As this code is a direct translation from a Matlab code, we have to flip the order of the polynomials in the vector p.
	p   = reverse(p,dims=1)
	pp	= Polynomial(p)
	# Compute the gauss-hermite polynomial roots:
	x	= sort(roots(pp))
	# Compute the weights imposing that integration is exact for lower order polynomials
	A	= zeros(n,n)
	A[1,:]	= ones(1,n)
	A[2,:]	= 2*x
	for i in 1 : n-2
	    A[i+2,:] = 2*x.*A[i+1,:]-2*i*A[i,:]
	end
	w	= A\[sqrt(pi) ; zeros(n-1,1)]
	w 	= dropdims(w,dims=2)
	return x, w
end

function gauss_herm(n :: Int64, FUN, VarArgIn...)
	p0	= 1
	p1	= [2 ; 0]
	p	= zeros(n,1)
	for i in 1 : n-1
	   p	= 2*[p1 ; 0] - 2*i*[0 ; 0 ; p0]
	   p0	= p1
	   p1	= p
	end
	# Using the polynomial coefficients in p, construct the polynomial pp using the Polynomials package.  This will allow us to use the "roots()" command.
	# Note: Matlab defines the polynomial of [4,2,-1] as 4x^2 + 2 - 1 = 0 whereas the "Polynomials" package defines the polynomial on the same vector as
	# 4 + 2x - x^2 = 0.  As this code is a direct translation from a Matlab code, we have to flip the order of the polynomials in the vector p.
	p   = reverse(p,dims=1)
	pp	= Polynomial(p)
	# Compute the gauss-hermite polynomial roots:
	x	= sort(roots(pp))
	# Compute the weights imposing that integration is exact for lower order polynomials
	A	= zeros(n,n)
	A[1,:]	= ones(1,n)
	A[2,:]	= 2*x
	for i in 1 : n-2;
	   A[i+2,:]	= 2*x.*A[i+1,:]-2*i*A[i,:]
	end
	w	= A\[sqrt(pi) ; zeros(n-1,1)]
	int = w'*apply(FUN,x,VarArgIn)
	w 	= dropdims(w,dims=2)
	return x, w, int
end

x, w = gauss_herm(6)