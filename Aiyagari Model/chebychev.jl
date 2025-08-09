# Use this code to construct chebychev nodes.  x is a set of points in [-1,1] at which the n chebychev polynomials are to be evaluated

function chebychev(x :: Array, n :: Int64)
	X	= x[:]
	lx	= size(X,1)
	if n < 0
		println("n should be a positive integer")
	end
	if n == 0
	   Tx=[ones(lx,1)];
	elseif n == 1
	   Tx=[ones(lx,1) X];
	else
	   Tx=[ones(lx,1) X];
	   for i in 3:n+1
	      Tx=[Tx 2*X.*Tx[:,i-1]-Tx[:,i-2]]
	   end
	end
	return Tx
end